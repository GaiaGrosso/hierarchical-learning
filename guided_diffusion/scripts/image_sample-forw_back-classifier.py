"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

# from torchvision.models.feature_extraction import create_feature_extractor
# from functools import partial

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_classifier,
    classifier_defaults,
)
from guided_diffusion.image_datasets import load_data
import datetime
import pickle
import copy
from guided_diffusion.torch_classifiers import load_classifier

# import torch.fx
# torch.fx.wrap('len')

# def make_feature_extractor(model, return_nodes):
    # return create_feature_extractor(model, return_nodes)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.output)

    logger.log("creating data loader...")
    data_start = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
        class_cond=True,
        random_crop=False,
        random_flip=False,
    )

    all_images = []
    all_logits_samples = []
    all_labels = []
    all_start_images = []
    all_logits_start = []
    all_noisy_images = []
    all_logits_noisy = []
    dict_list = []
    while len(all_images) * args.batch_size < args.num_samples:
        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        batch_start, extra = next(data_start)
        labels_start = extra["y"].to(dist_util.dev())
        batch_start = batch_start.to(dist_util.dev())
        # class_eval_start = class_eval(batch_start)

        logger.log("sampling...")
        # Sample noisy images from the diffusion process at time t_reverse given by the step_reverse argument
        t_reverse = diffusion._scale_timesteps(th.tensor([args.step_reverse])).to(dist_util.dev())
        batch_noisy = diffusion.q_sample(batch_start, t_reverse)
        # class_eval_noisy = class_eval(batch_noisy, t_reverse)

        model_kwargs = {}
        if args.class_cond:
            classes = labels_start # Condition the diffusion on the labels of the original images
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop_forw_back if not args.use_ddim else diffusion.ddim_sample_loop_forw_back
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            step_reverse = args.step_reverse,  # step when to reverse the diffusion process
            noise=batch_noisy,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        del model, diffusion

        ##Classifier part
        logger.log("loading classifier...")
        # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        # classifier.load_state_dict(
            # dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        # )
        classifier, preprocess, module_names = load_classifier()
        classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def class_eval(x, t=0.0):
            shape = x.shape
            time = th.tensor([t] * shape[0], device=dist_util.dev())
            with th.no_grad():
                # logits = classifier(x, time)
                logits = classifier(preprocess(x))
                # log_probs = F.log_softmax(logits, dim=-1)
                return logits

        # get_name_modules = lambda model: [name for name, _ in model.named_modules()]
        # modules = get_name_modules(classifier)
        # print(modules)
        # module_names = ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'fc']
        # module_names = [f'input_blocks.{i}' for i in range(18)] + [f'middle_block.{i}' for i in range(3)] + [f'out.{i}' for i in range(3)]
        # partial(classifier)
        # feature_extractor = make_feature_extractor(classifier, return_nodes=module_names)
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        hooks = []
        for layer_name in module_names:
            layer = dict([*classifier.named_modules()])[layer_name]
            hook = layer.register_forward_hook(get_activation(layer_name))
            hooks.append(hook)

        logger.log("evaluating classifier...")
        class_eval_start = class_eval(batch_start)
        activations_start = copy.deepcopy(activations)
        class_eval_noisy = class_eval(batch_noisy, t_reverse)
        # activations_noisy = copy.deepcopy(activations)
        class_eval_sample = class_eval(sample, 0.0)
        activations_sample = copy.deepcopy(activations)

        diff_activations = {}
        cosine_sim = th.nn.CosineSimilarity(dim=1, eps=1e-8)
        for key in activations_start.keys():
            diff_activations[key] = {}
            activations_sample[key] = activations_sample[key].flatten(start_dim=1)
            activations_start[key]  = activations_start[key].flatten(start_dim=1)
            diff_activations[key]['L2'] = th.linalg.norm(activations_sample[key] - activations_start[key], dim=1)**2
            diff_activations[key]['L2_normalized'] = diff_activations[key]['L2'] / (th.linalg.norm(activations_sample[key], dim=1) * th.linalg.norm(activations_start[key], dim=1))
            diff_activations[key]['cosine'] = cosine_sim(activations_sample[key], activations_start[key])

        # print(activations_start.keys())
        # print(activations_start['input_blocks.0'])
        for hook in hooks: hook.remove()

        # t_feat = th.tensor([0.0] * batch_start.shape[0], device=dist_util.dev())
        # print(feature_extractor((batch_start, t_feat)))

        del classifier

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        class_eval_sample = class_eval_sample.contiguous()
        gathered_logits_samples = [
                th.zeros_like(class_eval_sample) for _ in range(dist.get_world_size())
            ]
        dist.all_gather(gathered_logits_samples, class_eval_sample)
        all_logits_samples.extend([logits.cpu().numpy() for logits in gathered_logits_samples])
        
        # if args.class_cond:
        gathered_labels = [
            th.zeros_like(labels_start) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, labels_start)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        # Save the start images
        batch_start = ((batch_start + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch_start = batch_start.permute(0, 2, 3, 1)
        batch_start = batch_start.contiguous()
        gathered_start_samples = [th.zeros_like(batch_start) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_start_samples, batch_start)  # gather not supported with NCCL
        all_start_images.extend([sample.cpu().numpy() for sample in gathered_start_samples])
        
        class_eval_start = class_eval_start.contiguous()
        gathered_logits_start = [
                th.zeros_like(class_eval_start) for _ in range(dist.get_world_size())
            ]
        dist.all_gather(gathered_logits_start, class_eval_start)
        all_logits_start.extend([logits.cpu().numpy() for logits in gathered_logits_start])
        
        # Save the noised images
        batch_noisy = ((batch_noisy + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch_noisy = batch_noisy.permute(0, 2, 3, 1)
        batch_noisy = batch_noisy.contiguous()
        gathered_noisy_samples = [th.zeros_like(batch_noisy) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_noisy_samples, batch_noisy)  # gather not supported with NCCL
        all_noisy_images.extend([sample.cpu().numpy() for sample in gathered_noisy_samples])

        class_eval_noisy = class_eval_noisy.contiguous()
        gathered_logits_noisy = [
                th.zeros_like(class_eval_noisy) for _ in range(dist.get_world_size())
            ]
        dist.all_gather(gathered_logits_noisy, class_eval_noisy)
        all_logits_noisy.extend([logits.cpu().numpy() for logits in gathered_logits_noisy])

        dict_list.append(diff_activations)

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # Save the activations
    dictionary_act = {}
    for key in dict_list[0].keys():
        dictionary_act[key] = {}
        for key2 in dict_list[0][key].keys():
            dictionary_act[key][key2] = th.cat([dict_list[i][key][key2] for i in range(len(dict_list))], dim=0).cpu().numpy()

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    logits_arr = np.concatenate(all_logits_samples, axis=0)
    logits_arr = logits_arr[: args.num_samples]
    # if args.class_cond:
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]

    arr_start = np.concatenate(all_start_images, axis=0)
    arr_start = arr_start[: args.num_samples]
    logits_arr_start = np.concatenate(all_logits_start, axis=0)
    logits_arr_start = logits_arr_start[: args.num_samples]
    arr_noisy = np.concatenate(all_noisy_images, axis=0)
    arr_noisy = arr_noisy[: args.num_samples]
    logits_arr_noisy = np.concatenate(all_logits_noisy, axis=0)
    logits_arr_noisy = logits_arr_noisy[: args.num_samples]
    if dist.get_rank() == 0:
        # Save the arguments of the run
        out_args = os.path.join(logger.get_dir(), f"args_{args.step_reverse}.pk")
        logger.log(f"saving args to {out_args}")
        with open(out_args, 'wb') as handle: pickle.dump(args, handle)
        # Save the data of the run
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{args.step_reverse}.npz")
        logger.log(f"saving to {out_path}")
        # if args.class_cond:
        np.savez(out_path, arr, label_arr, arr_start, arr_noisy, logits_arr, logits_arr_start, logits_arr_noisy)
        # else:
        #     np.savez(out_path, arr, arr_start, arr_noisy, logits_arr, logits_arr_start, logits_arr_noisy)
        out_act = os.path.join(logger.get_dir(), f"acts_{args.step_reverse}.pk")
        logger.log(f"saving activations to {out_act}")
        with open(out_act, 'wb') as handle: pickle.dump(dictionary_act, handle)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults.update(dict(
        step_reverse = 100,
        classifier_path = 'models/64x64_classifier.pt',
        data_dir =  'datasets/imagenet64_startingImgs',
        output  =  os.path.join(os.getcwd(),
             'results',
             'forw_back',
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")),
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
