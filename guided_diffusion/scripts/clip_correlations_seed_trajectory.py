import argparse
import os
import torch as th
import pickle
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import (
    load_data,
    _list_starting_images,
    _list_image_files_recursively,
)
import time
import sys
import numpy as np
import clip


def clip_embedding(visual, x: th.Tensor):
    '''Compute the CLIP embedding of the input token sequence x.

    Args:
        visual (nn.Module): Visual backbone of the CLIP model.
        x (torch.Tensor): Input token sequence x of shape (B, C, 224, 224).

    Returns:
        x (torch.Tensor): Output token sequence x of shape (B, 7, 7, 768).
        class_token (torch.Tensor): Class token of shape (B, 512).
    '''

    with th.no_grad():
        x = x.type(th.cuda.HalfTensor)
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = th.cat([visual.class_embedding.to(x.dtype) + th.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        class_token = visual.ln_post(x[:, 0, :])

        class_token = class_token @ visual.proj

        x = x[:, 1:, :]
        x = x.reshape(x.shape[0], 7, 7, x.shape[2])

        return x, class_token


def compute_dot_correlations(x, y):
    '''Compute the correlation matrix and the mean of the input token sequences x and y.
    
    Args:
        x (torch.Tensor): Input token sequence x of shape (B, H, W, C).
        y (torch.Tensor): Input token sequence y of shape (B, H, W, C).
    '''
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    y = y.reshape(y.shape[0], -1, y.shape[-1])

    xy = th.einsum('bic,bjc->ij', x, y) / x.shape[-1]
    x = x.sum(dim=0)
    y = y.sum(dim=0)

    return xy, x, y


def compute_delta_correlations(x, y):
    '''Compute the correlation matrix and the mean of the input token sequences x and y.
    
    Args:
        x (torch.Tensor): Input token sequence x of shape (B, H, W, C).
        y (torch.Tensor): Input token sequence y of shape (B, H, W, C).
    '''
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    y = y.reshape(y.shape[0], -1, y.shape[-1])

    delta = x - y

    deltaC = th.einsum('bic,bjc->ij', delta, delta) / x.shape[-1]
    deltam = x.sum(dim=0)

    return deltaC, deltam

def compute_deltaMagn_correlations(x, y):
    '''Compute the correlation matrix and the mean of the input token sequences x and y.
    
    Args:
        x (torch.Tensor): Input token sequence x of shape (B, H, W, C).
        y (torch.Tensor): Input token sequence y of shape (B, H, W, C).
    '''
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    y = y.reshape(y.shape[0], -1, y.shape[-1])

    delta = th.linalg.norm(x - y, dim=-1) / x.shape[-1]**.5

    deltaC = th.einsum('bi,bj->ij', delta, delta)
    deltam = delta.sum(dim=0)

    return deltaC, deltam


def compute_cos_correlations(x, y):
    '''Compute the correlation matrix and the mean of the input token sequences x and y.
    
    Args:
        x (torch.Tensor): Input token sequence x of shape (B, H, W, C).
        y (torch.Tensor): Input token sequence y of shape (B, H, W, C).
    '''
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    x = x / th.linalg.norm(x, dim=-1, keepdim=True)
    y = y.reshape(y.shape[0], -1, y.shape[-1])
    y = y / th.linalg.norm(y, dim=-1, keepdim=True)

    xy = th.einsum('bic,bjc->ij', x, y)
    x = x.sum(dim=0)
    y = y.sum(dim=0)

    return xy, x, y


def compute_correlation_function(size, C):
    size2 = size*size

    idx = th.triu_indices(size2, size2, offset=0, device=C.device)
    Cvalues = th.tensor(C[idx[0], idx[1]])

    print('Computing distances...', flush=True)
    labels = ((idx[1]//size) - (idx[0]//size))**2 
    labels += ((idx[1]%size) - (idx[0]%size))**2

    time_0 = time.time()
    nested_list = [[i**2 + j**2 for j in range(i+1)] for i in range(size)]
    unique_labels = set([item for sublist in nested_list for item in sublist])
    map_dist_index = {}
    for idx, lab in enumerate(unique_labels):
        map_dist_index[lab] = idx
    # unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    time_1 = time.time()
    print(f'Computed unique labels. It took {time_1-time_0:.2f} s.', flush=True)

    res = th.zeros(len(unique_labels), dtype=th.float, device=C.device)
    labels_count = th.zeros(len(unique_labels), dtype=th.int, device=C.device)
    count = 0
    time_0 = time.time()
    print('start loop...', flush=True)

    batch_size = 1000
    for ii in range(len(Cvalues)//batch_size):
        indeces = th.tensor([map_dist_index[lab.item()] for lab in labels[ii*batch_size:(ii+1)*batch_size]], device=C.device)
        res = res.scatter_add_(0, indeces, Cvalues[ii*batch_size:(ii+1)*batch_size])
        labels_count = labels_count.scatter_add_(0, indeces, th.ones_like(indeces, dtype=th.int))

        count += 1
        if count%1000==0:
            print(f'Computed {count}. Time{time.time()-time_0:.2f}', flush=True)
    # res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, indeces, Cvalues)

    count *= batch_size
    if count < len(Cvalues):
        indeces = th.tensor([map_dist_index[lab.item()] for lab in labels[count:]], device=C.device)
        res = res.scatter_add_(0, indeces, Cvalues[count:])
        labels_count = labels_count.scatter_add_(0, indeces, th.ones_like(indeces, dtype=th.int))

    res = res / labels_count.float()
    print('Computed correlation function.', flush=True)

    # with open(args.output, 'wb') as f:
    #     pickle.dump({'labels': torch.tensor([dist**0.5 for dist in unique_labels]).cpu().numpy(), 'correlation_function': res.cpu().numpy(), 'susceptibility': Cvalues.sum().cpu().numpy()}, f)

    print('Done.', flush=True)

    return th.tensor([dist**0.5 for dist in unique_labels]), res, Cvalues.sum()



def check_same_images(list_sample, list_start):
    for i in range(len(list_sample)):
        name_sample = list_sample[i].split(".")[0][:-6]
        name_start = list_start[i].split(".")[0]
        if name_sample != name_start:
            return False
    return True


def main():
    args = create_argparser().parse_args()
    args.output = os.path.join(args.output, args.data_dir.split("/")[-1])

    dist_util.setup_dist()
    device = dist_util.dev() if args.device is None else th.device(args.device)
    print(f"device: {device}")
    logger.configure(dir=args.output)

    model, preprocess = clip.load("ViT-B/32", device=device)

    # Time steps to evaluate
    for time_step in args.time_series:

        # Load starting data
        name_at_time_step = f"t_{time_step}_{args.timestep_respacing}_images"
        sample_data_dir = os.path.join(args.data_dir, name_at_time_step)

        logger.log("creating data loader...")
        list_sample_imgs = _list_image_files_recursively(sample_data_dir)
        list_start_imgs = _list_starting_images(args.starting_data_dir, sample_data_dir)

        num_samples = len(list_sample_imgs)
        num_start = len(list_start_imgs)
        assert (
            num_samples == num_start
        ), "Number of samples and starting images must be the same"

        assert (
            "-seed_0" in sample_data_dir
        ), "The sample folder must be that with seed 0"

        data_corr = {}
        for i in range(num_samples):
            logger.log(f"Image {i+1} / {num_samples}")
            start_img = list_start_imgs[i]
            sample_parts = list_sample_imgs[i].split("-seed_0")

            seeds_sample_imgs = [sample_parts[0] + f"-seed_{se}" + sample_parts[1] for se in range(args.num_seeds)]
            seeds_start_imgs = [start_img] * args.num_seeds

            data_sample = load_data(
                data_dir=sample_data_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                deterministic=True,
                class_cond=True,
                random_crop=False,
                random_flip=False,
                list_images=seeds_sample_imgs,
                drop_last=False,  # It is important when batch_size < num_samples, otherwise it doesn't yield
                preprocess_fn=preprocess,
            )

            data_start = load_data(
                data_dir=args.starting_data_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                deterministic=True,
                class_cond=True,
                random_crop=False,
                random_flip=False,
                list_images=seeds_start_imgs,
                drop_last=False,  # It is important when batch_size < num_samples, otherwise it doesn't yield
                preprocess_fn=preprocess,
            )

            dim = 7
            embed_dim = 768

            delta_corr = th.zeros(dim * dim, dim * dim, device=device)
            mean_delta = th.zeros(dim * dim, embed_dim, device=device)
            deltaMgn_corr = th.zeros(dim * dim, dim * dim, device=device)
            mean_deltaMgn = th.zeros(dim * dim, device=device)
            evaluated_trajectories = 0

            time_start = time.time()
            while evaluated_trajectories < len(seeds_sample_imgs):
                batch_start, extra_start = next(data_start)     # B, C, H, W
                batch_sample, extra_sample = next(data_sample)  # B, C, H, W
                if evaluated_trajectories == 0:
                    print(f"batch shape: {batch_start.shape}")

                # labels_start = extra["y"].to(dist_util.dev())
                batch_start = batch_start.to(device)
                batch_sample = batch_sample.to(device)
                start_names = extra_start["img_name"]
                sample_names = extra_sample["img_name"]
                assert check_same_images(
                    sample_names, start_names
                ), "Images in sample and starting batches must be the same"

                start_x, start_feat = clip_embedding(model.visual, batch_start)
                sample_x, sample_feat = clip_embedding(model.visual, batch_sample)

                # Cdot_batch, m0_batch, mt_batch = compute_dot_correlations(start_x, sample_x)
                Cdelta_batch, mdelta_batch = compute_delta_correlations(start_x, sample_x)
                CdeltaMgn_batch, mdeltaMgn_batch = compute_deltaMagn_correlations(start_x, sample_x)
                # Ccos_bacth, mcos0_batch, mcost_batch = compute_cos_correlations(start_x, sample_x)

                # dot_corr += Cdot_batch
                # mean0 += m0_batch
                # meant += mt_batch
                delta_corr += Cdelta_batch
                mean_delta += mdelta_batch
                deltaMgn_corr += CdeltaMgn_batch
                mean_deltaMgn += mdeltaMgn_batch
                # cos_corr += Ccos_bacth
                # meancos0 += mcos0_batch
                # meancost += mcost_batch

                # del Cdot_batch, m0_batch, mt_batch, Cdelta_batch, mdelta_batch, CdeltaMgn_batch, mdeltaMgn_batch, Ccos_bacth, mcos0_batch, mcost_batch
                del Cdelta_batch, mdelta_batch, CdeltaMgn_batch, mdeltaMgn_batch

                evaluated_trajectories += batch_start.shape[0]

                logger.log(
                    f"evaluated {evaluated_trajectories} samples in {time.time() - time_start:.1f} seconds"
                )
            
            delta_corr = delta_corr / evaluated_trajectories
            mean_delta = mean_delta / evaluated_trajectories
            delta_corr = delta_corr - mean_delta @ mean_delta.T / mean_delta.shape[-1]
            delta_dist, delta_Cfun, delta_chi =compute_correlation_function(dim, delta_corr)

            deltaMgn_corr = deltaMgn_corr / evaluated_trajectories
            mean_deltaMgn = mean_deltaMgn / evaluated_trajectories
            deltaMgn_corr = deltaMgn_corr - mean_deltaMgn[:,None] @ mean_deltaMgn[None, :]
            deltaMgn_dist, deltaMgn_Cfun, deltaMgn_chi =compute_correlation_function(dim, deltaMgn_corr)

            delta_corr = delta_corr.reshape(dim, dim, dim, dim)
            mean_delta = mean_delta.reshape(dim, dim, embed_dim)

            deltaMgn_corr = deltaMgn_corr.reshape(dim, dim, dim, dim)
            mean_deltaMgn = mean_deltaMgn.reshape(dim, dim)

            # Save data per starting image
            data_corr[i] = {"delta_Cfun": delta_Cfun.cpu().numpy(),
                        "delta_chi": delta_chi.cpu().numpy(),
                        "deltaMgn_Cfun": deltaMgn_Cfun.cpu().numpy(),
                        "deltaMgn_chi": deltaMgn_chi.cpu().numpy(),
                        "sample_features": sample_feat.cpu().numpy(),
                        # "delta_dist": delta_dist.cpu().numpy(),
                        # "deltaMgn_dist": deltaMgn_dist.cpu().numpy(),
                        }

        # Average correlation functions over starting images
        mean_delta_Cfun = np.array([data_corr[kk]["delta_Cfun"] for kk in data_corr.keys()]).mean(axis=0)
        mean_deltaMgn_Cfun = np.array([data_corr[kk]["deltaMgn_Cfun"] for kk in data_corr.keys()]).mean(axis=0)
        mean_delta_chi = np.array([data_corr[kk]["delta_chi"] for kk in data_corr.keys()]).mean()
        mean_deltaMgn_chi = np.array([data_corr[kk]["deltaMgn_chi"] for kk in data_corr.keys()]).mean()

        # Save correlations
        outfile = os.path.join(
                args.output,
                f"clip_correlation_seeds-t_{time_step}_{args.timestep_respacing}.pk",
            )
        logger.log(f"saving clip correlations to {outfile}")

        with open(outfile, "wb") as handle:
            pickle.dump(
                {   "delta_dist": delta_dist.cpu().numpy(),
                    "delta_Magn_dist": deltaMgn_dist.cpu().numpy(),
                    "mean_delta_Cfun": mean_delta_Cfun,
                    "mean_deltaMgn_Cfun": mean_deltaMgn_Cfun,
                    "mean_delta_chi": mean_delta_chi,
                    "mean_deltaMgn_chi": mean_deltaMgn_chi,
                    "seed_correlations": data_corr,
                },
                handle,
            )

    ## End of time steps loop

    dist.barrier()
    logger.log("Done!")


def create_argparser():
    defaults = dict(
        device=None,
        # num_samples=10000,
        batch_size=128,
        image_size=256,
        timestep_respacing="250",
        starting_data_dir="datasets/ILSVRC2012/validation",
        data_dir=os.path.join(os.getcwd(), "results", "diffused_ILSVRC2012_validation-seed_0"),
        output=os.path.join(os.getcwd(), "Clip_correlations"),
        spin_like=False,
        num_seeds=128
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--time_series",
        nargs="+",
        type=int,
        default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
        help="Time steps to evaluate. Pass like: --time_series 25 50 100",
    )
    return parser


if __name__ == "__main__":
    main()
