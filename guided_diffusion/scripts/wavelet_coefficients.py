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
sys.path.insert(0, '/home/sclocchi')
from local_probability_models_of_images.code.wavelet_func import multi_scale_decompose


def compute_correlations(delta_x):
    delta_x  = delta_x.flatten(start_dim=1)
    mean = delta_x.sum(dim=0)
    corr = delta_x.T @ delta_x
    return corr, mean


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

    if args.spin_like:
        logger.log("Computing correlations for spin-like data")

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

        data_sample = load_data(
            data_dir=sample_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            deterministic=True,
            class_cond=True,
            random_crop=False,
            random_flip=False,
            list_images=list_sample_imgs,
            drop_last=False,  # It is important when batch_size < num_samples, otherwise it doesn't yield
        )

        data_start = load_data(
            data_dir=args.starting_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            deterministic=True,
            class_cond=True,
            random_crop=False,
            random_flip=False,
            list_images=list_start_imgs,
            drop_last=False,  # It is important when batch_size < num_samples, otherwise it doesn't yield
        )

        dim = args.image_size
        scales = 8

        mean_ck0 = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]
        mean_ckt = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]
        mean_abs_ck0 = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]
        mean_abs_ckt = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]
        mean_norm2_ck0 = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]
        mean_norm2_ckt = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]
        mean_delta_ck = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]
        mean_abs_delta_ck = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]
        mean_norm2_delta_ck = [th.zeros(12, dim//2**(j+1), dim//2**(j+1), device=device) for j in range(scales)]

        # mean_abs_delta_ck = th.zeros(dim, dim, device=device)
        # mean_delta_ck = th.zeros(dim, dim, device=device)
        # mean_abs_ck0 = th.zeros(dim, dim, device=device)
        # mean_ck0 = th.zeros(dim, dim, device=device)
        # mean_delta_ck_over_ck0 = th.zeros(dim, dim, device=device)
        evaluated_samples = 0

        time_start = time.time()
        while evaluated_samples < num_samples:
            batch_start, extra_start = next(data_start)     # B, C, H, W
            batch_sample, extra_sample = next(data_sample)  # B, C, H, W
            if evaluated_samples == 0:
                print(f"batch shape: {batch_start.shape}")

            # labels_start = extra["y"].to(dist_util.dev())
            batch_start = batch_start.to(device)
            batch_sample = batch_sample.to(device)
            start_names = extra_start["img_name"]
            sample_names = extra_sample["img_name"]
            assert check_same_images(
                sample_names, start_names
            ), "Images in sample and starting batches must be the same"

            ck0 = multi_scale_decompose(batch_start, J = 8, device=batch_start.device)   # list(J) of B, C, H, W
            ckt = multi_scale_decompose(batch_sample, J = 8, device=batch_sample.device) # list(J) of B, C, H, W

            
            for j in range(8):
                ck0j = ck0[j]
                cktj = ckt[j]
                mean_ck0[j] += ck0j.sum(dim=(0))
                mean_ckt[j] += cktj.sum(dim=(0))
                mean_delta_ck[j] += (cktj - ck0j).sum(dim=(0))
                mean_abs_ck0[j] += ck0j.abs().sum(dim=(0))
                mean_abs_ckt[j] += cktj.abs().sum(dim=(0))
                mean_abs_delta_ck[j] += (cktj - ck0j).abs().sum(dim=(0))
                mean_norm2_ck0[j] += (ck0j**2).sum(dim=(0))
                mean_norm2_ckt[j] += (cktj**2).sum(dim=(0))
                mean_norm2_delta_ck[j] += ((cktj - ck0j)**2).sum(dim=(0))
                # mean_abs_delta_ck[j] += ((cktj - ck0j)**2)[:,3:,:,:].sum(dim=(0,1))

            evaluated_samples += batch_start.shape[0]

            logger.log(
                f"evaluated {evaluated_samples} samples in {time.time() - time_start:.1f} seconds"
            )

        # Compute mean values
        for j in range(8):
            mean_ck0[j] /= evaluated_samples
            mean_ckt[j] /= evaluated_samples
            mean_delta_ck[j] /= evaluated_samples
            mean_abs_ck0[j] /= evaluated_samples
            mean_abs_ckt[j] /= evaluated_samples
            mean_abs_delta_ck[j] /= evaluated_samples
            mean_norm2_ck0[j] /= evaluated_samples
            mean_norm2_ckt[j] /= evaluated_samples
            mean_norm2_delta_ck[j] /= evaluated_samples

        # Save Fourier coefficients
        outfile = os.path.join(
                args.output,
                f"wavelet_coeff-t_{time_step}_{args.timestep_respacing}-complete.pk",
            )
        logger.log(f"saving Fourier coefficients to {outfile}")

        with open(outfile, "wb") as handle:
            pickle.dump(
                {   
                    "mean_ck0": [mean_ck0[j].cpu().numpy() for j in range(8)],
                    "mean_ckt": [mean_ckt[j].cpu().numpy() for j in range(8)],
                    "mean_delta_ck": [mean_delta_ck[j].cpu().numpy() for j in range(8)],
                    "mean_abs_ck0": [mean_abs_ck0[j].cpu().numpy() for j in range(8)],
                    "mean_abs_ckt": [mean_abs_ckt[j].cpu().numpy() for j in range(8)],
                    "mean_abs_delta_ck": [mean_abs_delta_ck[j].cpu().numpy() for j in range(8)],
                    "mean_norm2_ck0": [mean_norm2_ck0[j].cpu().numpy() for j in range(8)],
                    "mean_norm2_ckt": [mean_norm2_ckt[j].cpu().numpy() for j in range(8)],
                    "mean_norm2_delta_ck": [mean_norm2_delta_ck[j].cpu().numpy() for j in range(8)],
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
        data_dir=os.path.join(os.getcwd(), "results", "diffused_ILSVRC2012_validation"),
        output=os.path.join(os.getcwd(), "Wavelet_coefficients"),
        spin_like=False,
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
