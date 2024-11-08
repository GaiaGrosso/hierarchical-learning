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


def compute_correlations(batch_start, batch_sample, dim):
    batch_start = batch_start.flatten(start_dim=1)  # [:, :dim*dim]
    batch_sample = batch_sample.flatten(start_dim=1)  # [:, :dim*dim]
    mean_start = batch_start.sum(dim=0)
    mean_sample = batch_sample.sum(dim=0)
    corr = batch_start.T @ batch_sample
    return corr, mean_start, mean_sample


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

        for channel in range(args.num_channels):
            logger.log(f"computing correlations for channel {channel}...")
            corr = th.zeros(dim * dim, dim * dim, device=device)
            m_start = th.zeros(dim * dim, device=device)
            m_sample = th.zeros(dim * dim, device=device)
            evaluated_samples = 0

            time_start = time.time()
            while evaluated_samples < num_samples:
                batch_start, extra_start = next(data_start)
                batch_sample, extra_sample = next(data_sample)
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

                C_batch, m_start_batch, m_sample_batch = compute_correlations(
                    batch_start[:, channel, :, :], batch_sample[:, channel, :, :], dim
                )

                corr += C_batch
                m_start += m_start_batch
                m_sample += m_sample_batch

                del C_batch, m_start_batch, m_sample_batch

                evaluated_samples += batch_start.shape[0]

                logger.log(
                    f"evaluated {evaluated_samples} samples in {time.time() - time_start:.1f} seconds"
                )

            corr = corr / evaluated_samples
            m_start = m_start / evaluated_samples
            m_sample = m_sample / evaluated_samples

            corr = corr.reshape(dim, dim, dim, dim)
            m_start = m_start.reshape(dim, dim)
            m_sample = m_sample.reshape(dim, dim)

            # Save correlations

            outfile = os.path.join(
                args.output,
                f"correlations-t_{time_step}_{args.timestep_respacing}-channel{channel}.pk",
            )
            logger.log(f"saving correlations to {outfile}")

            with open(outfile, "wb") as handle:
                pickle.dump(
                    {
                        "corr": corr.cpu().numpy(),
                        "m_start": m_start.cpu().numpy(),
                        "m_sample": m_sample.cpu().numpy(),
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
        num_channels=3,
        timestep_respacing="250",
        starting_data_dir="datasets/ILSVRC2012/validation",
        data_dir=os.path.join(os.getcwd(), "results", "diffused_ILSVRC2012_validation"),
        output=os.path.join(os.getcwd(), "correlations_measurements"),
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
