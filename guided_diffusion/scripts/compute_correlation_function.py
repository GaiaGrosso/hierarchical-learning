import torch
import numpy as np
import pickle
import argparse
import os
import time

from guided_diffusion.script_util import (
    add_dict_to_argparser,
)


def create_argparser():
    defaults = dict(
        device=None,
        # num_samples=10000,
        image_size=256,
        unique_labels = None,
        data_prefix = '/scratch/sclocchi/guided-diffusion/correlations_measurements/diffused_ILSVRC2012_validation/correlations_deltaX-t_',
        data_suffix = '_250-magnitude.pk',
        output=os.path.join(os.getcwd(), "correlations_measurements"),
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--time_series",
        nargs="+",
        type=int,
        # default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
        default=[100],
        help="Time steps to evaluate. Pass like: --time_series 25 50 100",
    )
    return parser


def run_compute_labels(args):
    size2 = args.image_size * args.image_size
    idx = torch.triu_indices(size2, size2, offset=0)
    labels = ((idx[1]//args.image_size) - (idx[0]//args.image_size))**2 + ((idx[1]%args.image_size) - (idx[0]%args.image_size))**2
    file_labels = f'/scratch/sclocchi/guided-diffusion/correlations_measurements/labels_{args.image_size}.pt'
    torch.save({'labels': labels}, file_labels)
    print('Done.', flush=True)


def run_compute_mean(args):
    # size2 = args.image_size * args.image_size
    # file = '/scratch/sclocchi/guided-diffusion/correlations_measurements/diffused_ILSVRC2012_validation/correlations_deltaX-t_100_250-magnitude.pk'
    print("Loading data...", flush=True)
    with open(args.file, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded.", flush=True)

    print('Computing mean...', flush=True)
    mean = data['mean'].mean()
    print('Computed mean.', flush=True)
    file_mean = args.file[:-3] + '_mean.pt'
    torch.save({'mean': mean}, file_mean)
    print('Done.', flush=True)


def run_correlation_function(args):
    size2 = args.image_size * args.image_size
    # file = '/scratch/sclocchi/guided-diffusion/correlations_measurements/diffused_ILSVRC2012_validation/correlations_deltaX-t_100_250-magnitude.pk'
    print("Loading data...", flush=True)
    with open(args.file, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded.", flush=True)

    print('Computing correlation matrix...', flush=True)
    C = data['corr'].reshape(size2, size2) - data['mean'].reshape(size2, 1) @ data['mean'].reshape(1, size2)
    print('Computed correlation matrix.', flush=True)

    print('Deleting data...', flush=True)
    del data
    print('Deleted data.', flush=True)

    print('Computing indeces...', flush=True)
    idx = torch.triu_indices(size2, size2, offset=0)
    print('Computed indeces.', flush=True)

    print('Computing C values...', flush=True)
    Cvalues = torch.tensor(C[idx[0], idx[1]])
    print('Computed C values.', flush=True)

    print('Deleting C...', flush=True)
    del C
    print('Deleted C.', flush=True)

    print('Computing distances...', flush=True)
    # labels = ((idx[1]//args.image_size) - (idx[0]//args.image_size))**2 
    # labels += ((idx[1]%args.image_size) - (idx[0]%args.image_size))**2
    file_labels = f'/scratch/sclocchi/guided-diffusion/correlations_measurements/labels_{args.image_size}.pt'
    lab_dic = torch.load(file_labels)
    labels = lab_dic['labels']
    del lab_dic
    print('Computed distances.', flush=True)

    if args.unique_labels is not None:
        lab_dic = torch.load(f)
        unique_labels = lab_dic['labels']
        labels_count = lab_dic['labels_count']
    else:
        print('Computing unique labels...', flush=True)
        time_0 = time.time()
        nested_list = [[i**2 + j**2 for j in range(i+1)] for i in range(args.image_size)]
        unique_labels = set([item for sublist in nested_list for item in sublist])
        map_dist_index = {}
        for idx, lab in enumerate(unique_labels):
            map_dist_index[lab] = idx
        # unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        time_1 = time.time()
        print(f'Computed unique labels. It took {time_1-time_0:.2f} s.', flush=True)

        # file_labels = f'../correlations_measurements/labels_{args.image_size}.pt'
        # torch.save({'labels': unique_labels, 'labels_count': labels_count}, file_labels)


    print('Computing correlation function...', flush=True)
    res = torch.zeros(len(unique_labels), dtype=torch.float)
    labels_count = torch.zeros(len(unique_labels), dtype=torch.int)
    count = 0
    time_0 = time.time()
    print('start loop...', flush=True)

    # for ii in range(len(Cvalues)):
    #     res[map_dist_index[labels[ii].item()]] += Cvalues[ii]
    #     labels_count[map_dist_index[labels[ii].item()]] += 1
    #     count += 1
    #     if count%1000000==0:
    #         print(f'Computed {count}. Time{time.time()-time_0:.2f}', flush=True)

    batch_size = 1000
    for ii in range(len(Cvalues)//batch_size):
        indeces = torch.tensor([map_dist_index[lab.item()] for lab in labels[ii*batch_size:(ii+1)*batch_size]])
        res = res.scatter_add_(0, indeces, Cvalues[ii*batch_size:(ii+1)*batch_size])
        labels_count = labels_count.scatter_add_(0, indeces, torch.ones_like(indeces, dtype=torch.int))

        count += 1
        if count%1000==0:
            print(f'Computed {count}. Time{time.time()-time_0:.2f}', flush=True)
    # res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, indeces, Cvalues)

    count *= batch_size
    if count < len(Cvalues):
        indeces = torch.tensor([map_dist_index[lab.item()] for lab in labels[count:]])
        res = res.scatter_add_(0, indeces, Cvalues[count:])
        labels_count = labels_count.scatter_add_(0, indeces, torch.ones_like(indeces, dtype=torch.int))

    res = res / labels_count.float()
    print('Computed correlation function.', flush=True)

    with open(args.output, 'wb') as f:
        pickle.dump({'labels': torch.tensor([dist**0.5 for dist in unique_labels]).cpu().numpy(), 'correlation_function': res.cpu().numpy(), 'susceptibility': Cvalues.sum().cpu().numpy()}, f)

    print('Done.', flush=True)


def main():
    args = create_argparser().parse_args()

    # run_compute_labels(args)
    for time in args.time_series:
        args.file = args.data_prefix + str(time) + args.data_suffix
        args.output = args.file[:-3] + '_correlation_function.pk'
        run_correlation_function(args)
        # run_compute_mean(args)

    print('All done.', flush=True)
    

if __name__ == "__main__":
    main()
