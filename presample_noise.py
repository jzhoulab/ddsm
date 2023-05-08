import argparse
import os.path

import torch

from ddsm import noise_factory

def parse_args():
    parser = argparse.ArgumentParser("Pre generate jacobi process values with specified number of "
                                     "categories and time points")

    parser.add_argument("-n", "--num_samples", type=int,
                        help="Number of the different samples pre generated (default = 100000)",
                        default=100000) 
    parser.add_argument("-c", "--num_cat", type=int,
                        help="Number of categories", required=True) 
    parser.add_argument("-t", '--num_time_steps', type=int,
                        help="Number of time steps between <min_time> and <max_time> (default = 400)",
                        default=400) 
    parser.add_argument("--speed_balance", action='store_true',
                        help="Adding speed balance to Jacobi Process")
    parser.add_argument("--max_time", type=float,
                        help="Last time point (default = 4.0)",
                        default=4.0)
    parser.add_argument("--out_path", type=str,
                        help="Path to output directory, where precomputed noise will be saved",
                        default=".")
    parser.add_argument("--order", type=int,
                        help="Order of Jacobi polynomials. It affects precision of the noise overall (default = 1000)",
                        default=1000)
    parser.add_argument("--steps_per_tick", type=int,
                        help="Number of steps per time tick. One tick is (<max_time> - <min_time>) / num_time_steps "
                             "(default = 200)",
                        default=200)
    parser.add_argument("--mode", choices=['path', 'independent'],
                        help="Mode for calculating values at each time points. If it is path, previous time point "
                             "will be chosen. If it is independent, each time point will be computed from <min_time>.",
                        default='path')
    parser.add_argument("--logspace", action='store_true',
                        help="Use logspace time points")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    elif not os.path.isdir(args.out_path):
        print(f"{args.out_path} is already exists and it is not a directory")
        exit(1)

    str_speed = ".speed_balance" if args.speed_balance else ""
    filename = f'steps{args.num_time_steps}.cat{args.num_cat}{str_speed}.time{args.max_time}.' \
               f'samples{args.num_samples}'
    filepath = os.path.join(args.out_path, filename + ".pth")

    if os.path.exists(filepath):
        print("File is already exists.")
        exit(1)

    torch.set_default_dtype(torch.float64)

    alpha = torch.ones(args.num_cat - 1)
    beta =  torch.arange(args.num_cat - 1, 0, -1)

    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = noise_factory(args.num_samples,
                                                                             args.num_time_steps,
                                                                             alpha,
                                                                             beta,
                                                                             total_time=args.max_time,
                                                                             order=args.order,
                                                                             time_steps=args.steps_per_tick,
                                                                             logspace=args.logspace,
                                                                             speed_balanced=args.speed_balance,
                                                                             mode=args.mode)

    v_one = v_one.cpu()
    v_zero = v_zero.cpu()
    v_one_loggrad = v_one_loggrad.cpu()
    v_zero_loggrad = v_zero_loggrad.cpu()
    timepoints = torch.FloatTensor(timepoints)

    torch.save((v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints), filepath)
