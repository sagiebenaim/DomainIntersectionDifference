import argparse
import os

import torch

from models.models import E_common, E_separate_A, E_separate_B, Decoder
from utils import load_model_for_eval, save_chosen_imgs


def eval(args):
    e_common = E_common(args.sep, int((args.resize / 64)))
    e_separate_A = E_separate_A(args.sep, int((args.resize / 64)))
    e_separate_B = E_separate_B(args.sep, int((args.resize / 64)))
    decoder = Decoder(int((args.resize / 64)))

    if torch.cuda.is_available():
        e_common = e_common.cuda()
        e_separate_A = e_separate_A.cuda()
        e_separate_B = e_separate_B.cuda()
        decoder = decoder.cuda()

    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model_for_eval(save_file, e_common, e_separate_A, e_separate_B, decoder)

    e_common = e_common.eval()
    e_separate_A = e_separate_A.eval()
    e_separate_B = e_separate_B.eval()
    decoder = decoder.eval()

    if not os.path.exists(args.out) and args.out != "":
        os.mkdir(args.out)

    save_chosen_imgs(args, e_common, e_separate_A, e_separate_B, decoder, _iter, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4],
                     False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--load', default='')
    parser.add_argument('--out', default='')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_display', type=int, default=5)

    args = parser.parse_args()

    eval(args)
