import argparse
import os

from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models.models import *
from utils import Logger
from utils import get_train_dataset
from utils import save_imgs, save_model, load_model, save_stripped_imgs


def train(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    _iter = 0
    domA_train, domB_train = get_train_dataset(args)

    size = args.resize // 64
    dim = 512

    e_common = E_common(args.sep, size, dim=dim)
    e_separate_A = E_separate_A(args.sep, size)
    e_separate_B = E_separate_B(args.sep, size)
    decoder = Decoder(size, dim=dim)
    disc = Disc(args.sep, size, dim=dim)

    A_label = torch.full((args.bs,), 1)
    B_label = torch.full((args.bs,), 0)
    zero_encoding = torch.full((args.bs, args.sep * (size) * (size)), 0)
    one_encoding = torch.full((args.bs, args.sep * (size) * (size)), 1)

    l1 = nn.L1Loss()
    bce = nn.BCELoss()

    if torch.cuda.is_available():
        e_common = e_common.cuda()
        e_separate_A = e_separate_A.cuda()
        e_separate_B = e_separate_B.cuda()
        decoder = decoder.cuda()
        disc = disc.cuda()

        A_label = A_label.cuda()
        B_label = B_label.cuda()
        zero_encoding = zero_encoding.cuda()
        one_encoding = one_encoding.cuda()

        l1 = l1.cuda()
        bce = bce.cuda()

    ae_params = list(e_common.parameters()) + list(e_separate_A.parameters()) + list(
        e_separate_B.parameters()) + list(decoder.parameters())
    ae_optimizer = optim.Adam(ae_params, lr=args.lr, betas=(0.5, 0.999))

    disc_params = disc.parameters()
    disc_optimizer = optim.Adam(disc_params, lr=args.disclr, betas=(0.5, 0.999))

    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model(save_file, e_common, e_separate_A, e_separate_B, decoder, ae_optimizer, disc,
                           disc_optimizer)

    e_common = e_common.train()
    e_separate_A = e_separate_A.train()
    e_separate_B = e_separate_B.train()
    decoder = decoder.train()
    disc = disc.train()

    logger = Logger(args.out)

    print('Started training...')
    while True:
        domA_loader = torch.utils.data.DataLoader(domA_train, batch_size=args.bs,
                                                  shuffle=True, num_workers=6)
        domB_loader = torch.utils.data.DataLoader(domB_train, batch_size=args.bs,
                                                  shuffle=True, num_workers=6)
        if _iter >= args.iters:
            break

        for domA_img, domB_img in zip(domA_loader, domB_loader):

            if domA_img.size(0) != args.bs or domB_img.size(0) != args.bs:
                break

            domA_img = Variable(domA_img)
            domB_img = Variable(domB_img)

            if torch.cuda.is_available():
                domA_img = domA_img.cuda()
                domB_img = domB_img.cuda()

            domA_img = domA_img.view((-1, 3, args.resize, args.resize))
            domB_img = domB_img.view((-1, 3, args.resize, args.resize))

            ae_optimizer.zero_grad()

            A_common = e_common(domA_img)
            A_separate_A = e_separate_A(domA_img)
            A_separate_B = e_separate_B(domA_img)
            if args.no_flag:
                A_encoding = torch.cat([A_common, A_separate_A, A_separate_A], dim=1)
            else:
                A_encoding = torch.cat([A_common, A_separate_A, zero_encoding], dim=1)
            B_common = e_common(domB_img)
            B_separate_A = e_separate_A(domB_img)
            B_separate_B = e_separate_B(domB_img)

            if args.one_encoding:
                B_encoding = torch.cat([B_common, B_separate_B, one_encoding], dim=1)
            elif args.no_flag:
                B_encoding = torch.cat([B_common, B_separate_B, B_separate_B], dim=1)
            else:
                B_encoding = torch.cat([B_common, zero_encoding, B_separate_B], dim=1)

            A_decoding = decoder(A_encoding)
            B_decoding = decoder(B_encoding)

            A_reconstruction_loss = l1(A_decoding, domA_img)
            B_reconstruction_loss = l1(B_decoding, domB_img)

            A_separate_B_loss = l1(A_separate_B, zero_encoding)
            B_separate_A_loss = l1(B_separate_A, zero_encoding)

            logger.add_value('A_recon', A_reconstruction_loss)
            logger.add_value('B_recon', B_reconstruction_loss)
            logger.add_value('A_sep_B', A_separate_B_loss)
            logger.add_value('B_sep_A', B_separate_A_loss)

            loss = 0

            if args.reconweight > 0:
                loss += args.reconweight * (A_reconstruction_loss + B_reconstruction_loss)

            if args.zeroweight > 0:
                loss += args.zeroweight * (A_separate_B_loss + B_separate_A_loss)

            if args.discweight > 0:
                preds_A = disc(A_common)
                preds_B = disc(B_common)
                distribution_adverserial_loss = args.discweight * \
                                                (bce(preds_A, B_label) + bce(preds_B, B_label))
                logger.add_value('distribution_adverserial', distribution_adverserial_loss)
                loss += distribution_adverserial_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, 5)
            ae_optimizer.step()

            if args.discweight > 0:
                disc_optimizer.zero_grad()

                A_common = e_common(domA_img)
                B_common = e_common(domB_img)

                disc_A = disc(A_common)
                disc_B = disc(B_common)

                loss = bce(disc_A, A_label) + bce(disc_B, B_label)
                logger.add_value('dist_disc', loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 5)
                disc_optimizer.step()

            if _iter % args.progress_iter == 0:
                print('Outfile: %s <<>> Iteration %d' % (args.out, _iter))

            if _iter % args.log_iter == 0:
                logger.log(_iter)

            logger.reset()

            if _iter % args.display_iter == 0:
                e_common = e_common.eval()
                e_separate_A = e_separate_A.eval()
                e_separate_B = e_separate_B.eval()
                decoder = decoder.eval()

                save_imgs(args, e_common, e_separate_A, e_separate_B, decoder, _iter, size=size, BtoA=True)
                save_imgs(args, e_common, e_separate_A, e_separate_B, decoder, _iter, size=size, BtoA=False)
                save_stripped_imgs(args, e_common, e_separate_A, e_separate_B, decoder, _iter, size=size, A=True)
                save_stripped_imgs(args, e_common, e_separate_A, e_separate_B, decoder, _iter, size=size, A=False)

                e_common = e_common.train()
                e_separate_A = e_separate_A.train()
                e_separate_B = e_separate_B.train()
                decoder = decoder.train()

            if _iter % args.save_iter == 0:
                save_file = os.path.join(args.out, 'checkpoint')
                save_model(save_file, e_common, e_separate_A, e_separate_B, decoder, ae_optimizer, disc,
                           disc_optimizer, _iter)

            _iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--disclr', type=float, default=0.0002)
    parser.add_argument('--progress_iter', type=int, default=100)
    parser.add_argument('--display_iter', type=int, default=1000)
    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--load', default='')
    parser.add_argument('--zeroweight', type=float, default=1.0)
    parser.add_argument('--reconweight', type=float, default=1.0)
    parser.add_argument('--discweight', type=float, default=0.001)
    parser.add_argument('--num_display', type=int, default=12)
    parser.add_argument('--one_encoding', type=int, default=0)
    parser.add_argument('--no_flag', type=int, default=0)


    args = parser.parse_args()

    train(args)
