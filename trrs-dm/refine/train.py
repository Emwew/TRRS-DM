import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

_STEP = {
    'v1': 15,
    'v2': 15,
    'v3': 4,
    'bicsr': 4,
    'inpaint_imagenet': 4,
    'inpaint_face': 4,
    'faceir': 4,
    'deblur': 4,
    }
_LINK = {
    'vqgan': 'autoencoder_vq_f4.pth',
    'deblur': 'deblur_s4.pth',
     }

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument(
            "-v",
            "--version",
            type=str,
            default="v1",
            choices=["v1", "v2", "v3"],
            help="Checkpoint version.",
            )
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256, 64],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--chop_stride",
            type=int,
            default=-1,
            help="Chopping stride.",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="realsr",
            choices=['realsr', 'bicsr', 'inpaint_imagenet', 'inpaint_face', 'faceir', 'deblur'],
            help="Chopping forward.",
            )
    args = parser.parse_args()

    return args

def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()

    if args.task == 'deblur':
        configs = OmegaConf.load('./configs/deblur_gopro256.yaml')
        assert args.scale == 1, 'Please set scale equals 1 for deblurring!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    else:
        raise TypeError(f"Unexpected task type: {args.task}!")

    # prepare the checkpoint
    if not ckpt_path.exists():
         load_file_from_url(
            url=ckpt_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
            )
    if not vqgan_path.exists():
         load_file_from_url(
            url=vqgan_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64) * (4 // args.scale)
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * (4 // args.scale)
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * (4 // args.scale)
        else:
            raise ValueError("Chop size must be in [512, 256]")
    else:
        chop_stride = args.chop_stride * (4 // args.scale)
    args.chop_size *= (4 // args.scale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    return configs, chop_stride

def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            sf=args.scale,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get('lq_size', 64),
            )

    # setting mask path for inpainting
    if args.task.startswith('inpaint'):
        assert args.mask_path, 'Please input the mask path for inpainting!'
        mask_path = args.mask_path
    else:
        mask_path = None

    resshift_sampler.inference(
            args.in_path,
            args.out_path,
            mask_path=mask_path,
            bs=args.bs,
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
