"""Evluate a plenoctree on test set.

Usage:

export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/PlenOctree/checkpoints/syn_sh16
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.evaluation \
    --input $CKPT_ROOT/$SCENE/octrees/tree_opt.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/
"""
import torch
import numpy as np
import os
from absl import app
from absl import flags
from tqdm import tqdm
import imageio

from octree.nerf import models
from octree.nerf import utils
from octree.nerf import datasets

import svox

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "input",
    "./tree_opt.npz",
    "Input octree npz from optimization.py",
)
flags.DEFINE_string(
    "write_vid",
    None,
    "If specified, writes rendered video to given path (*.mp4)",
)
flags.DEFINE_string(
    "write_images",
    None,
    "If specified, writes images to given path (*.png)",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def main(unused_argv):
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)

    dataset = datasets.get_dataset("test", FLAGS)

    print('N3Tree load', FLAGS.input)
    t = svox.N3Tree.load(FLAGS.input, map_location=device)

    avg_psnr, avg_ssim, avg_lpips, out_frames = utils.eval_octree(t, dataset, FLAGS,
            want_lpips=True,
            want_frames=FLAGS.write_vid is not None or FLAGS.write_images is not None)
    print('Average PSNR', avg_psnr, 'SSIM', avg_ssim, 'LPIPS', avg_lpips)

    if FLAGS.write_vid is not None and len(out_frames):
        print('Writing to', FLAGS.write_vid)
        imageio.mimwrite(FLAGS.write_vid, out_frames)

    if FLAGS.write_images is not None and len(out_frames):
        print('Writing to', FLAGS.write_images)
        os.makedirs(FLAGS.write_images, exist_ok=True)
        for idx, frame in tqdm(enumerate(out_frames)):
            imageio.imwrite(os.path.join(FLAGS.write_images, f"{idx:03d}.png"), frame)

if __name__ == "__main__":
    app.run(main)
