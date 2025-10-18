# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.

This script sets up the training process for a diffusion model. It includes
argument parsing, seed fixing, distributed setup, data loading, model creation,
and the training loop. The training process is platform-agnostic and supports
various logging platforms like WandB, ClearML, and Tensorboard.
"""

import os
import json
import pprint

from utils.fixseed import fixseed  # Utility to fix random seeds for reproducibility
from utils.parser_util import train_args  # Argument parser for training configurations
from utils import dist_util  # Utilities for distributed training setup
from train.training_loop import TrainLoop  # Core training loop implementation
from data_loaders.get_data import get_dataset_loader  # Data loader for datasets
from utils.model_util import create_model_and_diffusion  # Model and diffusion creation utility
from train.train_platforms import (
    WandBPlatform,  # Logging platform: Weights & Biases
    ClearmlPlatform,  # Logging platform: ClearML
    TensorboardPlatform,  # Logging platform: Tensorboard
    NoPlatform,  # No logging platform
)


def main():
    """
    Main function to set up and run the training process.
    """
    # Parse training arguments
    args = train_args()

    pprint.pprint(vars(args), indent=4)

    # Fix random seed for reproducibility
    fixseed(args.seed)

    # Initialize the training platform (e.g., WandB, ClearML, Tensorboard)
    train_platform_type = eval(args.train_platform_type)  # Dynamically evaluate platform type
    train_platform = train_platform_type(args.save_dir)  # Initialize platform with save directory
    train_platform.report_args(args, name='Args')  # Log training arguments

    # Validate and prepare the save directory
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)  # Create save directory if it doesn't exist
    else:
        pass

    # Save training arguments to a JSON file in the save directory
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # Set up distributed training utilities
    dist_util.setup_dist(args.device)

    print("creating data loader...")

    # Create the data loader for the specified dataset
    data = get_dataset_loader(
        name=args.dataset,  # Dataset name
        batch_size=args.batch_size,  # Batch size
        num_frames=args.num_frames,  # Number of frames per sample
        fixed_len=args.pred_len + args.context_len,  # Fixed sequence length
        pred_len=args.pred_len,  # Prediction length
        device=dist_util.dev(),  # Device for data loading
    )

    print("creating model and diffusion...")

    # Create the model and diffusion process
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())  # Move the model to the specified device
    model.rot2xyz.smpl_model.eval()  # Set the SMPL model to evaluation mode

    # Print the total number of trainable parameters in the model
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))

    print("Training...")

    # Initialize and run the training loop
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()

    # Close the training platform
    train_platform.close()


# Entry point for the script
if __name__ == "__main__":
    main()
