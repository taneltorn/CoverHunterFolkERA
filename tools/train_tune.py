"""
Created by @samuel-gauthier and @alanngnet in April-May 2024.

Tool to launch many sequential training runs for the purpose of discovering
optimal hyperparameter settings for a given dataset, aka "hyperparameter tuning."


"""

import glob, os, sys, shutil
import argparse
from datetime import date
import torch
import numpy as np
import random
from src.trainer import Trainer
from src.model import Model
from src.utils import load_hparams, create_logger

today = date.today().strftime("%Y-%m-%d")


def make_deterministic(seed):
    """
    @samuel-gauthier investigated non-deterministic training behavior on
    his CUDA platform which @alanngnet did not observe on his MPS platform.
    @samuel-gauthier reported: "my tests showed no variance with deterministic
    = true and benchmark = false, and variance if deterministic = false or
    benchmark = true". With benchmark = true, "I found myself with 3 or 4
    different results when launching 10 [training runs]." (quoted from
    correspondence with @alanngnet 29 May 2024). He arrived at this function's
    method of ensuring deterministic training on CUDA platforms.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    log_path,
    checkpoint_dir,
    hp,
    seed,
):
    make_deterministic(seed)
    os.makedirs(log_path, exist_ok=True)
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    directories = glob.glob(os.path.join(model_dir, "embed_*_"))
    for directory in directories:
        shutil.rmtree(directory)

    t = Trainer(
        hp,
        Model,
        hp["device"],
        log_path,
        checkpoint_dir,
        model_dir,
        only_eval=False,
        first_eval=False,
    )

    t.configure_optimizer()
    t.load_model()
    t.configure_scheduler()
    t.train(max_epochs=15)
    del t.model
    del t
    print(f"Completed experiment with seed {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train-tune: python3 -m tools.train-tune model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    args = parser.parse_args()
    model_dir = args.model_dir
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    experiments = load_hparams(
        os.path.join(model_dir, "config/hp_tuning.yaml")
    )

    # ensure at least one seed
    seeds = experiments.get("seeds",[hp["seed"]]) 
    chunk_frames = experiments["chunk_frames"]
    # ensure at least one mean_size
    mean_sizes = experiments.get("mean_sizes",[hp["mean_size"]])
    m_per_classes = experiments["m_per_classes"]
    spec_augmentations = experiments["spec_augmentations"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger()
    # Don't save the model's checkpoints
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]

    match hp["device"]:  # noqa requires python 3.10
        case "mps":
            if not torch.backends.mps.is_available():
                logger.error(
                    "You requested 'mps' device in your hyperparameters"
                    "but you are not running on an Apple M-series chip or "
                    "have not compiled PyTorch for MPS support."
                )
                sys.exit()
            device = torch.device("mps")
        case "cuda":
            if not torch.cuda.is_available():
                logger.error(
                    "You requested 'cuda' device in your hyperparameters"
                    "but you do not have a CUDA-compatible GPU available."
                )
                sys.exit()
            device = torch.device("cuda")
        case _:
            logger.error(
                "You set device: %s in your hyperparameters but that is "
                "not a valid option or is an untested option.",
                hp["device"],
            )
            sys.exit()

    for chunk_frame in chunk_frames:
        hp["chunk_frame"] = chunk_frame
        for mean_size in mean_sizes:
            hp["mean_size"] = mean_size
            hp["chunk_s"] = chunk_frame[0] * mean_size / 25
            for seed in seeds:
                log_path = os.path.join(
                    model_dir,
                    "logs",
                    "chunk_frame_"
                    + "_".join([str(c) for c in chunk_frame])
                    + f"_mean_size_{mean_size}_seed_{seed}",
                    today,
                )
                print(
                    "==========================================================="
                )
                print(
                    f"Running experiment with seed {seed}, chunk_frame {chunk_frame}, mean_size_{mean_size}"
                )
                run_experiment(log_path, checkpoint_dir, hp, seed)

    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]
    
    for m_per_class in m_per_classes:
        hp["m_per_class"] = m_per_class
        for seed in seeds:
            log_path = os.path.join(
                model_dir,
                "logs",
                f"m_per_class_{m_per_class}_seed_{seed}",
                today,
            )
            print(
                "==========================================================="
            )
            print(
                f"Running experiment with seed {seed}, m_per_class {m_per_class}"
            )
            run_experiment(log_path, checkpoint_dir, hp, seed)

    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]

    for spec_augmentation in spec_augmentations:
        hp["spec_augmentation"] = spec_augmentation
        random_erase_prob = spec_augmentation["random_erase"]["prob"]
        random_erase_num = spec_augmentation["random_erase"]["erase_num"]
        roll_pitch_prob = spec_augmentation["roll_pitch"]["prob"]
        roll_pitch_shift_num = spec_augmentation["roll_pitch"]["shift_num"]

        for seed in seeds:
            log_path = os.path.join(
                model_dir,
                "logs",
                f"erase_prob_{random_erase_prob}_erase_num_{random_erase_num}"
                f"_roll_prob_{roll_pitch_prob}_shift_num_{roll_pitch_shift_num}"
                f"_seed_{seed}",
                today,
            )
            print(
                "==========================================================="
            )
            print(
                f"Running experiment with seed {seed}, spec_augmentation {spec_augmentation}"
            )
            run_experiment(log_path, checkpoint_dir, hp, seed)
