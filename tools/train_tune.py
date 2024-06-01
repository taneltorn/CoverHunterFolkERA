"""
Created by @samuel-gauthier and @alanngnet in April-May 2024.

Tool to launch many sequential training runs for the purpose of discovering
optimal hyperparameter settings for a given dataset, aka "hyperparameter tuning."

You must set model_dir below in the "user-defined settings" lines below.

You may set an unlimited number of hyperparameters to test in the "user-defined
settings" lines below. To not test any values of a hyperparameter, leave it
empty. For example, to *not* test any m_per_class settings, make sure:
m_per_classes = []
is the final line mentioning m_per_classes = []. This way you can leave preceding
definitions of m_per_classes as a record of values you may have already tested.

"""

## Start of user-defined settings

model_dir = "training/covers80"

chunk_frames = [
    # seconds: 5, 4, 3
    #    [250, 200, 150],
    # seconds: 15, 12, 9
    [375, 300, 225],
    # seconds: 30, 24, 18
    [750, 600, 450],
    # seconds: 45, 36, 27   # default CoverHunter
    [1125, 900, 675],
]
chunk_frames = []

# You must include at least one mean_size.
mean_sizes = [3]

m_per_classes = [4,8]
m_per_classes = []

spec_augmentations = [
    {
        "random_erase": {
            "prob": 0.5,
            "erase_num": 4,
        },
        "roll_pitch": {
            "prob": 0.5,
            "shift_num": 12,
        },
    },
    {
        "random_erase": {
            "prob": 0,
            "erase_num": 4,
        },
        "roll_pitch": {
            "prob": 0.5,
            "shift_num": 4,
        },
    },
]
# spec_augmentations = []

# Run each configuration with multiple seeds to determine how much variance
# is just due to random aspects of the model, not due to hyperparameters.
seeds = [42, 123, 456]  

### End of user-defined settings

import glob, os, sys, shutil
from datetime import date
import torch
import numpy as np
import random
from src.trainer import Trainer
from src.model import Model
from src.utils import load_hparams, create_logger

today = date.today().strftime("%Y-%m-%d")


def make_deterministic(seed):
    """@samuel-gauthier investigated non-deterministic training behavior on
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


# def _main():

#     # No need to save the model
#     hp["every_n_epoch_to_save"] = 100

#     for chunk_frame in chunk_frames:
#         for mean_size in mean_sizes:

#             print(
#                 "==========================================================="
#             )
#             print("Trying with", chunk_frame, "mean_size", mean_size)
#             print(
#                 "==========================================================="
#             )
#             print()
#             directories = glob.glob(
#                 os.path.join(model_dir, "embed_*_reels50*")
#             )
#             for directory in directories:
#                 shutil.rmtree(directory)

#             torch.manual_seed(hp["seed"])

#             hp["chunk_frame"] = chunk_frame
#             hp["chunk_s"] = chunk_frame[0] / 25 * mean_size
#             hp["mean_size"] = mean_size

#             shutil.rmtree(checkpoint_dir)
#             os.makedirs(checkpoint_dir, exist_ok=True)

#             log_path = os.path.join(
#                 model_dir,
#                 "logs",
#                 "_".join([str(c) for c in chunk_frame])
#                 + f"mean_size{mean_size}",
#                 today,
#             )
#             os.makedirs(log_path, exist_ok=True)

#             t = Trainer(
#                 hp,
#                 Model,
#                 device,
#                 log_path,
#                 checkpoint_dir,
#                 model_dir,
#                 only_eval=False,
#                 first_eval=False,
#             )

#             t.configure_optimizer()
#             t.load_model()
#             t.configure_scheduler()
#             t.train(max_epochs=51)
#             del t.model
#             del t

#     chunk_frame = [1125, 900, 675]
#     hp["chunk_frame"] = chunk_frame
#     hp["chunk_s"] = chunk_frame[0] / 25 * 3

#     for m_per_class in m_per_classes:

#         print("===========================================================")
#         print("Trying with m_per_class", m_per_class)
#         print("===========================================================")
#         print()
#         directories = glob.glob(os.path.join(model_dir, "embed_*_reels50*"))
#         for directory in directories:
#             shutil.rmtree(directory)

#         torch.manual_seed(hp["seed"])

#         hp["m_per_class"] = m_per_class

#         shutil.rmtree(checkpoint_dir)
#         os.makedirs(checkpoint_dir, exist_ok=True)

#         log_path = os.path.join(
#             model_dir,
#             "logs",
#             f"m_per_class_{m_per_class}_",
#             "chunk_frame_" + "_".join([str(c) for c in chunk_frame]),
#             today,
#         )
#         os.makedirs(log_path, exist_ok=True)

#         t = Trainer(
#             hp,
#             Model,
#             device,
#             log_path,
#             checkpoint_dir,
#             model_dir,
#             only_eval=False,
#             first_eval=False,
#         )

#         t.configure_optimizer()
#         t.load_model()
#         t.configure_scheduler()
#         t.train(max_epochs=50)
#         del t.model
#         del t

#     hp["m_per_class"] = 8


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


if __name__ == "__main__":
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger()
    # Don't save the model's checkpoints
    hp["every_n_epoch_to_save"] = 100

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
        for mean_size in mean_sizes:
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
                print(f"Completed experiment with seed {seed}")

    for m_per_class in m_per_classes:
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
            print(f"Completed experiment with seed {seed}")

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
            print(f"Completed experiment with seed {seed}")
