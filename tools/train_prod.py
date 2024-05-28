#!/usr/bin/env python3
"""
Created on Wed May 22 19:27:14 2024
@author: Alan Ng

Command-line utility to train a production-ready model by learning all
available works and perfs. Uses a stratified K-fold split to handle realistic
data distributions in CSI data sets.

Expects hyperparameters in a hparams_prod.yaml file which expects all the same
hyperparameters as documented for the training hparams.yaml, except:

    Set train_path to point to your full dataset, such as the full.txt output 
    from tools/extract_csi_features.py
    
    val_path will be ignored, since validation sets will be generated for each
    fold from the train_path dataset.
    
    test_path will only be used during the final full-dataset training run,
    as its validation set.
    
    You may also specify one or more external testsets, which will be used
    normally during training for Tensorboard display.

    Set a lower early_stopping_patience than you used in research mode.

Example launch command:
python -m tools.train_prod training/yourdata

The required model_dir parameter is the relative path where this script
creates a subfolder "prod_checkpoints" containing checkpoint files.

"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from src.trainer import Trainer
from src.model import Model
from src.utils import create_logger, get_hparams_as_string, load_hparams
from src.dataset import AudioFeatDataset, read_lines


def extract_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        labels.append(label.item())
    return np.array(labels)


def write_temp_file(data_lines, indices, filepath):
    with open(filepath, "w") as f:
        for idx in indices:
            f.write(data_lines[idx] + "\n")


def save_fold_indices(fold_indices, filepath):
    with open(filepath, "w") as f:
        json.dump(fold_indices, f)


def load_fold_indices(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def cross_validate(
    hp, model_class, device, checkpoint_dir, model_dir, run_id, n_splits=5
):
    """
    Perform k-fold cross-validation to train on entire dataset
    for production use.
    """
    logger = create_logger()

    # Read data lines from the original file
    data_lines = read_lines(hp["train_path"])

    # Load the dataset once for splitting purposes
    initial_dataset = AudioFeatDataset(
        hp,
        data_path=None,
        data_lines=data_lines,
        train=False,  # No augmentation at this point
        mode=hp["mode"],
        chunk_len=hp["chunk_frame"][0] * hp["mean_size"],
    )

    # Extract labels
    labels = extract_labels(initial_dataset)

    # Generate indices for stratified k-fold
    indices = np.arange(len(initial_dataset))

    # Save folds in case training needs to be restarted after an interruption
    fold_indices_file = os.path.join(model_dir, "fold_indices.json")
    active_fold_file = os.path.join(model_dir, "active_fold.txt")
    last_completed_fold = -1

    if os.path.exists(active_fold_file):
        with open(active_fold_file, "r") as f:
            last_completed_fold = int(f.readline().strip())
        logger.info(f"Resuming from fold {last_completed_fold + 1}")
        fold_indices = load_fold_indices(fold_indices_file)
    else:
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=hp["seed"]
        )
        fold_indices = [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in kf.split(indices, labels)
        ]
        save_fold_indices(fold_indices, fold_indices_file)

    fold_results = []

    original_train_path = hp["train_path"]  # should be full dataset

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        if fold <= last_completed_fold:
           continue

        logger.info(f"Training on fold {fold+1}/{n_splits}")

        # Write temporary train and val files
        train_file = os.path.join(model_dir, f"train_fold_{fold+1}.txt")
        val_file = os.path.join(model_dir, f"val_fold_{fold+1}.txt")
        write_temp_file(data_lines, train_idx, train_file)
        write_temp_file(data_lines, val_idx, val_file)
        logger.info(
            f"Temporary fold {fold+1} data files created at {train_file} and {val_file}"
        )

        # Temporarily change hp paths
        hp["train_path"] = train_file
        hp["val_path"] = val_file

        # Create a unique log path for this fold
        log_path = os.path.join(model_dir, "logs", f"{run_id}_fold_{fold+1}")
        os.makedirs(log_path, exist_ok=True)

        # Instantiate and train a new Trainer instance for this fold
        trainer = Trainer(
            hp=hp,
            model=model_class,
            device=device,
            log_path=log_path,
            checkpoint_dir=checkpoint_dir,
            model_dir=model_dir,
            only_eval=False,
            first_eval=False,
        )
        trainer.configure_optimizer()
        trainer.load_model()
        trainer.configure_scheduler()
        trainer.train(max_epochs=100000)

        fold_results.append(
            {
                "fold": fold,
                "best_validation_loss": trainer.best_validation_loss,
            }
        )
        
        # Save the last completed fold
        with open(active_fold_file, 'w') as f:
            f.write(str(fold))

    logger.info("Cross-validation completed")

    # Train on the full dataset, and use a testset as the validation set
    logger.info("Training on the full dataset")
    hp["train_path"] = original_train_path
    hp["val_path"] = hp["covers80"]["query_path"]

    # Create a unique log path for the full dataset training
    log_path = os.path.join(model_dir, "logs", f"{run_id}_full")
    os.makedirs(log_path, exist_ok=True)

    # Instantiate and train a new Trainer instance for the full dataset
    full_trainer = Trainer(
        hp=hp,
        model=model_class,
        device=device,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
        only_eval=False,
        first_eval=False,
    )
    full_trainer.configure_optimizer()
    full_trainer.load_model()
    full_trainer.configure_scheduler()
#    full_trainer.train(max_epochs=(5 * hp["early_stopping_patience"]))
    # Train for the specified number of epochs with validation
    for epoch in range(5 * hp["early_stopping_patience"]):
        full_trainer.train_epoch(epoch, first_eval=False)
        full_trainer.validate_one("val")
        full_trainer.save_model()
        if full_trainer.early_stopping_counter >= hp.get("early_stopping_patience", 10000):
            logger.info(f"Early stopping at epoch {epoch} due to lack of improvement")
            break
    fold_results.append(
        {
            "fold": "full",
            "best_validation_loss": full_trainer.best_validation_loss,
        }
    )
    logger.info("Full dataset training completed")

    return fold_results


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train: python3 -m tools.train_prod model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="give more debug log",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for k-fold cross-validation",
    )
    parser.add_argument(
        "--runid",
        default="",
        action="store",
        help="put TensorBoard logs in these subfolders of ../logs/ one per fold",
    )

    args = parser.parse_args()
    model_dir = args.model_dir
    k_folds = args.k_folds
    run_id = args.runid

    logger = create_logger()
    hp = load_hparams(os.path.join(model_dir, "config/hparams_prod.yaml"))
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
                "You set device: %s"
                " in your hyperparameters but that is not a valid option or is an untested option.",
                hp["device"],
            )
            sys.exit()

    logger.info("%s", get_hparams_as_string(hp))

    torch.manual_seed(hp["seed"])

    checkpoint_dir = os.path.join(model_dir, "prod_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    fold_results = cross_validate(
        hp=hp,
        model_class=Model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
        run_id=run_id,
        n_splits=k_folds,
    )

    for result in fold_results:
        logger.info(
            f"Fold {result['fold']} - Best Validation Loss: {result['best_validation_loss']}"
        )


if __name__ == "__main__":
    _main()
