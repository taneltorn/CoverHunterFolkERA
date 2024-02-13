# CoverHunterMPS

Fork of [Liu Feng's CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter) project. Goal: make it run on Apple Silicon with MPS.

# Requirements

1. Apple computer with an Apple M-series chip
2. sox (only needed for data prep phase) and therefore a Java runtime
3. python3 (minimum version uncertain, tested on 3.11)

# Usage

Clone this repo or download it to a folder on your Mac. Run the following Terminal commands from that folder.

## Feature Extraction

Not yet tested for this fork. You must run this before proceeding to the Train step.

## Train

CoverHunter includes a prepared configuration to train on the Covers80 dataset located in the egs/covers80 project folder.
  
Optionally edit the hparams.yaml configuration file in the folder egs/covers80/config

This fork added an hparam setting of "early_stopping_patience" to support the added feature of early stopping (original CoverHunter defaulted to 10,000 epochs!).

`python -m tools.train egs/covers80/`

To see the TensorBoard visualization of the training progress:

`tensorboard --logdir=egs/covers80/logs`