# CoverHunterMPS

Fork of [Liu Feng's CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter) project. Goal: make it run on Apple Silicon with MPS.

# Requirements

Apple computer with an Apple M-series chip
sox (only needed for data prep phase) and therefore a Java runtime
python3 (minimum version uncertain, tested on 3.11)

# Usage

Clone this repo or download it to a folder on your Mac. Run the following Terminal commands from that folder.

## Train

Assuming you have a hparams.yaml file in the folder egs/covers80/config that defines where your training data is, then:

`python -m tools.train egs/covers80/`

