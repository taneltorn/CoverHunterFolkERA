# CoverHunterMPS

Fork of [Liu Feng's CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter) project. Goal: make it run on Apple Silicon with MPS. And document it better.

# Requirements

1. Apple computer with an Apple M-series chip
2. python3 (minimum version uncertain, tested on 3.11)
3. Only needed for the data prep phase: sox and therefore also a Java runtime

# Usage

Clone this repo or download it to a folder on your Mac. Run the following Terminal commands from that folder.

## Data Preparation

Follow the example of the prepared Covers80 dataset included with the original CoverHunter. Directions here are for using that prepared data. See also the "dataset.txt format" documentation below.

1. Download and extract the contents of the covers80.tgz file from http://labrosa.ee.columbia.edu/projects/coversongs/covers80/
2. Abandon the 2-level folder structure that came inside the covers80.tgz file, flattening so all the .mp3 files are in the same folder. One way to do this is:
    1. In Terminal, go to the extracted "coversongs" folder as the current directory. Then: 
    2. `cd covers32k && mv */* .; rm -r */`
3. Convert all the provided .mp3 files to .wav format. You will need to have sox (and therefore also a Java runtime) installed on your computer.
    1. `setopt EXTENDED_GLOB; for f in *.mp3; do sox "$f" -r 16000 "${f%%.mp3}.wav" && rm "$f"; done`
4. Move all these new .wav files to a new folder called "wav_16k" in the project "data/covers80" folder.
5. You can delete the rest of the downloaded covers80.tgz contents.

## Feature Extraction

You must run this before proceeding to the Train step. And you can't run this without first doing the Data Preparation step above.

From the project root folder, run:

`python3 -m tools.extract_csi_features data/covers80/`

## Train

CoverHunter includes a prepared configuration to train on the Covers80 dataset located in the egs/covers80 project folder.
  
Optionally edit the hparams.yaml configuration file in the folder egs/covers80/config

This fork added an hparam setting of "early_stopping_patience" to support the added feature of early stopping (original CoverHunter defaulted to 10,000 epochs!).

`python -m tools.train egs/covers80/`

To see the TensorBoard visualization of the training progress:

`tensorboard --logdir=egs/covers80/logs`

# dataset.txt format

A JSON formatted file expected by extract_csi_features.py that describes the training audio data, file by file.
| key | value |
| --- | --- |
| utt | Unique identifier. Probably an abbreviation for "utterance," borrowed from speech-recognition ML work. Example "cover80_00000000_0_0" |
| wav | relative path to the raw audio file. Example: "data/covers80/wav_16k/annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.wav" |
| dur_s | duration of the audio file in seconds. Example 316.728 |
| song | title of the song. Example "A_Whiter_Shade_Of_Pale" |
| version | Used for what? Example "annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.mp3" |
