# CoverHunterMPS

Fork of [Liu Feng's CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter) project. Goal: make it run on Apple Silicon with MPS. And document it better.

# Requirements

1. Apple computer with an Apple M-series chip
2. python3 (minimum version uncertain, tested on 3.11)
3. Only needed for the data prep phase: sox and therefore also a Java runtime

# Usage

Clone this repo or download it to a folder on your Mac. Run the following Terminal commands from that folder.

## Data Preparation

Follow the example of the prepared Covers80 dataset included with the original CoverHunter. Directions here are for using that prepared data. See also the "dataset.txt format" heading below.

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

CoverHunter includes a prepared configuration to train on the Covers80 dataset located in the 'egs/covers80' subfolder of the project. Specify the path to your training data as the one required command-line parameter:

`python -m tools.train egs/covers80/`

This fork also added a noptional --runid parameter so you can distinguish your training runs in Tensorboard in case you are experimenting:

`python -m tools.train egs/covers80/ --runid 'first try'`

To see the TensorBoard visualization of the training progress:

`tensorboard --logdir=egs/covers80/logs`

Optionally edit the hparams.yaml configuration file in the folder 'egs/covers80/config' before starting a training run.

This fork added an hparam.yaml setting of "early_stopping_patience" to support the added feature of early stopping (original CoverHunter defaulted to 10,000 epochs!).

# dataset.txt format

A JSON formatted file expected by extract_csi_features.py that describes the training audio data, file by file.
| key | value |
| --- | --- |
| utt | Unique identifier. Probably an abbreviation for "utterance," borrowed from speech-recognition ML work. Example "cover80_00000000_0_0" |
| wav | relative path to the raw audio file. Example: "data/covers80/wav_16k/annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.wav" |
| dur_s | duration of the audio file in seconds. Example 316.728 |
| song | title of the song. Example "A_Whiter_Shade_Of_Pale" |
| version | Used for what? Example "annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.mp3" |

# Hyperparameters (hparams.yaml)

| key | value |
| --- | --- |
| aug_speed_mode | list of ratios used for speed augmention of your raw training data. Example: [0.8, 0.9, 1.1, 1.2] means use 80%, 90%, 110%, and 120% speed variants of your original audio data. Experimentation proved you must exclude 1, because including the unmodified original audio for some reason consistently killed the model, so in this fork I hard-coded elimination of 99-101% output of the speed-augmentation function.|
| chunk_frame | list of numbers used with mean_size. CoverHunter package used [1125, 900, 675] | 
| data_type | "cqt" (default) or "raw" or "mel". Unknown whether CoverHunter actually implemented anything but CQT-based training |
| dev_sample_path | TBD: can apparently be the same path as train_path |
| early_stopping_patience | how many epochs to wait for avg_ce_loss to improve before early stopping |
| mean_size | used to multiply each member of chunk_frame to calculate chunk_len (TBD what the latter does) |
| mode | "random" (default) or "defined". Unclear what "defined" does. Changes behavior when loading training data in chunks in AudioFeatDataset. |
| query_path | TBD: can apparently be the same path as train_path |
| ref_path | TBD: can apparently be the same path as train_path |
| train_path | path to a text file listing certain required attributes of every training data sample |
| train_sample_path | TBD: can apparently be the same path as train_path |

 