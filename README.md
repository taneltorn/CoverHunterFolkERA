# CoverHunterMPS

Fork of [Liu Feng's CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter) project. Goal: make it run, and run fast, on Apple Silicon. And document it better.

# Requirements

1. Apple computer with an Apple M-series chip
2. python3 (minimum version uncertain, tested on 3.11)
3. sox and therefore also a Java runtime

# Usage

Clone this repo or download it to a folder on your Mac. Run the following Terminal commands from that folder.

## Data Preparation

Follow the example of the prepared Covers80 dataset included with the original CoverHunter. Directions here are for using that prepared data. See also the "dataset.txt format" heading below.

1. Download and extract the contents of the covers80.tgz file from http://labrosa.ee.columbia.edu/projects/coversongs/covers80/
2. Abandon the 2-level folder structure that came inside the covers80.tgz file, flattening so all the .mp3 files are in the same folder. One way to do this is:
    1. In Terminal, go to the extracted "coversongs" folder as the current directory. Then: 
    2. `cd covers32k && mv */* .; rm -r */`
3. Convert all the provided .mp3 files to .wav format. One way to do this is:
    1. `setopt EXTENDED_GLOB; for f in *.mp3; do sox "$f" -r 16000 "${f%%.mp3}.wav" && rm "$f"; done`
4. Move all these new .wav files to a new folder called "wav_16k" in the project "data/covers80" folder.
5. You can delete the rest of the downloaded covers80.tgz contents.

## Feature Extraction

You must run this before proceeding to the Train step. And you can't run this without first doing the Data Preparation step above.

From the project root folder, run:

`python3 -m tools.extract_csi_features data/covers80/`

Side note: I attempted MPS optimization of CQT feature extraction but failed. The main issue is that PyTorch MPS implementation is very incomplete for many of the tensor operations involved in this CQT implementation. Even the core fft() function just became available with torch torch-2.3.0.dev20240222 or higher. See my comments in extract_csi_features.py and cqt.py.

## Training

CoverHunter includes a prepared configuration to train on the Covers80 dataset located in the 'egs/covers80' subfolder of the project. Specify the path to your training data as the one required command-line parameter:

`python -m tools.train egs/covers80/`

This fork also added an optional --runid parameter so you can distinguish your training runs in TensorBoard in case you are experimenting:

`python -m tools.train egs/covers80/ --runid 'first try'`

To see the TensorBoard visualization of the training progress:

`tensorboard --logdir=egs/covers80/logs`

Optionally edit the hparams.yaml configuration file in the folder 'egs/covers80/config' before starting a training run.

This fork added an hparam.yaml setting of "early_stopping_patience" to support the added feature of early stopping (original CoverHunter defaulted to 10,000 epochs!).

Note: Don't use the `torchrun` launch command offered in original CoverHunter. In the single-computer Apple Silicon context, it is not only irrelevant, it actually slows down performance. In my tests it slowed down performance by about 20%.

## Coarse-to-Fine Alignment Training

The command to launch the alignment script that CoverHunter included is:

`python3 -m tools.alignment_for_frame pretrained_model data/covers80/full.txt data/covers80/alignment`

# Input and Output Files

## Hyperparameters (hparams.yaml)

There are two different hparams.yaml files, each used at different stages. 

1. The one located in the folder you provide on the command line to tools.extract_csi_features is used only by that script. Really "aug_speed_mode" is the only parameter that matters for Covers80 training.

| key | value |
| --- | --- |
|add_noise| Original CoverHunter provided the example of: <div>{<br> &nbsp; "prob": 0.75,<br> &nbsp; "sr": 16000,<br> &nbsp; "chunk": 3,<br> &nbsp; "name": "cqt_with_asr_noise",<br> &nbsp; "noise_path": "dataset/asr_as_noise/dataset.txt"<br>}<br>However, the CoverHunter repo did not include whatever might supposed to be in "dataset/asr_as_noise/dataset.txt" file nor does the CoverHunter research paper describe it. If that path does not exist in your project folder structure, then tools.extract_csi_features will just skip the stage of adding noise augmentation. At least for training successfully on Covers80, noise augmentation doesn't seem to be needed.|
| aug_speed_mode | list of ratios used in tools.extract_csi_features for speed augmention of your raw training data. Example: [0.8, 0.9, 1.0, 1.1, 1.2] means use 80%, 90%, 100%, 110%, and 120% speed variants of your original audio data.|

2. The one located in the "config" subfolder of the path you provide on the command line to tools.train uses all the other parameters listed below during training.

| key | value |
| --- | --- |
| chunk_frame | list of numbers used with mean_size. CoverHunter package used [1125, 900, 675] | 
| data_type | "cqt" (default) or "raw" or "mel". Unknown whether CoverHunter actually implemented anything but CQT-based training |
| dev_sample_path | TBD: can apparently be the same path as train_path |
| early_stopping_patience | how many epochs to wait for avg_ce_loss to improve before early stopping |
| mean_size | used to multiply each member of chunk_frame to calculate chunk_len (TBD what the latter does) |
| mode | "random" (default) or "defined". Changes behavior when loading training data in chunks in AudioFeatDataset. "random" described in CoverHunter code as "cut chunk from feat from random start". "defined" described as "cut feat with 'start/chunk_len' info from line"|
| query_path | TBD: can apparently be the same path as train_path |
| ref_path | TBD: can apparently be the same path as train_path |
| train_path | path to a text file listing certain required attributes of every training data sample. (See full.txt below) |
| train_sample_path | TBD: can apparently be the same path as train_path |

## dataset.txt

A JSON formatted file expected by extract_csi_features.py that describes the training audio data, file by file.
| key | value |
| --- | --- |
| utt | Unique identifier. Probably an abbreviation for "utterance," borrowed from speech-recognition ML work. Example "cover80_00000000_0_0". In a musical context we should call this a "performance." |
| wav | relative path to the raw audio file. Example: "data/covers80/wav_16k/annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.wav" |
| dur_s | duration of the audio file in seconds. Example 316.728 |
| song | title of the song. Example "A_Whiter_Shade_Of_Pale" The _add_song_id() function in extract_csi_features assumes that this string is a unique identifier for the parent cover song (so it can't handle musically distinct songs that happen to have the same title). |
| version | Used for what? Example "annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.mp3" |

## full.txt 

full.txt is the JSON-formatted training data catalog for tools.train.py, generated by tools.extract_csi_features. In case you do your own data prep instead of using tools.extract_csi_features, here's the structure of full.txt.

| key | value |
| --- | --- |
| utt | (see dataset.txt) |
| wav | (see dataset.txt) |
| dur_s |(see dataset.txt) |
| song | (see dataset.txt) |
| version | (see dataset.txt) |
| feat | path to the CQT features of this utt stored as .npy array. Example: "data/covers80/cqt_feat/sp_0.8-cover80_00000146_71_0.cqt.npy" |
| feat_len | output of len(np.load(feat)). Example: 9198 |
| song_id | internal, arbitrary unique identifier for the song. This is what teaches the model which utts (performances) are considered by humans to be the "same song." Example: 0 |
| version_id | internal, arbitrary unique identifier for each artificially augmented variant of the original utt (performance). Example: 0 |

Note: Original CoverHunter omitted the unmodified audio by accident due to a logic error at lines 104-112 of tools.extract_csi_features, by unintentionally appending the next value of `sp_utt` to the beginning of `local_data['utt']`. And if and only if the '1.0' member of the aug_speed_mode hyperparameter was not listed first, the result then was not only that the 1.0 variant was omitted, but also a duplicate copy of the 90% variant was created and included in the final output of full.txt in the end, both entries in full.txt pointing to the same cqt.npy file, just with different version_id values. 

That bug didn't prevent successful training, but fixing the bug did, until I discovered that because then, when the model was being fed the intended number of song versions (augmented from 1 to 5 instead of to 4 versions), CoverHunter's preset batch size of 16 became a barrier to success. Increasing the batch size hyperparameter to 32 and larger made a huge difference, resulting in much faster convergence and higher mAP than the original CoverHunter code.

## Other files generated by extract_csi_features.py

Listed in the order that the script creates them:

| filename | comments |
|---|---|
| data.init.txt | Copy of dataset.txt after sorting by ‘utt’ and de-duping. Not used by train.py |
| sp_aug subfolder | Sox-modified wav speed variants of the raw training .wav files, at the speeds defined in hparams.yaml. Not used by train.py |
| sp_aug.txt | Copy of data.init.txt but with addition of 1 new row for each augmented variant created in sp_aug/*.wav. Not used by train.py. |
| cqt_feat subfolder | Numpy array files of the CQT data for each file listed in full.txt. Needed by train.py |
| song_id_num.map | Text file, not used by train.py, maybe not by anything else? |
| song_id.map | Text file, not used by train.py, maybe not by anything else? |
| song_name_num.map | Text file, not used by train.py, maybe not by anything else? |
| full.txt | See above detailed description.| 
