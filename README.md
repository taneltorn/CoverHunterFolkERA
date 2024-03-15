# CoverHunterMPS

Fork of [Liu Feng's CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter) project. Goal: make it run, and run fast, on Apple Silicon or Nvidia GPUs. And document it better.

# Requirements

1. Either: 
    1. Apple computer with an Apple M-series chip
    2. Other computer with an Nvidia GPU
2. python3 (minimum version 3.10, tested on 3.11)
3. PyTorch with either CUDA or MPS support enabled.
4. sox and therefore also a Java runtime

# Usage

Clone this repo or download it to a folder on your computer. Run the following Unix commands from that folder.

Or run this project in Google Colab, using this Colab notebook:
https://colab.research.google.com/drive/1HKVT3_0ioRPy7lrKzikGXXGysxZrHpOr?usp=sharing

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

CoverHunter includes a prepared configuration to run a training session on the Covers80 dataset located in the 'egs/covers80' subfolder of the project. *Important note:* the default configuration that the CoverHunter authors provided was a nonsense or toy configuration that only demonstrated that you have a working project and environment. It used the same dataset for both training and validation, so by definition it rapidly converged and overfit.

This fork added a train/dev splitting function in the extract_csi_features tool, and built that into the default training hyperparameters. Coming soon: further splitting the train set into train/validation subsets as part of train.

Specify the path to your training data as the one required command-line parameter:

`python -m tools.train egs/covers80/`

This fork also added an optional --runid parameter so you can distinguish your training runs in TensorBoard in case you are experimenting:

`python -m tools.train egs/covers80/ --runid 'first try'`

To see the TensorBoard visualization of the training progress:

`tensorboard --logdir=egs/covers80/logs`

Optionally edit the hparams.yaml configuration file in the folder 'egs/covers80/config' before starting a training run.

This fork added an hparam.yaml setting of "early_stopping_patience" to support the added feature of early stopping (original CoverHunter defaulted to 10,000 epochs!).

Note: Don't use the `torchrun` launch command offered in original CoverHunter. In the single-computer Apple Silicon context, it is not only irrelevant, it actually slows down performance. In my tests it slowed down tools.train performance by about 20%.

The training script's output consists of:
1. The model checkpoint files contained in the egs/covers80/pt_model folder, organized into epoch-specific subfolders. 
2. Embedding vectors for each piece of training data, stored as numpy arrays in the embed-{epoch#}-covers80 subfolders, accompanied in each epoch with metadata text files query.txt and ref.txt, which are identical to each other in this case.

## Evaluation

CoverHunter provided this script to demonstrate how to use your trained model to classify data, aka query data.
1. Have a pre-trained CoverHunter model's output checkpoint files available. You only need your best set (typically your highest-numbered one). If you use original CoverHunter's pre-trained model from https://drive.google.com/file/d/1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j/view), unzip it, and move it to a folder that you rename to, in this example, 'pretrained_model'.
2. Run your query data through extract_csi_features. In the hparams.yaml file for the feature extraction, turn off all augmentation. See data/covers80_testset/hparams.yaml for an example configuration to treat covers80 as the query data:<br> `python3 -m tools.extract_csi_features data/covers80_testset`<br>
The important output from that is full.txt and the cqt_feat subfolder's contents.
3. Run the evaluation script:<br>
`python3 -m tools.eval_testset pretrained_model data/covers80/dataset.txt data/covers80/dataset.txt` 

CoverHunter only implemented evaluation for the case when query and reference data are identical. But there is an optional 4th parameter for `query_in_ref_path` that would be relevant if query and reference are not identical. See the "query_in_ref" heading below under "Input and Output Files."

## Coarse-to-Fine Alignment Training

The command to launch the alignment script that CoverHunter included is:

`python3 -m tools.alignment_for_frame pretrained_model data/covers80/full.txt data/covers80/alignment.txt`

Arguments to pass to the script:
1. Folder containing a pretrained model. For example if you use original CoverHunter's model from https://drive.google.com/file/d/1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j/view), unzip it, and move it to a folder that you rename to 'pretrained_model' at the top level of your project folder. That folder in turn must contain a 'pt_model' subfolder that contains the do_000[epoch] and g_000[epoch] checkpoint files.
2. The output file from the feature-extraction script described above. It must include song_id attributes for each utt (unlike the raw 'dataset.txt' file that CoverHunter provided for covers80).
3. The "alignment.txt" file will receive the output of this script.

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
| batch_size | Usual "batch size" meaning in the field of machine learning. An important parameter to experiment with. |
| chunk_frame | list of numbers used with mean_size. CoverHunter's covers80 config used [1125, 900, 675]. "chunk" references in this training script seem to be the chunks described in the time-domain pooling strategy part of their paper, not the chunks discussed in their coarse-to-fine alignment strategy. See chunk_s. | 
| chunk_s | duration of a chunk_frame in seconds. Apparently you are supposed to manually calculate chunk_s = chunk_frame / frames-per-second * mean_size. I'm not sure why the script doesn't just calculate this itself using CQT hop-size to get frames-per-second? |
| cqt: hop_size: | Fine-grained time resolution, measured as duration in seconds of each CQT spectrogram slice of the audio data. CoverHunter's covers80 setting is 0.04 with a comment "1s has 25 frames". 25 frames per second is hard-coded as an assumption into CoverHunter in various places. |
| data_type | "cqt" (default) or "raw" or "mel". Unknown whether CoverHunter actually implemented anything but CQT-based training |
| device | 'mps' or 'cuda', corresponding to your GPU hardware and PyTorch library support. Theoretically 'cpu' could work but untested and probably of no value. |
| dev_path | Compare train_path and train_sample_path. This dataset is used in each epoch to run the same validation calculation as with the train_sample_path. But these results are used for the early_stopping_patience calculation. Presumably one should include both classes and samples that were excluded from both train_path and train_sample_path. |
| early_stopping_patience | how many epochs to wait for validation loss to improve before early stopping |
| mean_size | See chunk_s above. An integer used to multiply chunk lengths to define the length of the feature chunks used in many stages of the training process. |
| mode | "random" (default) or "defined". Changes behavior when loading training data in chunks in AudioFeatDataset. "random" described in CoverHunter code as "cut chunk from feat from random start". "defined" described as "cut feat with 'start/chunk_len' info from line"|
| m_per_class | From CoverHunter code comments: "m_per_class must divide batch_size without any remainder" and: "At every iteration, this will return m samples per class. For example, if dataloader's batch-size is 100, and m = 5, then 20 classes with 5 samples iter will be returned." |
| query_path | TBD: can apparently be the same path as train_path |
| ref_path | TBD: can apparently be the same path as train_path |
| spec_augmentation | spectral(?) augmentation settings, used to generate temporary data augmentation on the fly during training. CoverHunter settings were:<br>random_erase:<br> &nbsp; prob: 0.5<br> &nbsp; erase_num: 4<br>roll_pitch:<br> &nbsp; prob: 0.5<br> &nbsp; shift_num: 12 |
| train_path | path to a JSON file containing metadata about the data to be used for model training (See full.txt below for details) |
| train_sample_path | path to a JSON file containing metadata about the data to be used for model validation. Compare dev_path above. Presumably one should include a balanced distribution of samples that are *not* included in the train_path dataset, but do include samples for the classes represented in the train_path dataset.(See full.txt below for details) |

## dataset.txt

A JSON formatted or tab-delimited key:value text file (see format defined in the utils.py::line_to_dict() function) expected by extract_csi_features.py that describes the training audio data, with one line per audio file.
| key | value |
| --- | --- |
| utt | Unique identifier. Probably an abbreviation for "utterance," borrowed from speech-recognition ML work. Example "cover80_00000000_0_0". In a musical context we should call this a "performance." |
| wav | relative path to the raw audio file. Example: "data/covers80/wav_16k/annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.wav" |
| dur_s | duration of the audio file in seconds. Example 316.728 |
| song | title of the song. Example "A_Whiter_Shade_Of_Pale" The _add_song_id() function in extract_csi_features assumes that this string is a unique identifier for the parent cover song (so it can't handle musically distinct songs that happen to have the same title). |
| version | Not used by CoverHunter. Example "annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.mp3" |

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

## query_in_ref

The file you can prepare for the tools/eval_testset.py script to pass as the 4th parameter `query_in_ref_path` (CoverHunter did not provide an example file or documentation) assumes:
- JSON or tab-delimited key:value format
- The only line contains a single key 'query_in_ref' with a value that is itself a collection of tuples, where each tuple represents a mapping between an index in the query input file and an index in the reference input file.
This mapping is only used by the _generate_dist_matrix function. That function explains: "List[(idx, idy), ...], means query[idx] is in ref[idy] so we skip that when computing mAP."

# Code Map

Hand-made visualization of how core functions of this project interact with each other. Also includes additional beginner-friendly or verbose code-commenting that I didn't add to the project code.

https://miro.com/app/board/uXjVNkDkn70=/ 
