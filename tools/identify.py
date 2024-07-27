#!/usr/bin/env python3
"""
Created on Sat Mar  2 17:31:23 2024
@author: Alan Ng

Command-line utility to identify closest matches to the target audio
from the model's training data. 

MVP proof-of-concept. Assumptions:
    target has a duration < chunk_s (135sec).
    You already ran tools/eval_testset.py to generate reference embeddings 
        in the model's "embed_...tmp/query_embed" folder

Example invocation:
python -m tools.identify data/covers80 training/covers80 query.wav -top=10

Parameters
----------
data_path : string
    The relative path that expects an hparams.yaml file as documented for use with tools.extract_csi_features. Those hyperparameters are necessary to generate a CQT spectrogram for your query that uses the same CQT parameters as were used for the audio during training of the model. Most important is that n_bins must match.

model_path : string
    The relative path that must contain: 
        1) a subfolder "checkpoints" containing checkpoint files
        2) the abovementioned tmp/query_embed/ folder of reference embeddings
        3) The model's hyperparameters as hparams.yaml

query_path : string
    Relative path to the query audio. 

top : integer
    Optional. Return N=top closest matches to this query.
    


"""
import os, torch, torchaudio, numpy as np
from src.model import Model
from src.cqt import shorter
from src.utils import (
    load_hparams,
    #    RARE_DELIMITER,
    line_to_dict,
    read_lines,
)
import argparse
from heapq import nsmallest
from scipy.spatial.distance import cosine
from nnAudio.features.cqt import CQT, CQT2010v2
from tabulate import tabulate


def _make_feat(wav_path, fmin, max_freq, n_bins, bins_per_octave, device):
    """
    Borrowed CQT logic from tools.extract_csi_features
        _extract_cqt_worker_torchaudio()
    This function should stay synchronized with that one.

    Parameters
    ----------

    wav_path : string
       path to the .wav audio file that you want to identify

    For other parameters see:
        tools.extract_csi_features::_extract_cqt_worker_torchaudio()

    Returns
    -------
    CQT array

    """
    # CQT seems faster on mps, and CQT2010v2 faster on cuda
    if device == "mps":
        transform = CQT
    elif device == "cuda":
        transform = CQT2010v2

    signal, sr = torchaudio.load(wav_path)
    signal = signal.to(device)
    signal = (
        signal
        / torch.max(
            torch.tensor(0.001).to(device), torch.max(torch.abs(signal))
        )
        * 0.999
    )
    signal = transform(
        16000,
        hop_length=640,
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        verbose=False,
    ).to(device)(signal)
    signal = signal + 1e-9
    signal = signal.squeeze(0)

    # Add contrast
    ref_value_log10 = torch.log10(torch.max(signal))
    signal = 20 * torch.log10(signal) - 20 * ref_value_log10

    signal = torch.swapaxes(signal, 0, 1)
    cqt = signal.numpy(force=True)
    return cqt


def _get_feat(query_path, data_hp, chunk_len, mean_size):
    """
    adapted from dataset.py AudioFeatDataset::__getitem__()
    assumes "mode" = "defined" and start = 0
    """
    if "bins_per_octave" not in data_hp:
        bins_per_octave = 12
    else:
        bins_per_octave = data_hp["bins_per_octave"]
    max_freq = data_hp["fmin"] * (2 ** (data_hp["n_bins"] / bins_per_octave))
    feat = _make_feat(
        query_path,
        data_hp["fmin"],
        max_freq,
        data_hp["n_bins"],
        bins_per_octave,
        data_hp["device"],
    )
    feat = feat[0:chunk_len]
    if len(feat) < chunk_len:
        feat = np.pad(
            feat,
            pad_width=((0, chunk_len - len(feat)), (0, 0)),
            mode="constant",
            constant_values=-100,
        )
    feat = shorter(feat, mean_size)
    return torch.from_numpy(feat)


def _load_ref_embeds(ref_lines):
    """
    adapted from src/eval_testset.py _load_data_from_dir()

    returns dictionary of "label" -> embedding.npy associations
    """
    ref_embeds = {}
    for line in ref_lines:
        local_data = line_to_dict(line)
        #        perf = local_data["perf"].split(f"-{RARE_DELIMITER}start-")[0]
        label = local_data["work_id"]
        ref_embeds[label] = np.load(local_data["embed"])
    return ref_embeds


def _main():
    """
    Parameters
    ----------
    See documentation at top of this file

    Returns
    -------
        list: A list of ranked reference labels, from closest to farthest.
    """
    parser = argparse.ArgumentParser(
        description="use model to rank closest matches to input CQT"
    )
    parser.add_argument("data_path")
    parser.add_argument("model_path")
    parser.add_argument("query_path")
    parser.add_argument("-top", default=0, type=int)
    args = parser.parse_args()
    data_dir = args.data_path
    model_dir = args.model_path
    query_path = args.query_path
    top = args.top
    data_hp = load_hparams(os.path.join(data_dir, "hparams.yaml"))
    model_hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))

    match model_hp["device"]:  # noqa requires Python 3.10
        case "mps":
            assert (
                torch.backends.mps.is_available()
            ), "You requested 'mps' device in your hyperparameters but you are not running on an Apple M-series chip or have not compiled PyTorch for MPS support."
            device = torch.device("mps")
        case "cuda":
            assert (
                torch.cuda.is_available()
            ), "You requested 'cuda' device in your hyperparameters but you do not have a CUDA-compatible GPU available."
            device = torch.device("cuda")
        case _:
            print(
                "You set device: ",
                model_hp["device"],
                " in your hyperparameters but that is not a valid option.",
            )
            exit()

    ## Get query embedding
    # next logic copied from eval_testset.py eval_for_map_with_feat()
    if isinstance(model_hp["chunk_frame"], list):
        infer_frame = model_hp["chunk_frame"][0] * model_hp["mean_size"]
    else:
        infer_frame = model_hp["chunk_frame"] * model_hp["mean_size"]
    chunk_s = model_hp["chunk_s"]
    # assumes 25 frames per second
    assert (
        infer_frame == chunk_s * 25
    ), "Error for mismatch of chunk_frame and chunk_s: {}!={}*25".format(
        infer_frame, chunk_s
    )
    #   query_feat = _get_feat(local_data["feat"],hp,infer_frame)
    query_feat = _get_feat(query_path, data_hp, infer_frame,model_hp["mean_size"])

    # unsqueeze to simulate being packaged by DataLoader as expected by the model
    query_feat = query_feat.unsqueeze(0).to(device)  # needs float()?
    with torch.no_grad():
        model = Model(model_hp).to(device)
        model.eval()
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        epoch = model.load_model_parameters(checkpoint_dir, device=device)
        query_embed, _ = model.inference(query_feat)
    query_embed = query_embed.cpu().numpy()[0]

    # Get ref embeddings. Adapted from _cut_lines_with_dur()
    embed_dir = os.path.join(model_dir, "embed_{}_{}".format(epoch, "tmp"))
    ref_lines = read_lines(os.path.join(embed_dir, "ref.txt"))
    ref_embeds = _load_ref_embeds(ref_lines)
    top = len(ref_embeds) if top == 0 else top

    # Calculate cosine similarity between query embedding and reference embeddings
    cos_dists = {
        label: round(cosine(query_embed, ref_embed), 6)
        for label, ref_embed in ref_embeds.items()
    }

    # Get the top N closest reference embeddings
    top_n = nsmallest(top, cos_dists.items(), key=lambda x: x[1])
    top_n_labels, top_n_distances = zip(*top_n)

    return top_n_labels, top_n_distances


if __name__ == "__main__":
    labels, distances = _main()
    table = zip(labels, distances)
    print(tabulate(table, headers=["Label", "Distance"], tablefmt="grid"))
    pass
