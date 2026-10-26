#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to generate reference embeddings for all production-use audio data
available, using the production-ready model you trained using, for example,
tools.train_prod.py. Intended for use by tools.compute_centroids,
tools.identify.py, or other applications you might create that use your
fully trained model.

Example invocation:
    python -m tools.make_embeds data/covers80 training/covers80

Add --raw for research use (no normalization, no calibration; embeddings
stored exactly as the model emits them):
    python -m tools.make_embeds data/covers80 training/covers80 --raw

Tune calibration sample size (default 1,000,000):
    python -m tools.make_embeds data/covers80 training/covers80 \\
        --calibration-pairs 2000000

Skip recordings whose CQT .npy files are absent (prints a report first):
    python -m tools.make_embeds data/covers80 training/covers80 \\
        --skip-missing

Parameters
----------
data_path : string
    Relative path to the project folder containing a full.txt file that
    you generated using tools.extract_csi_features.py, for example the one
    you used to train your model. These will be the recordings that your
    inference solution will "know."
    Example: "data/covers80"

model_path : string
    Relative path to the project folder containing your trained model.
    Example: "training/covers80"
    This script requires reuse of the following files that you used and
    generated during training of your model:
        [model_path]/config/hparams.yaml
        [model_path]/checkpoints/...

--raw : flag
    If set, store embeddings exactly as the model emits them (no L2
    normalization), and skip calibration. Use only for research into
    the radial coordinate; the resulting file is not consumable by
    tools.identify or downstream centroid utilities.

--calibration-pairs : int
    Number of cross-work pairs to sample for calibration. Default
    1,000,000. The p-value resolution floor is 1/n_pairs, so a
    threshold at p must be read as resting on roughly p * n_pairs
    sampled negatives; keep that number in the tens for any threshold
    you deploy. Ignored if --raw is set.

--skip-missing : flag
    Before building the dataset, check each entry's CQT .npy path for
    existence. Missing files are listed and then skipped; processing
    continues with whatever is present. Without this flag, missing files
    cause a crash inside the DataLoader.

Embedding protocol
------------------
Each recording is embedded from a single deterministic window: frames
[0, chunk_frame[0] * mean_size) of its CQT (start = 0, "defined" mode),
padded with -100 dB if shorter. This matches tools.identify (start 0)
and src.eval_testset ("defined"). It is independent of the training
hparams' `mode`, which governs random cropping during training only.

Output
------
Pickle file of reference embeddings saved to data_path:
    [data_path]/reference_embeddings.pkl

Schema (default, normalized):
    {
        "embeddings":         dict perf_id -> unit np.ndarray (float32)
        "norms":              dict perf_id -> float (pre-norm magnitude)
        "normalized":         True
        "embed_dim":          int
        "model_fingerprint":  str (16 hex chars)
        "calibration":        dict per src.calibration schema
        "perf_to_work":       dict perf_id -> work_id (str), from the
                              authoritative "work" field of full.txt
        "provenance":         dict describing how this file was built
    }

Schema (--raw): as above, with raw (un-normalized) embeddings,
"normalized": False and "calibration": None.

"perf_to_work" and "provenance" were added 2026-10-05.
src.utils.load_reference_embeddings requires only the first six keys, so
older readers still load new files; tools.compute_centroids falls back
(with a warning) when reading older files that lack "perf_to_work".

Revision history
----------------
Originally created on Sat Jul 27 10:54:22 2024, generating raw embeddings
only. @author: alanngnet with Claude

Complete rewrite 2026-05-03 by alanngnet with Claude Opus 4.7, switching to
normalized embeddings as default, with calibration data supporting both
end-user CSI/VI applications and research-grade data science work.

Revised 2026-10-05 by alanngnet with Claude (Opus 5.5), following a review
of calibration validity on singleton-heavy corpora and a held-out
evaluation of irishtune.id decision rules (tools.check_calibration_transfer,
tools.eval_identify_decisions):
  - Embeddings use a deterministic start-0 window ("defined" mode)
    regardless of the training hparams' `mode`. With mode "random" (the
    covers80 default) each run embedded a random, unseeded crop, so
    reference embeddings and the seeded calibration were not
    reproducible and disagreed with tools.identify / eval_testset.
    No change for configs that already used "defined".
  - The authoritative perf_id -> work_id mapping is persisted as
    "perf_to_work" so tools.compute_centroids no longer has to infer
    works from the perf_id prefix (which silently made every perf its
    own work on SHS100K, Meertens, EstFolk, ...). work_id is stored as
    str, matching the prefix-derived keys of earlier centroids files.
  - Default --calibration-pairs raised from 50,000 to 1,000,000. Deployed
    thresholds near p = 1e-5..1e-4 previously rested on 1-8 sampled
    negatives; src.calibration now also ties the ECDF floor to 1/n_pairs.
  - Fail loud on duplicate perf_ids or missing "work" fields in full.txt.
  - chunk_frame may be a scalar or a list; window length is checked
    against chunk_s as in tools.identify and src.eval_testset.
  - "provenance" records dataset, model, window, corpus composition
    (including the singleton-work fraction) and creation time.
  NOTE: regenerating embeddings changes the calibration sample, so
  p-values for a given distance shift slightly. Re-run
  tools.eval_identify_decisions before reusing hand-tuned p thresholds
  (e.g. in downstream inference applications) with a new file.
"""

import argparse
import datetime as dt
import os
import pickle
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.calibration import compute_calibration
from src.model import Model
from src.dataset import AudioFeatDataset
from src.utils import (
    compute_model_fingerprint,
    load_hparams,
    line_to_dict,
    read_lines,
)


def generate_embeddings(model, data_loader, device):
    """Generate raw embeddings for all samples in the data loader.

    Returns a dict perf_id -> np.ndarray (float32, un-normalized).
    Normalization, if any, is applied later in main().
    """
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating embeddings"):
            perf, feat, _ = batch
            feat = feat.float().to(device)
            embed, _ = model.inference(feat)
            for i, p in enumerate(perf):
                embeddings[p] = embed[i].cpu().numpy().astype(np.float32)
    return embeddings


def _inference_frames(model_hp):
    """Raw (25 fps) CQT frames per embedding window, checked against
    chunk_s exactly as tools.identify and src.eval_testset do."""
    chunk_frame = model_hp["chunk_frame"]
    if isinstance(chunk_frame, list):
        chunk_frame = chunk_frame[0]
    infer_frame = chunk_frame * model_hp["mean_size"]
    chunk_s = model_hp["chunk_s"]
    if infer_frame != chunk_s * 25:
        raise ValueError(
            f"chunk_frame[0] * mean_size = {infer_frame} frames does not "
            f"match chunk_s * 25 = {chunk_s * 25}. Fix hparams.yaml."
        )
    return infer_frame


def main(data_path, model_path, raw, calibration_pairs, skip_missing):
    # Load hyperparameters
    model_hp = load_hparams(os.path.join(model_path, "config/hparams.yaml"))
    infer_frame = _inference_frames(model_hp)

    # Set up device
    device = torch.device(model_hp["device"])

    # Initialize model
    model = Model(model_hp).to(device)
    checkpoint_dir = os.path.join(model_path, "checkpoints")
    model.load_model_parameters(checkpoint_dir, device=device)

    # Compute fingerprint from the loaded weights. Doing this after
    # load_model_parameters ensures we hash the actual weights now in
    # memory, not whatever happens to be on disk.
    fingerprint = compute_model_fingerprint(model.state_dict())
    print(f"Model fingerprint: {fingerprint}")

    # Prepare dataset
    # Need to use full.txt because only it has the necessary work field
    dataset_file = os.path.join(data_path, "full.txt")
    lines = read_lines(dataset_file)
    # Filter out speed-augmented perfs
    lines = [
        line
        for line in lines
        if not re.match(r"sp_[0-9.]+-.+", line_to_dict(line)["perf"])
    ]
    print(f"Loaded {len(lines)} CQT arrays to compute their embeddings.")

    if skip_missing:
        present, missing = [], []
        for line in lines:
            feat_path = line_to_dict(line)["feat"]
            (present if os.path.exists(feat_path) else missing).append(line)
        if missing:
            print(
                f"WARNING: {len(missing)} CQT file(s) not found and will "
                f"be skipped:"
            )
            for line in missing:
                d = line_to_dict(line)
                print(f"  {d['perf']}: {d['feat']}")
        lines = present
        print(f"Proceeding with {len(lines)} recordings.")

    # Map perf_id -> work_id from the explicit "work" field in full.txt.
    # Do NOT derive the work from the perf_id string (e.g. p.split(".")[0]):
    # that convention only holds for the Irish production datasets and
    # silently makes every performance its own work on datasets whose
    # perf_ids have no work prefix (SHS100K, meertens, estfolk, ...),
    # which would contaminate the calibration negative distribution with
    # genuine cover pairs. The "work" field is authoritative for all
    # datasets produced by tools.extract_csi_features. Stored as str so
    # centroid keys are the same type whether the JSON held a number or
    # a string.
    dicts = [line_to_dict(line) for line in lines]
    dup = [p for p, c in Counter(d["perf"] for d in dicts).items() if c > 1]
    if dup:
        raise ValueError(
            f"{len(dup)} duplicate perf_id(s) in {dataset_file}, e.g. "
            f"{dup[:5]}. Each recording must appear once."
        )
    no_work = [d["perf"] for d in dicts if "work" not in d]
    if no_work:
        raise ValueError(
            f"{len(no_work)} line(s) in {dataset_file} lack a 'work' "
            f"field, e.g. {no_work[:5]}. Regenerate with "
            f"tools.extract_csi_features."
        )
    perf_to_work = {d["perf"]: str(d["work"]) for d in dicts}

    # Deterministic window: start 0, "defined" mode (see module docstring).
    dataset = AudioFeatDataset(
        model_hp,
        data_lines=lines,
        train=False,
        mode="defined",
        chunk_len=infer_frame,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=model_hp["batch_size"],
        shuffle=False,
        num_workers=model_hp["num_workers"],
        pin_memory=True,
    )

    raw_embeddings = generate_embeddings(model, data_loader, device)

    # Always extract pre-normalization norms (cheap; preserved as OOD
    # signal even when --raw is not set).
    norms = {p: float(np.linalg.norm(v)) for p, v in raw_embeddings.items()}

    # Sanity check: any zero-norm embedding is a degenerate case that
    # would divide-by-zero on normalization. Fail loud.
    zero_norm = [p for p, n in norms.items() if n == 0.0]
    if zero_norm:
        raise RuntimeError(
            f"{len(zero_norm)} embedding(s) have zero norm: "
            f"{zero_norm[:5]}{'...' if len(zero_norm) > 5 else ''}. "
            f"This indicates a corrupt input or model bug."
        )

    # Restrict the mapping to what was actually embedded.
    perf_to_work = {p: perf_to_work[p] for p in raw_embeddings}
    work_to_perfs = {}
    for p, w in perf_to_work.items():
        work_to_perfs.setdefault(w, []).append(p)
    sizes = np.array([len(v) for v in work_to_perfs.values()])
    n_singletons = int((sizes == 1).sum())
    print(
        f"Corpus: {len(perf_to_work)} perfs in {len(work_to_perfs)} works; "
        f"{n_singletons} singleton works ({n_singletons / len(sizes):.1%} "
        f"of works, {n_singletons / len(perf_to_work):.1%} of perfs)"
    )

    if raw:
        embeddings_out = raw_embeddings
        normalized = False
        calibration = None
        print("Storing raw embeddings (no normalization, no calibration).")
    else:
        embeddings_out = {
            p: (v / norms[p]).astype(np.float32)
            for p, v in raw_embeddings.items()
        }
        normalized = True

        print(
            f"Computing calibration from up to {calibration_pairs} "
            f"cross-work pairs..."
        )
        calibration = compute_calibration(
            embeddings_out,
            work_to_perfs,
            n_pairs=calibration_pairs,
        )
        print(
            f"Calibration: mu={calibration['mu']:.4f}, "
            f"sigma={calibration['sigma']:.4f}, "
            f"n_pairs={calibration['n_pairs']} (p-value floor "
            f"{1 / calibration['n_pairs']:.1e})"
        )

    # Determine embed_dim from any embedding (all have same shape)
    any_perf = next(iter(embeddings_out))
    embed_dim = int(embeddings_out[any_perf].shape[0])

    provenance = {
        "created": dt.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "tool": "tools.make_embeds (rev 2026-10-05)",
        "data_path": os.path.abspath(data_path),
        "model_path": os.path.abspath(model_path),
        "window": f"start=0, {infer_frame} raw frames "
        f"({infer_frame / 25:g} s), mean_size={model_hp['mean_size']}, "
        f"pad -100 dB",
        "n_perfs": len(perf_to_work),
        "n_works": len(work_to_perfs),
        "n_singleton_works": n_singletons,
        "skip_missing": bool(skip_missing),
    }

    record = {
        "embeddings": embeddings_out,
        "norms": norms,
        "normalized": normalized,
        "embed_dim": embed_dim,
        "model_fingerprint": fingerprint,
        "calibration": calibration,
        "perf_to_work": perf_to_work,
        "provenance": provenance,
    }

    output_file = os.path.join(data_path, "reference_embeddings.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(record, f)

    print(f"Reference embeddings saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reference embeddings for production use."
    )
    parser.add_argument(
        "data_path", help="Path to the data folder containing full.txt"
    )
    parser.add_argument(
        "model_path", help="Path to the folder containing the trained model"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Store raw (un-normalized) embeddings, skip calibration. "
        "Research-only; output is not consumable by tools.identify.",
    )
    parser.add_argument(
        "--calibration-pairs",
        type=int,
        default=1_000_000,
        help="Number of cross-work pairs to sample for calibration "
        "(default: 1000000; p-value floor is 1/n). Ignored if --raw.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Check each CQT .npy path before building the dataset; report "
        "missing files and proceed without them instead of crashing.",
    )
    args = parser.parse_args()

    main(
        args.data_path,
        args.model_path,
        raw=args.raw,
        calibration_pairs=args.calibration_pairs,
        skip_missing=args.skip_missing,
    )
