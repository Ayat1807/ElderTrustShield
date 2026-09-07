"""
Preprocessing and checkpointed batched inference over a labels CSV.

Checkpointing exists because Colab has disconnected mid-run in this project
before (confirmed: a ~4% loss during one inference run, traced to a
websocket/browser-side disconnect, not a full runtime crash — see
CLAUDE.md). Saving to Drive periodically means a disconnect only costs
progress since the last checkpoint, not the whole run.
"""
import csv
import pickle
from pathlib import Path

import torch
import torchaudio
from tqdm.auto import tqdm

from .model import get_probabilities

TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = 64600  # ~4s at 16kHz, AASIST3's expected input length

_resampler_cache = {}


def load_and_preprocess(audio_path: Path, audio_base: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_base / audio_path))

    if sample_rate != TARGET_SAMPLE_RATE:
        if sample_rate not in _resampler_cache:
            _resampler_cache[sample_rate] = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        waveform = _resampler_cache[sample_rate](waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)
    if waveform.shape[0] < TARGET_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_SAMPLES - waveform.shape[0]))
    else:
        waveform = waveform[:TARGET_SAMPLES]

    return waveform


def run_inference(
    model,
    device,
    csv_path: Path,
    audio_base: Path,
    checkpoint_path: Path,
    batch_size: int = 16,
    checkpoint_every: int = 50,
):
    """Run batched inference, resuming from checkpoint_path if it exists."""
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    label_to_int = {"bonafide": 0, "spoof": 1}

    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as f:
            state = pickle.load(f)
        true_labels = state["true_labels"]
        predicted_labels = state["predicted_labels"]
        bonafide_scores = state["bonafide_scores"]
        attack_algorithms = state["attack_algorithms"]
        start_row = state["next_row"]
        print(f"Resuming: {len(true_labels)} files already done, continuing from row {start_row}")
    else:
        true_labels, predicted_labels, bonafide_scores, attack_algorithms = [], [], [], []
        start_row = 0

    starts = list(range(start_row, len(rows), batch_size))
    for batch_num, start in enumerate(tqdm(starts, desc="Running AASIST3 inference")):
        batch_rows = rows[start : start + batch_size]
        batch_waveforms = torch.stack(
            [load_and_preprocess(row["filename"], audio_base) for row in batch_rows]
        ).to(device, non_blocking=True)

        bonafide_prob, _ = get_probabilities(model, batch_waveforms)
        predictions = (bonafide_prob < 0.5).long()  # 0=bonafide, 1=spoof, correctly oriented

        true_labels.extend(label_to_int[row["label"].strip().lower()] for row in batch_rows)
        predicted_labels.extend(predictions.cpu().tolist())
        bonafide_scores.extend(bonafide_prob.cpu().tolist())
        attack_algorithms.extend(
            row.get("attack_algorithm", row.get("attack_type_code", "unknown")) for row in batch_rows
        )

        if (batch_num + 1) % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, true_labels, predicted_labels, bonafide_scores, attack_algorithms, start + batch_size)

    _save_checkpoint(checkpoint_path, true_labels, predicted_labels, bonafide_scores, attack_algorithms, len(rows))
    return true_labels, predicted_labels, bonafide_scores, attack_algorithms


def _save_checkpoint(path, true_labels, predicted_labels, bonafide_scores, attack_algorithms, next_row):
    with path.open("wb") as f:
        pickle.dump(
            {
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
                "bonafide_scores": bonafide_scores,
                "attack_algorithms": attack_algorithms,
                "next_row": next_row,
            },
            f,
        )
