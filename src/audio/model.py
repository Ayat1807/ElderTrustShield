"""
AASIST3 loading and scoring.

AASIST3 requires a custom PyTorchModelHubMixin-based loader, not a standard
transformers AutoModel:
    git clone https://github.com/mtuciru/AASIST3.git
    from model import aasist3
    model = aasist3.from_pretrained("MTUCI/AASIST3")

CRITICAL CONFIRMED FIX — do not remove without re-verifying:
The AASIST3 model card states "0 = bonafide, 1 = spoof," but for this
loaded checkpoint that convention is INVERTED. Confirmed empirically:
using the card's stated convention gave EER=80.77% (mathematically
impossible to read as "just a weak model" — EER above 50% means strong
separation read backward). Swapping raw indices dropped EER to 19.23% and
flipped per-attack-type accuracy from ~0% to ~100% across every spoof type
(A09-A16). This module returns the CORRECTED orientation so every caller
gets it right without needing to remember the swap. See CLAUDE.md for the
full evidence trail.
"""
import subprocess
import sys
from pathlib import Path

import torch


def clone_aasist3_repo(repo_dir: Path) -> None:
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/mtuciru/AASIST3.git", str(repo_dir)],
            check=True,
        )
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def load_model(repo_dir: Path, device: torch.device):
    """Clone the AASIST3 repo (if needed) and load the pretrained checkpoint."""
    clone_aasist3_repo(repo_dir)
    from model import aasist3  # noqa: E402  (only importable after repo clone)

    model = aasist3.from_pretrained("MTUCI/AASIST3")
    model = model.to(device)
    model.eval()
    print("MODEL_LOADED=", type(model).__name__)
    return model


def get_probabilities(model, batch_waveforms: torch.Tensor):
    """Run inference and return (bonafide_prob, spoof_prob) — CORRECTLY oriented.

    Do not use raw softmax output directly elsewhere; always go through this
    function so the confirmed index swap is applied consistently.
    """
    with torch.inference_mode():
        logits = model(batch_waveforms)
        raw_probs = torch.softmax(logits, dim=1)
    # Confirmed swap: raw index 0 is actually spoof probability, index 1 is bonafide.
    spoof_prob = raw_probs[:, 0]
    bonafide_prob = raw_probs[:, 1]
    return bonafide_prob, spoof_prob
