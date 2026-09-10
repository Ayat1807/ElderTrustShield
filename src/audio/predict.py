from pathlib import Path
import torch

from .model import load_model, get_probabilities
from .inference import load_and_preprocess

_model = None
_device = None

def _ensure_loaded(aasist3_repo_dir: str = "/content/AASIST3"):
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = load_model(Path(aasist3_repo_dir), _device)
    return _model, _device

def score_audio_file(audio_path: str, aasist3_repo_dir: str = "/content/AASIST3") -> float:
    """Returns probability (0-1) that the audio is a deepfake/spoofed voice."""
    model, device = _ensure_loaded(aasist3_repo_dir)
    # audio_base="." with an absolute audio_path works correctly via
    # pathlib's join behavior (Path(".") / "/abs/path" == "/abs/path")
    waveform = load_and_preprocess(Path(audio_path), Path("."))
    batch = waveform.unsqueeze(0).to(device)
    _, spoof_prob = get_probabilities(model, batch)
    return float(spoof_prob.item())
