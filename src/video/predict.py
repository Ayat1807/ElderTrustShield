import sys, os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .crop import crop_face, load_predictor

np.sctypes = {
    'int': [np.int8, np.int16, np.int32, np.int64],
    'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
    'float': [np.float16, np.float32, np.float64, np.longdouble],
    'complex': [np.complex64, np.complex128, np.clongdouble],
    'others': [bool, object, bytes, str, np.void],
}
# NumPy 2.0 removed this attribute; DeepfakeBench's bundled imgaug
# dependency still references it at import time. This must live in the
# actual file (not just a notebook cell) so it applies in any process
# that imports this module, including app.py running standalone.

_model = None
_device = None

def _ensure_loaded(deepfakebench_dir: str, dlib_predictor_path: str):
    global _model, _device
    if _model is None:
        sys.path.insert(0, f"{deepfakebench_dir}/training")
        os.chdir(deepfakebench_dir)
        from detectors import DETECTOR

        with open("training/config/detector/ucf.yaml", "r") as f:
            config = yaml.safe_load(f)

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_class = DETECTOR[config["model_name"]]
        _model = model_class(config).to(_device)
        ckpt = torch.load("training/weights/ucf_best.pth", map_location=_device)
        _model.load_state_dict(ckpt, strict=True)
        _model.eval()
        load_predictor(dlib_predictor_path)
    return _model, _device

def _score_frame(frame_bgr):
    cropped_rgb = crop_face(frame_bgr)
    if cropped_rgb is None:
        return None
    arr = (cropped_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(_device)
    dummy_label = torch.zeros(1, dtype=torch.long).to(_device)
    with torch.no_grad():
        predictions = _model({"image": tensor, "label": dummy_label, "mask": None, "landmark": None}, inference=True)
    probs = F.softmax(predictions["cls"], dim=1).cpu().numpy()[0]
    return float(probs[1])

def score_video_file(video_path: str,
                      deepfakebench_dir: str = "/content/repo/external/DeepfakeBench",
                      dlib_predictor_path: str = "/content/repo/external/DeepfakeBench/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat",
                      sample_every_seconds: float = 1.0,
                      aggregate: str = "top2_mean") -> float:
    """Returns probability (0-1) of AI-generated/manipulated face, aggregated
    across sampled frames. aggregate: "top2_mean" (default) - averages the
    two highest-scoring usable frames, weighted toward catching a real signal
    without letting one anomalous frame single-handedly decide the result the
    way "max" did. Also accepts "max" or "mean". Frames with no detected
    face are skipped, not scored as zero."""
    _ensure_loaded(deepfakebench_dir, dlib_predictor_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps * sample_every_seconds))
    scores = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            score = _score_frame(frame)
            if score is not None:
                scores.append(score)
        frame_idx += 1
    cap.release()
    if not scores:
        raise ValueError("No face detected in any sampled frame of this video.")
    if aggregate == "max":
        return float(max(scores))
    elif aggregate == "mean":
        return float(np.mean(scores))
    elif aggregate == "top2_mean":
        top_scores = sorted(scores, reverse=True)[:2]
        return float(np.mean(top_scores))
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}")
