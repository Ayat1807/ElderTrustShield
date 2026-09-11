import dlib
import cv2
import numpy as np
from skimage import transform as trans

_face_detector = dlib.get_frontal_face_detector()
_predictor = None  # call load_predictor() with the actual .dat file path before use


def load_predictor(predictor_path: str):
    """Must be called once before crop_face(), pointing at
    shape_predictor_81_face_landmarks.dat's local path - this file is not
    bundled in the repo, it's downloaded separately (see README)."""
    global _predictor
    _predictor = dlib.shape_predictor(predictor_path)


def _get_keypts(image_rgb, face, predictor):
    shape = predictor(image_rgb, face)
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)


def _align_crop(img_rgb, landmark, outsize=256, scale=1.3):
    """Same ArcFace-style template/margins as DeepfakeBench's own preprocess.py."""
    target_size = [112, 112]
    dst = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                     [33.5493, 92.3655], [62.7299, 92.2041]], dtype=np.float32)
    dst[:, 0] += 8.0
    dst[:, 0] *= outsize / target_size[0]
    dst[:, 1] *= outsize / target_size[1]
    margin = outsize * (scale - 1) / 2.
    dst[:, 0] = (dst[:, 0] + margin) * outsize / (outsize + 2 * margin)
    dst[:, 1] = (dst[:, 1] + margin) * outsize / (outsize + 2 * margin)

    tform = trans.SimilarityTransform()
    tform.estimate(landmark.astype(np.float32), dst)
    M = tform.params[0:2, :]
    return cv2.warpAffine(img_rgb, M, (outsize, outsize))


def crop_face(image_bgr, min_face_size=80):
    """Raw image (BGR, as cv2.imread loads it) -> 256x256 aligned RGB face crop,
    or None if no face detected. Returns RGB deliberately - matches the
    already-validated normalization path, not DeepfakeBench's own BGR output.
    Requires load_predictor() to have been called first."""
    if _predictor is None:
        raise RuntimeError("Call load_predictor(path) before crop_face()")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = _face_detector(rgb, 1)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda r: r.width() * r.height())
    if face.width() < min_face_size or face.height() < min_face_size:
        return None  # face detected but too small/distant to crop reliably
    landmarks = _get_keypts(rgb, face, _predictor)
    return _align_crop(rgb, landmarks, outsize=256, scale=1.3)
