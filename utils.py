from model import load_model
import numpy as np
from typing import Dict, Any, List
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

face_model = load_model()

def __forward(model: Model, img: np.ndarray) -> List[float]:
    return model(img, training=False).numpy()[0]


def compare(model: Model, img1_path: str, img2_path: str) -> Dict[str, Any]:
    emb1 = __calculate_embedding(model, img1_path)
    emb2 = __calculate_embedding(model, img2_path)
    distance = __calculate_cosine(emb1, emb2)
    return {"confidence": 1.0, "distance": distance, "equal": distance <= 0.4}


def __find_face(img_path):
    img = cv2.imread(img_path)
    
    return {
        "face": img[:, :, ::-1],
        "facial_area": {
            "x": 0,
            "y": 0,
            "w": img.shape[1],
            "h": img.shape[0],
            "left_eye": None,
            "right_eye": None,
            },
        "confidence": 0,
    }

def __resize(img, target_size):
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    if img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img

def __calculate_embedding(model: Model, img_path: str) -> List[float]:
    face = __find_face(img_path=img_path)
    img = face["face"][:, :, ::-1]
    target_size = (160, 160)

    img = __resize(img, (target_size[1], target_size[0]))
    embedding = __forward(model, img)
    return embedding

def __calculate_cosine(emb1, emb2) -> np.float64:
    a = np.matmul(np.transpose(emb1), emb2)
    b = np.sum(np.multiply(emb1, emb1))
    c = np.sum(np.multiply(emb2, emb2))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
