
import json
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model
 
# ── Load models ───────────────────────────────────────────────
feature_extractor = load_model("model/feature_extractor.keras")
binary_model      = load_model("model/binary_model.keras")
 
with open("model/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
 
with open("model/label_map.json", "r") as f:
    label_map = json.load(f)
 
# ── Constants ─────────────────────────────────────────────────
THRESHOLD = 0.5
IMG_SIZE  = (256, 256)
 
# ── Preprocessing ─────────────────────────────────────────────
def preprocess_for_cnn(image):
    """
    Accepts a PIL image (from Streamlit file uploader).
    Handles transparent/black backgrounds robustly.
    """
    # PIL → numpy BGR
    img = np.array(image)
 
    # RGBA → composite onto white background
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, alpha = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        white = np.full(b.shape, 255, dtype=np.uint8)
        mask  = alpha.astype("float32") / 255.0
        img   = cv2.merge([
            (b * mask + white * (1 - mask)).astype(np.uint8),
            (g * mask + white * (1 - mask)).astype(np.uint8),
            (r * mask + white * (1 - mask)).astype(np.uint8),
        ])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
 
    # Flood-fill black background from corners → replace with white
    if img.ndim == 3 and img.shape[2] == 3:
        border_px = np.concatenate([
            img[:3,  :  ].reshape(-1, 3),
            img[-3:, :  ].reshape(-1, 3),
            img[:,  :3  ].reshape(-1, 3),
            img[:, -3:  ].reshape(-1, 3),
        ], axis=0)
        border_rounded  = (border_px // 10) * 10
        unique, counts  = np.unique(border_rounded, axis=0, return_counts=True)
        bg_color        = unique[np.argmax(counts)]
 
        if np.all(bg_color < 15):
            h, w        = img.shape[:2]
            flood_mask  = np.zeros((h + 2, w + 2), dtype=np.uint8)
            temp        = img.copy()
            for (fy, fx) in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
                cv2.floodFill(temp, flood_mask, (fx, fy),
                              (128, 128, 128),
                              loDiff=(15,15,15), upDiff=(15,15,15))
            filled = (temp[:,:,0]==128)&(temp[:,:,1]==128)&(temp[:,:,2]==128)
            img[filled] = [255, 255, 255]
 
    # Resize, convert, normalize
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img
 
# ── Prediction ────────────────────────────────────────────────
def predict_image(image):
    """
    Returns (label, probability) where:
      label → "clean" or "stego"
      probability → float 0.0–1.0 (probability of being stego)
    """
    img        = preprocess_for_cnn(image)
    img_batch  = np.expand_dims(img, axis=0)   # (1, 256, 256, 3)
 
    # Extract CNN features
    features   = feature_extractor.predict(img_batch, verbose=0)
 
    # XGBoost prediction (more reliable)
    prob       = xgb_model.predict_proba(features)[0][1]
    label      = "stego" if prob >= THRESHOLD else "clean"
 
    return label, float(prob)
