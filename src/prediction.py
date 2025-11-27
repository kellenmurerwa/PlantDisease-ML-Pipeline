import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage
import json

MODEL_PATH = "models/plantnet_v1.h5"
CLASS_INDICES = "models/class_indices.json"

def load():
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES) as f:
        class_indices = json.load(f)
    inv_map = {v:k for k,v in class_indices.items()}
    return model, inv_map

def preprocess(img_path, target_size=(224,224)):
    img = kimage.load_img(img_path, target_size=target_size)
    arr = kimage.img_to_array(img)/255.0
    arr = np.expand_dims(arr, 0)
    return arr

def predict(img_path):
    model, inv_map = load()
    x = preprocess(img_path)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    return {"class": inv_map[idx], "probability": float(probs[idx]), "all_probs": probs.tolist()}

if __name__ == "__main__":
    import sys
    print(predict(sys.argv[1]))
