import os
import random
import json
from pathlib import Path
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)

def ensure_dirs():
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/test").mkdir(parents=True, exist_ok=True)

def resize_and_copy(src_dir, dst_dir, img_size=IMG_SIZE):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    ensure_dirs()
    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir(): continue
        out_class_dir = dst_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        for img_path in class_dir.glob("*"):
            try:
                im = Image.open(img_path).convert("RGB")
                im = im.resize(img_size)
                out_path = out_class_dir / img_path.name
                im.save(out_path)
            except Exception as e:
                print(f"skip {img_path}: {e}")

def get_generators(batch_size=32, train_dir="data/train", val_dir="data/test"):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(train_dir,
                                                  target_size=IMG_SIZE,
                                                  batch_size=batch_size,
                                                  class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(val_dir,
                                              target_size=IMG_SIZE,
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False)
    # Save class indices
    with open("models/class_indices.json","w") as f:
        json.dump(train_gen.class_indices, f)
    return train_gen, val_gen

if __name__ == "__main__":
    resize_and_copy("data/raw", "data/train")
   
