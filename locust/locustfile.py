from locust import HttpUser, task, between
from pathlib import Path
import os

# Folder containing all images to test
IMAGE_FOLDER = Path("locust")  # Change if your folder is different
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Collect all images in the folder
image_paths = [p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

class PlantDiseaseUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_all_images(self):
        for image_path in image_paths:
            if not image_path.exists():
                print(f"⚠️ Image file not found: {image_path}")
                continue

            with open(image_path, "rb") as f:
                files = {"file": (image_path.name, f, "image/jpeg")}
                response = self.client.post("/predict", files=files)

            try:
                print(f"Response for {image_path.name}:", response.json())
            except Exception:
                print(f"Non-JSON response for {image_path.name}:", response.text)
