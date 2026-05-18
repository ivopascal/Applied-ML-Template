"""Scanner script that runs through target path and subfolders and indexes all files into a dataframe and parquet file"""

import pandas as pd
import os
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
NAME_MAP = {
    "adm": "ADM",
    "glide": "Glide",
    "midjourney": "Midjourney",
    "sdv4": "Stable Diffusion v1.4",
    "sdv5": "Stable Diffusion v1.5",
    "vqdm": "VQDM",
    "wukong": "Wukong",
}


class DatasetBuilder:
    def __init__(self, size: int = 1000, root: str = "./data/raw"):
        self.root_dir = Path(root).resolve()
        self.data = []
        self.size = size

    def get_paths(self) -> list:
        file_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                    file_paths.append(os.path.join(root, file))
        print(f"{len(file_paths):,} images found")
        return file_paths

    def scan(self):
        errors = []
        image_records = []
        file_paths = self.get_paths()
        for file_path in tqdm(file_paths, desc="parsing img"):
            try:
                file_size = os.path.getsize(file_path)

                with Image.open(file_path) as img:
                    w, h = img.size
                    path_obj = Path(file_path)
                    folder_name = path_obj.parent.name
                    model_type = NAME_MAP.get(folder_name, folder_name)
                    parts = [p.lower() for p in path_obj.parts]
                    category = "ai" if "ai" in parts else "nature"

                    image_records.append(
                        {
                            "path": file_path,
                            "filename": path_obj.name,
                            "model_type": model_type,
                            "category": category,
                            "width": w,
                            "height": h,
                            "aspect_ratio": round(w / h, 2),
                            "megapixels": round((w * h) / 1e6, 2),
                            "size_bytes": file_size,
                            "format": img.format,
                        }
                    )
            except Exception as e:
                errors.append((file_path, str(e)))
                continue

        print(f"Skipped {len(errors)} files due to errors.")
        self.data = pd.DataFrame(image_records)

    def save(self, name: str = "data/dataset.parquet"):
        if self.data.empty:
            print("No data to save.")
            return
        self.data.to_parquet(name, engine="pyarrow", index=False)

    def filter(self):
        if self.data.empty:
            print("No data to filter.")
            return
        self.data = self.data[~self.data["filename"].str.startswith(("[DUPE]", "[LQ]"))]

    def undersample(self):
        if self.data.empty:
            print("No data to sample.")
            return
        self.data = self.data.groupby(["category", "model_type"]).sample(
            n=self.size, random_state=42
        )

    def read(self, name: str = "data/dataset.parquet"):
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"No dataset file found at {path.resolve()}")
        self.data = pd.read_parquet(path)

    def __len__(self):
        return len(self.data)
