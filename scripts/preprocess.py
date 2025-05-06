import logging
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_image(image_path: str, output_dir: str, size=(512, 512)):
    """
    Resize and normalize image, save tensor to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(image)

    save_path = Path(output_dir) / (Path(image_path).stem + ".pt")
    torch.save(tensor, save_path)
    logger.info(f"Processed image saved to: {save_path}")


def preprocess_folder(input_dir: str, output_dir: str):
    logger.info(f"Preprocessing images from {input_dir} to {output_dir}")
    input_dir = Path(input_dir)
    for file in input_dir.glob("*.jpg"):
        preprocess_image(str(file), output_dir)


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config")
        preprocess_folder(cfg.paths.raw_images, cfg.paths.processed_images)
