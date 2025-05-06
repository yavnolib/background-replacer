import os

import cv2
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from background_replacer.models.unet import UNetModel
from background_replacer.utils.transforms import get_val_transforms


@hydra.main(config_path="../configs", config_name="config")
def infer_main(cfg: DictConfig):
    model = UNetModel()
    model.load_state_dict(torch.load(cfg.infer.model_path))
    model.eval()

    transforms = get_val_transforms()

    image_paths = sorted(
        [
            os.path.join(cfg.infer.input_path, x)
            for x in os.listdir(cfg.infer.input_path)
        ]
    )
    results = []

    for img_path in tqdm(image_paths):
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transforms(image=image_rgb)["image"].unsqueeze(0)
        with torch.no_grad():
            mask_pred = torch.sigmoid(model(augmented)).squeeze().cpu().numpy()
        results.append(
            {"image": os.path.basename(img_path), "mean_mask_val": mask_pred.mean()}
        )

    pd.DataFrame(results).to_csv(cfg.infer.output_file, index=False)
