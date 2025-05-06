import pytorch_lightning as pl
from background_replacer.models.unet import UNetModel
from background_replacer.data.datamodule import SegmentationDataModule
import torch.nn.functional as F
import torch
import hydra
from omegaconf import DictConfig

class LitSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNetModel()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

@hydra.main(config_path="../configs", config_name="config")
def train_main(cfg: DictConfig):
    dm = SegmentationDataModule(cfg.data.image_dir, cfg.data.mask_dir, cfg.data.batch_size)
    dm.setup()
    model = LitSegmentation()
    trainer = pl.Trainer(max_epochs=cfg.train.epochs)
    trainer.fit(model, dm)