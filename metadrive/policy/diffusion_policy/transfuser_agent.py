from typing import Any, Dict, List, Optional
import warnings

import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig

from metadrive.policy.diffusion_policy.modules.scheduler import WarmupCosLR
from metadrive.policy.diffusion_policy.transfuser_callback import TransfuserCallback
from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from metadrive.policy.diffusion_policy.transfuser_loss import transfuser_loss
from metadrive.policy.diffusion_policy.transfuser_model_v2 import V2TransfuserModel


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    cfg = cfg.copy()
    obj_type = cfg.pop("type")
    return getattr(obj, obj_type)(**cfg, **kwargs)


class TransfuserAgent(pl.LightningModule):
    """MetaDrive-native LightningModule for TransFuser training and inference."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._transfuser_model = V2TransfuserModel(config)
        self.init_from_pretrained()

    def init_from_pretrained(self):
        if not self._checkpoint_path:
            return
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
        model_state = self.state_dict()
        mismatched_keys = [
            key for key, value in state_dict.items()
            if key in model_state and getattr(value, "shape", None) != model_state[key].shape
        ]
        if mismatched_keys:
            warnings.warn(
                f"Dropping {len(mismatched_keys)} checkpoint key(s) due to shape mismatch "
                f"(will use model-init values): {mismatched_keys}",
                stacklevel=2,
            )
            for key in mismatched_keys:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict[str, torch.Tensor]] = None):
        return self._transfuser_model(features, targets=targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return transfuser_loss(targets, predictions, self._config)

    def training_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self(features, targets)
        loss_dict = self.compute_loss(features, targets, predictions)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, prog_bar=("loss" in loss_dict), batch_size=features["camera_feature"].shape[0])
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self(features, targets)
        loss_dict = self.compute_loss(features, targets, predictions)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, prog_bar=("loss" in loss_dict), batch_size=features["camera_feature"].shape[0])
        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self(features, targets)
        loss_dict = self.compute_loss(features, targets, predictions)
        self.log_dict({f"test/{k}": v for k, v in loss_dict.items()}, batch_size=features["camera_feature"].shape[0])
        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer_cfg = DictConfig(
            dict(
                type=self._config.optimizer_type,
                lr=self._lr,
                weight_decay=self._config.weight_decay,
            )
        )
        optimizer = build_from_configs(optim, optimizer_cfg, params=self._transfuser_model.parameters())
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=self._config.min_lr,
            epochs=self._config.max_epochs,
            warmup_epochs=self._config.warmup_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [TransfuserCallback(self._config)]
