from typing import Any, List
import torch
from torch import nn, optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MaxMetric, Accuracy


class BaseModule(LightningModule):
    def __init__(
            self,
            net: nn.Module,
            learning_rate: float,
            weight_decay: float
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc_best = MaxMetric()
        
    def forward(self, x: torch.Tensor):
        return self.net(x)
        
    def on_train_start(self):
        self.val_acc_best.reset()

    def step(self, data: Any):
        y_hat = self.forward(data)
        loss = self.loss_fn(y_hat, data.y)
        batch_size = y_hat.size(0)
        return y_hat, loss, batch_size

    def training_step(self, data: Any, batch_idx: int):
        y_hat, loss, batch_size = self.step(data)

        acc = self.train_acc(y_hat, data.y)
        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True,
                 batch_size=batch_size, sync_dist=True)
        self.log('train/acc', acc, on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=batch_size, sync_dist=True)
        return {"loss": loss}

    def train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(self, data: Any, batch_idx: int):
        y_hat, loss, batch_size = self.step(data)

        acc = self.val_acc(y_hat, data.y)
        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=False,
                 batch_size=batch_size, sync_dist=True)
        self.log('val/acc', acc, on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=batch_size, sync_dist=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute() 
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, data: Any, batch_idx: int):
        y_hat, loss, batch_size = self.step(data)

        acc = self.test_acc(y_hat, data.y)
        self.log('test/loss', loss, on_epoch=True, on_step=False, prog_bar=False,
                 batch_size=batch_size, sync_dist=True)
        self.log('test/acc', acc, on_epoch=True, on_step=False, prog_bar=True,
                 batch_size=batch_size, sync_dist=True)
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val/loss'}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
