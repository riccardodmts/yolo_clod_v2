from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union

from accelerate import Accelerator
from tqdm import tqdm

from torch.cuda import device
import torch.nn as nn
import torch

@dataclass
class Stage:
    train: int = 0
    val: int = 1
    test: int = 2

class MicroMind(ABC):
    def __init__(self):
        # here we should handle devices etc.
        self.modules = torch.nn.ModuleList([]) # init empty modules list

        self.device = "cpu"
        self.accelerator = Accelerator()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def compute_loss(self, pred, batch):
        pass

    def configure_optimizers(self):
        opt_conf = {"lr": 0.001, "momentum": 0.9}
        opt = torch.optim.SGD(self.modules.parameters(), **opt_conf)
        return opt, None    # None is for learning rate sched

    def __call__(self, *x, **xv):
        return self.forward(*x, **xv)

    def on_train_start(self):
        self.opt, self.lr_sched = self.configure_optimizers()

        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        convert = [self.modules, self.opt, self.lr_sched] + list(self.datasets.values())
        accelerated = self.accelerator.prepare(convert)
        self.modules, self.opt, self.lr_sched = accelerated[:3]
        for i, key in enumerate(self.datasets):
            self.datasets[key] = accelerated[-(i + 1)]

    def train(
            self,
            epochs: int = 1,
            datasets: Dict = {},
            debug: bool = False
        ) -> None:
        self.datasets = datasets
        assert "train" in self.datasets, "Training dataloader was not specified."
        assert epochs > 0, "You must specify at least one epoch."

        self.debug = debug
        self.modules.train()

        self.on_train_start()

        for e in range(epochs):
            pbar = tqdm(self.datasets["train"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
            loss_epoch = 0
            pbar.set_description(f"Running epoch {e + 1}/{epochs}")
            for idx, batch in enumerate(pbar):
                self.opt.zero_grad()
    
                loss = self.compute_loss(self(batch), batch)

                self.accelerator.backward(loss)
                self.opt.step()
    
                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch/(idx + 1))
    
                if self.debug and idx > 10: break

            pbar.close()

            if e >= 1 and self.debug: break     # not sure this is getting called

            if "val" in datasets: self.validate()

        return None

    @torch.no_grad()
    def validate(self) -> None:
        assert "val" in self.datasets, "Validation dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(self.datasets["val"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
        loss_epoch = 0
        pbar.set_description(f"Validation...")
        for idx, batch in enumerate(pbar):
            self.opt.zero_grad()

            loss = self.compute_loss(self(batch), batch)

            loss_epoch += loss.item()
            pbar.set_postfix(loss=loss_epoch/(idx + 1))

            if self.debug and idx > 10: break

        pbar.close()

        return None

    @torch.no_grad()
    def test(self, datasets: Dict = {}) -> None:
        assert "test" in self.datasets, "Test dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(self.datasets["test"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
        loss_epoch = 0
        pbar.set_description(f"Testing...")
        for idx, batch in enumerate(pbar):
            self.opt.zero_grad()

            loss = self.compute_loss(self(batch), batch)

            loss_epoch += loss.item()
            pbar.set_postfix(loss=loss_epoch/(idx + 1))

        pbar.close()

        return None

