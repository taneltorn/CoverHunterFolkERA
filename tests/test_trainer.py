import logging
import os
import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import Model
from src.trainer import train_one_epoch, validate
from src.scheduler import UserDefineExponentialLR
from src.dataset import AudioFeatDataset, MPerClassSampler


class TestTrainer(unittest.TestCase):
    """
    Test for trainer code.

    XXX: Too much code is being copied from tools/train.py, let's share it in a
    Trainer class.
    """

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.hp = {
            "batch_size": 32,
            # XXX: add reduced sample data in tests or dedicated dir, maybe try with train-sample
            "train_path": "data/covers80/train.txt",
            "dev_path": "data/covers80/dev.txt",
            "chunk_frame": [1125, 900, 675],
            "mode": "random",
            "learning_rate": 0.001,
            "mean_size": 3,
            "num_workers": 1,
            "m_per_class": 8,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.99,  # 0.9975
            "min_lr": 0.0001,
            "input_dim": 96,
            "embed_dim": 128,
            "encoder": {
                "output_dims": 128,
                "num_blocks": 6,
                "attention_dim": 256,
            },
            "ce": {
                "output_dims": 3000,
                "gamma": 2,
                "weight": 1.0,
            },
            "triplet": {
                "margin": 0.3,
                "weight": 0.1,
            },
            "center": {
                "weight": 0,
            },
        }
        log_path = os.path.join("/tmp", "cover_hunter_logs")
        os.makedirs(log_path, exist_ok=True)
        self.sw = SummaryWriter(log_path)

    def _test_train(self, device, epochs):
        # XXX: find ways to speed this up
        self.hp["device"] = device
        self.model = Model(self.hp).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.hp["learning_rate"],
            betas=[self.hp["adam_b1"], self.hp["adam_b2"]],
        )
        self.scheduler = UserDefineExponentialLR(
            self.optimizer,
            gamma=self.hp["lr_decay"],
            min_lr=self.hp["min_lr"],
            last_epoch=-1,
        )
        self.train_loader_lst = []
        for chunk_len in self.hp["chunk_frame"]:
            train_dataset = AudioFeatDataset(
                self.hp,
                self.hp["train_path"],
                train=True,
                mode=self.hp["mode"],
                chunk_len=chunk_len * self.hp["mean_size"],
                logger=None,
            )
            sampler = MPerClassSampler(
                data_path=self.hp["train_path"],
                m=self.hp["m_per_class"],
                batch_size=self.hp["batch_size"],
                distribute=False,
                logger=None,
            )
            train_loader = DataLoader(
                train_dataset,
                num_workers=self.hp["num_workers"],
                shuffle=(sampler is None),
                sampler=sampler,
                batch_size=self.hp["batch_size"],
                pin_memory=True,
                drop_last=True,
            )
            self.train_loader_lst.append(train_loader)

        infer_len = self.hp["chunk_frame"][0] * self.hp["mean_size"]
        dataset = AudioFeatDataset(
            self.hp,
            self.hp["dev_path"],
            chunk_len=infer_len,
            mode=self.hp["mode"],
            logger=None,
        )
        sampler = MPerClassSampler(
            data_path=self.hp["dev_path"],
            m=self.hp["m_per_class"],
            batch_size=self.hp["batch_size"],
            distribute=False,
            logger=None,
        )
        self.dev_loader = DataLoader(
            dataset,
            num_workers=1,
            shuffle=False,
            sampler=sampler,
            batch_size=self.hp["batch_size"],
            pin_memory=True,
            collate_fn=None,
            drop_last=False,
        )

        step = 1
        for _epoch in range(1, 1 + epochs):
            train_step = None
            step = train_one_epoch(
                self.model,
                self.optimizer,
                self.scheduler,
                self.train_loader_lst,
                step,
                train_step=train_step,
                device=device,
                sw=self.sw,
                logger=None,
            )

        dev_res = validate(
            self.model,
            self.dev_loader,
            "dev",
            epoch_num=_epoch,
            device=device,
            sw=self.sw,
            logger=None,
        )
        validation_loss = dev_res["ce_loss"] / dev_res["count"]
        return validation_loss

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_train_cuda(self):
        """
        Ensure that one round of training is working without exception (cuda version).

        XXX: train more times on a smaller batch, in a temporary directory.
        """
        _validation_loss = self._test_train("cuda", 1)

    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_train_mps(self):
        """
        Ensure that one round of training is working without exception (mps version).
        """
        _validation_loss = self._test_train("mps", 1)

    def _test_logging(self, device):
        with unittest.mock.patch.object(self.sw, "add_scalar") as add_scalar_mock:
            self._test_train(device, 1)

        expected_calls = [
            unittest.mock.call("csi_dev/ce_loss", unittest.mock.ANY, 1),
            unittest.mock.call("csi_dev/tri_loss", unittest.mock.ANY, 1),
        ]
        add_scalar_mock.assert_has_calls(expected_calls)

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_logging_cuda(self):
        """
        Check that the SummaryWriter is properly called (cuda version).

        XXX: only part of the code calling it is actually tested here.
        """
        self._test_logging("cuda")

    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_logging_mps(self):
        """
        Check that the SummaryWriter is properly called (mps version).
        """
        self._test_logging("mps")


if __name__ == "__main__":
    unittest.main()
