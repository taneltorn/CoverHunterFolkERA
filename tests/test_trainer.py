import logging
import os
import unittest

import torch
from torch.utils.tensorboard import SummaryWriter

from src.model import Model

from src.trainer import Trainer


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
            "covers80": {
                "query_path": "data/covers80/full.txt",
                "ref_path": "data/covers80/full.txt",
                "every_n_epoch_to_dev": 1,
            },
            "train_path": "data/covers80/train.txt",
            "dev_path": "data/covers80/dev.txt",
            "chunk_frame": [1125, 900, 675],
            "chunk_s": 135,
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
        self.log_path = "/tmp/cover_hunter_logs"
        self.checkpoint_dir = "/tmp/cover_hunter_logs"
        self.model_dir = "/tmp/cover_hunter_logs"
        self.sw = SummaryWriter(self.log_path)

    def _test_train(self, device, max_epochs):
        # XXX: find ways to speed this up
        self.hp["device"] = device
        self.trainer = Trainer(
            self.hp,
            Model,
            device,
            self.log_path,
            self.checkpoint_dir,
            self.model_dir,
            only_eval=False,
            first_eval=False,
        )
        self.trainer.summary_writer = self.sw
        self.trainer.configure_optimizer()
        self.trainer.configure_scheduler()
        self.trainer.train(max_epochs=max_epochs)

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_train_cuda(self):
        """
        Ensure that one round of training is working without exception (cuda version).

        XXX: train more times on a smaller batch, in a temporary directory.
        """
        self._test_train("cuda", 1)

    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_train_mps(self):
        """
        Ensure that one round of training is working without exception (mps version).
        """
        self._test_train("mps", 1)

    def _test_logging(self, device):
        with unittest.mock.patch.object(self.sw, "add_scalar") as add_scalar_mock:
            self._test_train(device, 1)

        expected_calls = [
            # 1 because the step is logged for csi
            unittest.mock.call("csi/lr", unittest.mock.ANY, 1),
            unittest.mock.call("csi/ce_loss", unittest.mock.ANY, 1),
            unittest.mock.call("csi/tri_loss", unittest.mock.ANY, 1),
            unittest.mock.call("csi/total", unittest.mock.ANY, 1),
            # 0 because the epoch is logged for the rest
            unittest.mock.call("csi_dev/ce_loss", unittest.mock.ANY, 0),
            unittest.mock.call("csi_dev/tri_loss", unittest.mock.ANY, 0),
            unittest.mock.call("mAP/covers80", unittest.mock.ANY, 0),
            unittest.mock.call("hit_rate/covers80", unittest.mock.ANY, 0),
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
