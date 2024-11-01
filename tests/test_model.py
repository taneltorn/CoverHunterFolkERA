import random
import unittest

import numpy as np
import torch

from src.model import Model

import logging


class TestModel(unittest.TestCase):
    """
    Test for model.
    """

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.device = "cpu"
        self.hp = {
            "device": self.device,
            "batch_size": 32,
            "learning_rate": 0.001,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "input_dim": 96,
            "embed_dim": 128,
            "encoder": {
                "output_dims": 128,
                "num_blocks": 6,
                "attention_dim": 256,
            },
            "foc": {
                "output_dims": 3000,
                "gamma": 2,
                "weight": 1.0,
            },
            "triplet": {
                "margin": 0.3,
                "weight": 0.1,
            },
            "center": {
                "weight": 0.001,
            },
        }
        self.model = Model(self.hp)
        self.input_tensor = torch.zeros((32, 1125, 96))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.hp["learning_rate"],
            betas=[self.hp["adam_b1"], self.hp["adam_b2"]],
        )

    @staticmethod
    def make_deterministic(seed=1234) -> None:
        """
        Set the randomness with a given seed.
        """
        # PyTorch
        torch.manual_seed(seed)
        # Numpy
        np.random.seed(seed)
        # Built-in Python
        random.seed(seed)

    @torch.no_grad()
    def test_shape(self):
        """
        Test that the model output shape has not changed.
        """
        output_tensor = self.model(self.input_tensor)
        expected_shape = (32, 128)
        self.assertEqual(expected_shape, output_tensor.shape)

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_device_moving_cuda(self):
        """
        Test that the model can be moved from cpu to gpu with similar results.
        """
        model = Model(self.hp)
        model_on_gpu = model.to("cuda")
        model_back_on_cpu = model_on_gpu.cpu()

        inputs = torch.randn((32, 1125, 96))

        self.make_deterministic()
        outputs_cpu = model(inputs)
        self.make_deterministic()
        outputs_gpu = model_on_gpu(inputs)
        self.make_deterministic()
        outputs_back_on_cpu = model_back_on_cpu(inputs)

        torch.testing.assert_close(outputs_cpu, outputs_gpu)
        torch.testing.assert_close(outputs_cpu, outputs_back_on_cpu)

    @torch.no_grad()
    @unittest.skipUnless(torch.backends.mps.is_available(), "No MPS was detected")
    def test_device_moving_mps(self):
        """
        Test that the model can be moved from cpu to mps with similar results.
        """
        model = Model(self.hp)
        model_on_mps = model.to("mps")
        model_back_on_cpu = model_on_mps.cpu()

        inputs = torch.randn((32, 1125, 96))

        self.make_deterministic()
        outputs_cpu = model(inputs)
        self.make_deterministic()
        outputs_mps = model_on_mps(inputs)
        self.make_deterministic()
        outputs_back_on_cpu = model_back_on_cpu(inputs)

        torch.testing.assert_close(outputs_cpu, outputs_mps)
        torch.testing.assert_close(outputs_cpu, outputs_back_on_cpu)

    def _test_loss_function(self):
        """
        XXX: Fix this test function.
        """
        input_shape = (32, 1125, 96)
        input_tensor = torch.zeros(input_shape)
        label_tensor = torch.zeros(32)
        self.model.eval()
        total_loss, losses = self.model.compute_loss(input_tensor, label_tensor)
        self.model.train()
        total_loss.backward()
        gradients = [var.grad for var in self.model.parameters()]
        expected_gradients = [torch.zeros_like(var) for var in self.model.parameters()]
        for var in self.model.parameters():
            print(var.grad)

        torch.testing.assert_close(expected_gradients, gradients)

    def test_all_parameters_updated(self):
        """
        Ensure that we have no dead subgraph in the network architecture.
        """
        inputs = torch.randn((32, 1125, 96))
        labels = torch.zeros((32))
        total_loss, losses = self.model.compute_loss(inputs, labels)
        total_loss.backward()
        self.optimizer.step()

        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0.0, torch.sum(param.grad**2))


if __name__ == "__main__":
    unittest.main()
