import unittest

import torch

from modules import TimeGAN


class ReconstructionEventsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.real = torch.zeros(2, 8, 42)
        self.real[:, 3:, 1] = 2.0
        self.real[:, 5:, 3] = -1.0
        self.std = torch.ones(42)

    def test_exact_reconstruction_has_zero_loss(self):
        parts = TimeGAN.reconstruction_event_losses(self.real, self.real, self.std)
        self.assertTrue(all(value.item() == 0 for value in parts.values()))

    def test_attenuated_and_mistimed_jumps_are_penalised(self):
        small = TimeGAN.reconstruction_event_losses(self.real, self.real * 0.1, self.std)
        shifted = TimeGAN.reconstruction_event_losses(self.real, self.real.roll(1, 1), self.std)
        self.assertGreater(small["jumps"].item(), 0.5)
        self.assertGreater(shifted["alignment"].item(), 0)
        self.assertGreater(shifted["coordination"].item(), 0)

    def test_original_units_and_offsets(self):
        fake = self.real * 0.5
        reference = TimeGAN.reconstruction_event_losses(self.real, fake, self.std)
        normalised = TimeGAN.reconstruction_event_losses(
            self.real / 2 + 4, fake / 2 + 4, self.std * 2
        )
        for name in reference:
            torch.testing.assert_close(reference[name], normalised[name])

    def test_constant_and_single_step_batches_have_finite_gradients(self):
        for steps in (1, 8):
            real = torch.zeros(2, steps, 42)
            fake = torch.randn_like(real, requires_grad=True)
            loss = sum(TimeGAN.reconstruction_event_losses(real, fake, self.std).values())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(fake.grad).all())

    def test_both_training_stages_apply_event_weight(self):
        for stage in ("pretrain_autoencoder", "train_embedder_step"):
            model = TimeGAN(42, hidden_dim=8, noise_dim=4, num_layers=1)
            x = self.real + torch.randn_like(self.real) * 0.01
            parts = model.reconstruction_event_losses(x, model.reconstruct(x), model.feature_std)
            event_loss = sum(parts.values()).item()
            before = model.recoverer.fc.weight.detach().clone()
            # A zero learning rate permits an exact objective comparison.
            for group in model.optimizer_er.param_groups:
                group["lr"] = 0
            model.reconstruction_event_weight = 0
            base = getattr(model, stage)(x)
            model.reconstruction_event_weight = 2
            weighted = getattr(model, stage)(x)
            self.assertAlmostEqual(weighted - base, 2 * event_loss, places=4)
            for group in model.optimizer_er.param_groups:
                group["lr"] = 1e-3
            getattr(model, stage)(x)
            self.assertFalse(torch.equal(before, model.recoverer.fc.weight))


if __name__ == "__main__":
    unittest.main()
