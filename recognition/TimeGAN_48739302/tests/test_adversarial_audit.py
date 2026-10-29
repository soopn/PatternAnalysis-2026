import copy
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from adversarial_audit import StandardisedDiscriminator
from modules import TimeGAN
import train
from dataset import LOBFeatureTransform
from predict import load_model


class AdversarialAuditTest(unittest.TestCase):
    def model(self, **kwargs):
        return TimeGAN(42, hidden_dim=8, noise_dim=4, num_layers=1, seq_len=12, **kwargs)

    def test_conditioning_preserves_predictions_and_source(self):
        model = self.model()
        before = {k: v.clone() for k, v in model.discriminator.state_dict().items()}
        h = 0.5 + 0.03 * torch.randn(6, 12, 8)
        conditioned = StandardisedDiscriminator(model.discriminator, h)
        torch.testing.assert_close(conditioned(h), model.discriminator(h), atol=1e-6, rtol=1e-5)
        for name, value in model.discriminator.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_training_conditioner_fits_once_and_preserves_function(self):
        model = self.model(discriminator_input="standardised")
        h = 0.5 + 0.03 * torch.randn(6, 12, 8)
        expected = model.discriminator(h).detach()
        model.discriminator.fit_input_statistics(h)
        torch.testing.assert_close(model.discriminator(h), expected, atol=1e-6, rtol=1e-5)
        center = model.discriminator.input_center.clone()
        scale = model.discriminator.input_scale.clone()
        model.discriminator.fit_input_statistics(h + 100)
        self.assertTrue(torch.equal(center, model.discriminator.input_center))
        self.assertTrue(torch.equal(scale, model.discriminator.input_scale))

    def test_diagnostic_does_not_populate_parameter_gradients(self):
        model = self.model()
        terms = model.generator_objectives(torch.randn(3, 12, 42))
        result = model.generator_gradient_diagnostics(terms, detailed=True)
        self.assertEqual(len(result["weighted_losses"]), 10)
        self.assertGreater(result["gradients"]["combined"]["adversarial_norm"], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        sum(terms.values()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.generator.parameters()))

    def test_profiling_does_not_change_generator_update(self):
        torch.manual_seed(7)
        plain = self.model()
        profiled = copy.deepcopy(plain)
        x = torch.randn(3, 12, 42)
        torch.manual_seed(11)
        expected = sum(plain.generator_objectives(x).values()).item()
        torch.manual_seed(11)
        loss_plain = plain.train_generator_step(x)
        torch.manual_seed(11)
        loss_profiled = profiled.train_generator_step(x, profile_gradients=True)
        self.assertAlmostEqual(expected, loss_plain, places=5)
        self.assertEqual(loss_plain, loss_profiled)
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(value, profiled.state_dict()[key], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in profiled.discriminator.parameters()))

    def test_auxiliary_scaling_leaves_adversarial_term_unchanged(self):
        model = self.model()
        x = torch.randn(3, 12, 42)
        z = model.sample_noise(3, 12, x.device)
        baseline = model.generator_objectives(x, z)
        model.generator_auxiliary_scale = 0.1
        scaled = model.generator_objectives(x, z)
        for name in baseline:
            factor = 1.0 if name == "adversarial" else 0.1
            torch.testing.assert_close(scaled[name], baseline[name] * factor)

    def test_validation_auc_handles_ties_without_changing_rng(self):
        model = self.model()
        with torch.no_grad():
            model.discriminator.fc.weight.zero_()
            model.discriminator.fc.bias.zero_()
        x = torch.randn(3, 12, 42)
        state = torch.random.get_rng_state().clone()
        metrics = train.discriminator_validation(model, x)
        self.assertEqual(metrics, {"auc_live": 0.5, "accuracy_live": 0.5,
                                   "auc_ema": 0.5, "accuracy_ema": 0.5})
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_training_loop_logs_and_saves_diagnostics(self):
        model = self.model(discriminator_input="standardised")
        model.generator_auxiliary_scale = 0.1
        transform = LOBFeatureTransform()
        transform.mean, transform.std = torch.zeros(42), torch.ones(42)
        x = torch.randn(3, 12, 42)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(train, AUTOENCODER_EPOCHS=1, SUPERVISOR_EPOCHS=1,
                                JOINT_EPOCHS=1, MIN_BEST_EPOCH=1, CKPT_DIR=Path(directory),
                                LOG_GRADIENTS=True), patch.object(
                                    train, "get_batches", side_effect=lambda *_: iter([x])
                                ), contextlib.redirect_stdout(output):
                train.main(model, None, None, transform)
            restored, _ = load_model(Path(directory) / "timegan_best.pt")
            self.assertEqual(restored.generator_auxiliary_scale, 0.1)
            self.assertEqual(restored.discriminator.input_mode, "standardised")
            self.assertEqual(restored.discriminator.input_ready.item(), 1)
            h = torch.randn(2, 12, 8)
            device = next(restored.parameters()).device
            torch.testing.assert_close(restored.discriminator(h.to(device)).cpu(), model.discriminator(h).cpu())
        self.assertIn("val_D_auc_live=", output.getvalue())
        self.assertIn("val_S_ratio=", output.getvalue())
        self.assertIn("[G sampled gradients]", output.getvalue())
        self.assertIn("[G sampled weighted losses]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
