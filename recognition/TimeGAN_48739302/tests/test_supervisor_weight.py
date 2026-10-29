import tempfile
import unittest
from pathlib import Path

import torch

from dataset import LOBFeatureTransform
from modules import TimeGAN
from predict import load_model
from training_balance_ablation import make_candidate
from train import save_checkpoint


class SupervisorWeightTest(unittest.TestCase):
    def source(self):
        model = TimeGAN(42, hidden_dim=8, noise_dim=4, num_layers=1, seq_len=12,
                        generator_auxiliary_scale=0.1, discriminator_input="standardised")
        with torch.no_grad():
            model.discriminator.fit_input_statistics(model.embedder(torch.randn(3, 12, 42)))
        return model

    def test_B_changes_only_supervised_term_and_its_gradient(self):
        source = self.source()
        a = make_candidate(source, {"supervisor_loss_weight": 10.0})
        b = make_candidate(source, {"supervisor_loss_weight": 100.0})
        x = torch.randn(3, 12, 42)
        z = source.sample_noise(3, 12, x.device)
        ta, tb = a.generator_objectives(x, z), b.generator_objectives(x, z)
        for name in ta:
            torch.testing.assert_close(tb[name], ta[name] * (10 if name == "supervised" else 1))
        ga = torch.autograd.grad(ta["supervised"], tuple(a.supervisor.parameters()))
        gb = torch.autograd.grad(tb["supervised"], tuple(b.supervisor.parameters()))
        for first, second in zip(ga, gb):
            torch.testing.assert_close(second, first * 10)
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)

    def test_legacy_default_preserves_old_scaling(self):
        source = self.source()
        self.assertEqual(source.effective_supervisor_loss_weight, 10)
        source.generator_auxiliary_scale = 1
        self.assertEqual(source.effective_supervisor_loss_weight, 100)
        source.supervisor_loss_weight = 100
        source.generator_auxiliary_scale = 0.1
        self.assertEqual(source.effective_supervisor_loss_weight, 100)

    def test_ER_updates_are_unchanged(self):
        source = self.source()
        a = make_candidate(source, {"supervisor_loss_weight": 10.0})
        b = make_candidate(source, {"supervisor_loss_weight": 100.0})
        x = torch.randn(3, 12, 42)
        self.assertEqual(a.train_embedder_step(x), b.train_embedder_step(x))
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)

    def test_checkpoint_restores_independent_weight(self):
        model = make_candidate(self.source(), {"supervisor_loss_weight": 100.0})
        transform = LOBFeatureTransform()
        transform.mean, transform.std = torch.zeros(42), torch.ones(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "B.pt"
            save_checkpoint(model, transform, path)
            restored, _ = load_model(path)
        self.assertEqual(restored.effective_supervisor_loss_weight, 100)
        self.assertEqual(restored.generator_auxiliary_scale, 0.1)
        self.assertEqual(restored.discriminator.input_mode, "standardised")
        self.assertEqual(restored.discriminator.input_ready.item(), 1)


if __name__ == "__main__":
    unittest.main()
