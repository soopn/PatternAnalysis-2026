import unittest

import torch
from torch import nn

from generation_path_checks import teacher_paths
from modules import TimeGAN
from training_balance_ablation import make_candidate


class ShiftSupervisor(nn.Module):
    def forward(self, h):
        return torch.cat([h[:, 1:], torch.zeros_like(h[:, :1])], dim=1)


class CumulativeRecovery(nn.Module):
    def forward(self, h):
        return torch.cumsum(h, dim=1)


class GenerationPathsTest(unittest.TestCase):
    def test_teacher_targets_and_decoder_prefix_are_aligned(self):
        h = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
        reconstruction, predicted, latent = teacher_paths(h, ShiftSupervisor(), CumulativeRecovery())
        torch.testing.assert_close(latent, h[:, 1:])
        torch.testing.assert_close(predicted, reconstruction[:, 1:])

    def test_continuations_have_identical_initial_generator_weights(self):
        source = TimeGAN(42, hidden_dim=8, noise_dim=4, num_layers=1, seq_len=12)
        source.feature_mean.fill_(2)
        source.feature_std.fill_(3)
        conditioned = make_candidate(source, {"discriminator_input": "standardised",
                                               "generator_auxiliary_scale": 0.1})
        z = source.sample_noise(3, 12, next(source.parameters()).device)
        expected = source.recoverer(source.supervisor(source.generator(z)))
        actual = conditioned.recoverer(conditioned.supervisor(conditioned.generator(z)))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(conditioned.generator_auxiliary_scale, 0.1)
        self.assertEqual(conditioned.discriminator.input_ready.item(), 0)
        torch.testing.assert_close(conditioned.feature_mean, source.feature_mean)
        self.assertFalse(conditioned.optimizer_g.state)


if __name__ == "__main__":
    unittest.main()
