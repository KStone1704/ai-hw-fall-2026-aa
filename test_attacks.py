import json
import unittest
from pathlib import Path

import torch
from torch import nn

from attacks import METHODS, attack
from model import CNN, PixelsToLogits


class AttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        cls.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).eval()
        cls.x = torch.rand(4, 1, 28, 28)
        cls.y = torch.tensor([0, 1, 2, 3])

    def test_zero_budget_and_no_input_mutation(self):
        before = self.x.clone()
        for method in METHODS:
            torch.testing.assert_close(attack(self.net, self.x, self.y, method, 0), before)
            attack(self.net, self.x, self.y, method, 0.15)
            torch.testing.assert_close(self.x, before)

    def test_bounds_and_no_parameter_grads(self):
        for method in METHODS:
            for eps in (0.05, 0.15, 0.3):
                adv = attack(self.net, self.x, self.y, method, eps)
                self.assertLessEqual((adv - self.x).abs().max().item(), eps + 1e-6)
                self.assertGreaterEqual(adv.min().item(), 0)
                self.assertLessEqual(adv.max().item(), 1)
                self.assertTrue(all(p.grad is None for p in self.net.parameters()))

    def test_fgsm_matches_direct_gradient(self):
        x = self.x.clone().requires_grad_(True)
        loss = nn.functional.cross_entropy(self.net(x), self.y, reduction='sum')
        grad, = torch.autograd.grad(loss, x)
        expected = (x + 0.15 * grad.sign()).clamp(0, 1).detach()
        torch.testing.assert_close(attack(self.net, self.x, self.y, 'fgsm', .15), expected)

    def test_seeded_pgd_repeats(self):
        args = self.net, self.x, self.y, 'pgd', .15
        torch.testing.assert_close(attack(*args, seed=3), attack(*args, seed=3))

    def test_cnn_shape_and_size(self):
        net = PixelsToLogits(CNN(), .13, .31)
        self.assertEqual(net(self.x).shape, (4, 10))
        self.assertEqual(sum(p.numel() for p in net.parameters()), 101129)

    def test_recorded_results(self):
        path = Path(__file__).with_name('results.json')
        payload = json.loads(path.read_text())
        n, clean = payload['test_size'], payload['clean_correct']
        self.assertEqual((n, clean), (10000, 9895))
        self.assertEqual(len(payload['results']), 9)
        for row in payload['results']:
            fooled = row['clean_correct_fooled']
            robust = row['correct_after_every_restart']
            recovered = row['clean_wrong_correct_after_every_restart']
            self.assertEqual(robust, clean - fooled + recovered)
            self.assertAlmostEqual(row['attack_success_rate'], fooled / clean)
            self.assertAlmostEqual(row['adversarial_accuracy'], robust / n)
            self.assertLessEqual(row['max_linf_observed'], row['epsilon'] + 1e-6)
