#################################################################################
#                                                                               #
#  Copyright (C) 2026 Seif Kandil (k0T0z)                                       #
#                                                                               #
#  This file is a part of the ENIGMA Development Environment.                   #
#                                                                               #
#                                                                               #
#  ENIGMA is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software    #
#  Foundation, version 3 of the license or any later version.                   #
#                                                                               #
#  This application and its source code is distributed AS-IS, WITHOUT ANY       #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS    #
#  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more        #
#  details.                                                                     #
#                                                                               #
#  You should have recieved a copy of the GNU General Public License along      #
#  with this code. If not, see <http://www.gnu.org/licenses/>                   #
#                                                                               #
#  ENIGMA is an environment designed to create games and other programs with a  #
#  high-level, fully compilable language. Developers of ENIGMA or anything      #
#  associated with ENIGMA are in no way responsible for its users or            #
#  applications created by its users, or damages caused by the environment      #
#  or programs made in the environment.                                         #
#                                                                               #
#################################################################################

"""Tests for :mod:`sga.analysis`, the target's Fourier signature and its seed screen.

``analysis`` depends only on ``numpy`` and the standard library, so these tests run without the
compiled ``shader_gen_py`` extension or a GL context. Run them with::

    python -m unittest discover -s tests
"""

import argparse
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sga.analysis import (  # noqa: E402
    build_initial_population,
    kmeans,
    select_seeds,
    spectral_signature,
)


def _image(seed, size: int = 32) -> np.ndarray:
    """A top-down ``(size, size, 4)`` uint8 image with reproducible random content."""
    return np.random.default_rng(seed).integers(0, 256, size=(size, size, 4), dtype=np.uint8)


class SpectralSignatureTests(unittest.TestCase):
    def test_length_matches_bins(self):
        image = _image(0)
        self.assertEqual(spectral_signature(image, bins=32).shape, (32,))
        self.assertEqual(spectral_signature(image, bins=8).shape, (8,))

    def test_bins_is_clamped_to_at_least_two(self):
        image = _image(1)
        self.assertEqual(spectral_signature(image, bins=1).shape, (2,))
        self.assertEqual(spectral_signature(image, bins=0).shape, (2,))

    def test_signature_is_unit_norm(self):
        signature = spectral_signature(_image(2))
        self.assertAlmostEqual(float(np.linalg.norm(signature)), 1.0, places=9)

    def test_constant_image_has_a_zero_signature(self):
        # A flat image carries no structure: the mean subtraction leaves nothing to bin.
        image = np.full((16, 16, 4), 137, dtype=np.uint8)
        np.testing.assert_array_equal(spectral_signature(image, bins=8), np.zeros(8))

    def test_ignores_brightness(self):
        # Leave headroom so the offset brightens every pixel without clipping, which would
        # change the image's structure rather than just its level.
        image = np.random.default_rng(3).integers(0, 200, size=(32, 32, 4), dtype=np.uint8)
        brighter = (image.astype(np.int64) + 40).astype(np.uint8)
        np.testing.assert_allclose(spectral_signature(image), spectral_signature(brighter), atol=1e-9)

    def test_ignores_rotation(self):
        # The bands are concentric radial slices of the magnitude spectrum, so a quarter turn
        # moves energy between pixels of the same band.
        image = _image(4)
        np.testing.assert_allclose(spectral_signature(image), spectral_signature(np.rot90(image)),
                                   atol=1e-6)


class KMeansTests(unittest.TestCase):
    def test_separates_well_spaced_clusters(self):
        rng = np.random.default_rng(0)
        features = np.vstack([rng.normal(0.0, 0.05, size=(10, 2)),
                              rng.normal(10.0, 0.05, size=(10, 2))])
        labels, centroids = kmeans(features, 2, random.Random(0))

        self.assertEqual(centroids.shape, (2, 2))
        self.assertEqual(len(set(labels[:10])), 1)
        self.assertEqual(len(set(labels[10:])), 1)
        self.assertNotEqual(labels[0], labels[10])

        ordered = sorted((float(c[0]) for c in centroids))
        self.assertAlmostEqual(ordered[0], 0.0, delta=1.0)
        self.assertAlmostEqual(ordered[1], 10.0, delta=1.0)

    def test_k_is_clamped_to_the_number_of_points(self):
        features = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        labels, centroids = kmeans(features, 5, random.Random(0))

        self.assertEqual(centroids.shape[0], 3)
        self.assertTrue(set(labels).issubset({0, 1, 2}))

    def test_is_deterministic_for_a_seed(self):
        features = np.random.default_rng(1).normal(size=(20, 4))
        labels_a, centroids_a = kmeans(features, 3, random.Random(7))
        labels_b, centroids_b = kmeans(features, 3, random.Random(7))

        np.testing.assert_array_equal(labels_a, labels_b)
        np.testing.assert_array_equal(centroids_a, centroids_b)


class SelectSeedsTests(unittest.TestCase):
    def test_keeps_the_closest_member_of_each_cluster(self):
        # Two far-apart signature groups; distances pick a clear winner inside each.
        signatures = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01],
                               [10.0, 10.0], [10.01, 10.0], [10.0, 10.01]])
        distances = [5.0, 1.0, 3.0, 9.0, 2.0, 7.0]
        pool = ["a0", "a1", "a2", "b0", "b1", "b2"]

        seeds = select_seeds(pool, signatures, distances, 2, random.Random(0))

        self.assertEqual(set(seeds), {"a1", "b1"})

    def test_returns_exactly_size_distinct_items(self):
        rng = np.random.default_rng(0)
        signatures = rng.normal(size=(12, 5))
        distances = list(rng.random(12))
        pool = list(range(12))

        for size in (1, 4, 12):
            seeds = select_seeds(pool, signatures, distances, size, random.Random(3))
            self.assertEqual(len(seeds), size)
            self.assertEqual(len(set(seeds)), size)

    def test_backfills_when_there_are_too_few_clusters(self):
        # Identical signatures collapse into a single cluster, so the remaining slots are
        # filled with the next-closest unused candidates.
        signatures = np.zeros((5, 3))
        distances = [4.0, 1.0, 3.0, 2.0, 0.5]
        pool = list("abcde")

        seeds = select_seeds(pool, signatures, distances, 3, random.Random(0))

        self.assertEqual(len(seeds), 3)
        self.assertEqual(len(set(seeds)), 3)
        self.assertIn("e", seeds)  # the globally closest candidate is always kept


class _FakeToolbox:
    """Stands in for the DEAP toolbox: ``individual`` labels each draw, ``population`` records
    the un-screened start that the blind fallback would return."""

    def __init__(self):
        self.count = 0

    def individual(self):
        self.count += 1
        return f"ind{self.count}"

    def population(self):
        return ["blind"]


class _FakeEvaluator:
    """``score_and_render`` as :mod:`sga.analysis` uses it: a target, and ``(fitness, image)``."""

    def __init__(self, target, image_for=None):
        self.target = target
        self.image_for = image_for or (lambda individual: _image(0))
        self.calls = 0

    def score_and_render(self, individual):
        self.calls += 1
        image = self.image_for(individual)
        if image is None:
            return 1e9, None
        return 0.0, image


def _args(population: int = 4, oversample: float = 3.0, bins: int = 16, seed: int = 0):
    return argparse.Namespace(population=population, spectral_oversample=oversample,
                              spectral_bins=bins, seed=seed)


class BuildInitialPopulationTests(unittest.TestCase):
    def test_returns_a_screened_population(self):
        toolbox, evaluator = _FakeToolbox(), _FakeEvaluator(_image(100))
        population = build_initial_population(toolbox, evaluator, _args(population=4))

        self.assertEqual(len(population), 4)
        self.assertTrue(all(item.startswith("ind") for item in population))
        self.assertNotIn("blind", population)  # enough renders, so no blind fallback

    def test_renders_an_oversampled_pool(self):
        toolbox, evaluator = _FakeToolbox(), _FakeEvaluator(_image(101))
        build_initial_population(toolbox, evaluator, _args(population=5, oversample=2.0))

        self.assertEqual(evaluator.calls, 10)  # max(population, round(oversample * population))

    def test_pool_is_never_smaller_than_the_population(self):
        toolbox, evaluator = _FakeToolbox(), _FakeEvaluator(_image(102))
        build_initial_population(toolbox, evaluator, _args(population=6, oversample=0.5))

        self.assertEqual(evaluator.calls, 6)

    def test_falls_back_to_a_blind_population_without_a_target(self):
        toolbox, evaluator = _FakeToolbox(), _FakeEvaluator(None)
        population = build_initial_population(toolbox, evaluator, _args())

        self.assertEqual(population, ["blind"])
        self.assertEqual(evaluator.calls, 0)

    def test_tops_up_when_too_few_candidates_render(self):
        toolbox = _FakeToolbox()
        evaluator = _FakeEvaluator(_image(103), image_for=lambda individual: None)
        population = build_initial_population(toolbox, evaluator, _args(population=3))

        self.assertEqual(len(population), 3)
        self.assertTrue(all(item.startswith("ind") for item in population))
        self.assertNotIn("blind", population)


if __name__ == "__main__":
    unittest.main()
