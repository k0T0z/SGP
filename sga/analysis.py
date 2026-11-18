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

"""Target-image analysis, used to seed the initial population.

A blind random start spends its first generations on candidates whose spatial frequency content
is nowhere near the target's. This module measures the target's Fourier signature once, screens
an oversampled random pool of candidates against it, and keeps the ones closest to the target,
spread out with K-means so the kept set is not one tight cluster of near-identical candidates.
It ranks by a signature that ignores orientation and brightness while the run itself is scored
by pixel MSE; that mismatch is deliberate, not an oversight (see :func:`spectral_signature`).

This is the SGP form of the heuristic:
"start with a handful of ... random nodes ... then just choose the ones that are the closest
based on that Fourier analysis", and "K-means clustering based on fourier values".
"""

import random

import numpy as np


def _luminance(image):
    """Rec. 601 luminance of a top-down ``(H, W, 4)`` uint8 image."""
    rgb = np.asarray(image, dtype=np.float64)[..., :3]
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def spectral_signature(image, bins: int = 32) -> np.ndarray:
    """A rotation-invariant Fourier signature of a top-down ``(H, W, 4)`` uint8 image.

    The mean is removed first, so the signature describes the image's *structure* rather than
    its brightness; the pixel MSE the GA minimises already accounts for level and colour. The
    luminance is windowed so the FFT does not read the image border as an edge, its magnitude
    spectrum is taken, and the spectrum is averaged into ``bins`` concentric radial bands from
    DC outward. Each band is an isotropic slice of the power spectrum, so the vector says how
    much energy the image carries at each spatial scale while ignoring where that energy sits
    or how it is turned. The result is L2-normalised so overall contrast does not dominate the
    comparison.

    This blindness to orientation and level is a deliberate, accepted mismatch with what the run
    actually optimises. The fitness is pixel MSE, which is *not* invariant: a candidate can carry
    the target's spectrum while sitting at the wrong rotation or exposure, and the screen will
    still rank it well. That is the point. The signature is only a cheap coarse filter whose job
    is to drop candidates whose spatial-frequency content is nowhere near the target's, before
    the GA pays for generations of blind search; orientation and level are left for the MSE to
    settle once the run starts. The screen answers "is this in the right neighbourhood?", not
    "is this the answer?".
    """
    bins = max(2, int(bins))
    luminance = _luminance(image)
    height, width = luminance.shape
    luminance = luminance - luminance.mean()
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(luminance * window)))

    rows, columns = np.indices((height, width))
    radius = np.hypot(rows - height // 2, columns - width // 2)
    edges = np.linspace(0.0, min(height, width) / 2.0, bins + 1)
    band_of_pixel = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)

    energy = np.bincount(band_of_pixel, weights=spectrum.ravel(), minlength=bins)[:bins]
    counts = np.bincount(band_of_pixel, minlength=bins)[:bins]
    bands = np.log1p(energy / np.maximum(counts, 1))

    norm = np.linalg.norm(bands)
    return bands / norm if norm > 0.0 else bands


def _seed_centroids(features: np.ndarray, k: int, rng: random.Random) -> np.ndarray:
    """K-means++ seeds: each new centre is drawn with probability proportional to its squared
    distance from the centres already chosen, so the initial centres start spread out."""
    n = features.shape[0]
    centroids = [features[rng.randrange(n)]]
    closest = np.linalg.norm(features - centroids[0], axis=1)
    for _ in range(1, k):
        total = float(closest.sum())
        index = rng.randrange(n) if total <= 0.0 else rng.choices(range(n), weights=(closest / total).tolist(), k=1)[0]
        centroids.append(features[index])
        closest = np.minimum(closest, np.linalg.norm(features - centroids[-1], axis=1))
    return np.asarray(centroids, dtype=np.float64)


def kmeans(features: np.ndarray, k: int, rng: random.Random, iterations: int = 25):
    """Lloyd's k-means over ``(n, dim)`` features, returning ``(labels, centroids)``."""
    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    k = max(1, min(k, n))
    centroids = _seed_centroids(features, k, rng)
    labels = np.full(n, -1, dtype=int)
    for _ in range(iterations):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(k):
            members = features[labels == index]
            if len(members):
                centroids[index] = members.mean(axis=0)
    return labels, centroids


def select_seeds(pool, signatures, distances, size: int, rng: random.Random):
    """Keep ``size`` candidates: the closest to the target within each K-means cluster.

    Clustering the signatures and taking one winner per cluster keeps the seeds near the target
    without collapsing them onto a single point of the search space; a plain "take the ``size``
    closest" would hand the GA a set of near-duplicates. Empty clusters are backfilled with the
    next-closest unused candidates.
    """
    labels, _ = kmeans(np.asarray(signatures, dtype=np.float64), size, rng)
    chosen, taken = [], set()
    for label in range(size):
        members = [index for index in range(len(pool)) if labels[index] == label]
        if not members:
            continue
        best = min(members, key=lambda index: distances[index])
        chosen.append(best)
        taken.add(best)
    for index in sorted(range(len(pool)), key=lambda index: distances[index]):
        if len(chosen) >= size:
            break
        if index not in taken:
            chosen.append(index)
            taken.add(index)
    return [pool[index] for index in chosen[:size]]


def build_initial_population(toolbox, evaluator, args):
    """The starting population, screened against the target's Fourier signature.

    Renders ``--spectral-oversample`` times ``--population`` random individuals, scores each
    render against the target's signature, and keeps the closest candidates one-per-K-means-cluster
    (see :func:`select_seeds`). Every render is cached by the evaluator, so the kept individuals
    are already scored when the run starts.

    Falls back to a plain random population when the target's pixels are unavailable or too few
    candidates render to fill ``--population``.
    """
    target = getattr(evaluator, "target", None)
    if target is None:
        return toolbox.population()

    target_signature = spectral_signature(target, args.spectral_bins)
    pool_size = max(args.population, int(round(args.spectral_oversample * args.population)))
    rng = random.Random(args.seed)

    pool, signatures, distances = [], [], []
    for _ in range(pool_size):
        individual = toolbox.individual()
        _, image = evaluator.score_and_render(individual)
        if image is None:
            continue
        signature = spectral_signature(image, args.spectral_bins)
        pool.append(individual)
        signatures.append(signature)
        distances.append(float(np.linalg.norm(signature - target_signature)))

    if len(pool) < args.population:
        while len(pool) < args.population:
            pool.append(toolbox.individual())
        return pool

    return select_seeds(pool, signatures, distances, args.population, rng)
