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

"""The fitness function.

Matching the target means drawing its ink in the same places. A candidate is scored by the
intersection over union (IoU) between the ink mask of its render and the ink mask of the
target, and is minimised:

    0.0  the render's ink covers exactly the target's ink
    1.0  the render's ink does not overlap the target's ink at all

The masks are taken from pixel brightness, so a uniform fill scores no better than it deserves:
unlike the mean squared error it replaces, painting the frame with the target's average colour
is not a way to score well, because the average colour is not ink.
"""

import numpy as np

from .runtime import shader_gen_py

# A graph that fails to generate or compile is assigned this. It is far worse than the worst
# real score, so invalid individuals always sort last.
PENALTY = 1e9

# A pixel counts as ink once its brightness clears this (of 255). The target is a hard
# silhouette, so a mid-range cut separates its ink from its background cleanly.
INK_THRESHOLD = 127


def ink_mask(image):
    """The binary silhouette of a top-down (H, W, 4) uint8 array: pixels brighter than the cut."""
    brightness = image[..., :3].mean(axis=2)
    return brightness > INK_THRESHOLD


def iou(a, b) -> float:
    """Intersection over union of two boolean masks. Two empty masks count as identical."""
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


class Evaluator:
    """Turns an individual into a ``(fitness,)`` tuple.

    The rebuilt graph string doubles as a cache key, so elites and repeated genomes do not pay
    for a second render.
    """

    def __init__(self, builder, scene, target):
        self.builder = builder
        self.scene = scene
        self.target_ink = ink_mask(target)
        self.cache = {}

    def __call__(self, individual):
        graph = self.builder(individual)
        fitness = self.cache.get(graph)
        if fitness is None:
            fitness = self._score(graph)
            self.cache[graph] = fitness
        return (fitness,)

    def _score(self, graph: str) -> float:
        status, glsl = shader_gen_py.generate_shader(graph)
        if not status:
            return PENALTY
        try:
            self.scene.update_shader(glsl)
            image = self.scene.render_pixels()
        except Exception:
            # The GA routinely produces parameter combinations whose shader fails to compile.
            return PENALTY
        return 1.0 - iou(ink_mask(image), self.target_ink)
