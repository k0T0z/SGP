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

Matching the target means drawing the same ink, in the same places, in the same colour.
shader-gen scores a candidate by summing ``|dA| + |dR| + |dG| + |dB|`` over every pixel
(``ai-agent/fitness.hpp``), which cannot express that: on this target it makes an empty canvas
the *global optimum*. An empty render scores 1.0 on that metric, any candidate that paints
something paints it in the wrong place and scores worse, so the search has nowhere to go but
towards drawing nothing.

This instead rewards agreement, and is minimised:

    score   = (IoU of the two ink masks + colour agreement over the shared ink) / 2
    fitness = 1 - score

    0.0  the render is pixel-identical to the target
    1.0  nothing is drawn, or nothing lands on the target

Drawing nothing is the worst result rather than the best, so a blind random start has a slope
to climb: every pixel placed on the target's ink raises the score, and landing the target's
shapes in the wrong colour still beats not landing them at all.
"""

import numpy as np

from .runtime import shader_gen_py

# A graph that fails to generate or compile is assigned this. It is far worse than the worst
# real score, so invalid individuals always sort last.
PENALTY = 1e9


def ink_mask(pixels):
    """Where a top-down (H, W, 4) uint8 image draws anything at all."""
    return pixels[..., 3] > 0


def coverage_distance(image, target) -> float:
    """``1 - score`` for a render, given the target.

    ``image`` and ``target`` are top-down (H, W, 4) uint8 arrays. Half the score is the
    intersection over union of the two ink masks; the other half is how closely the colours
    agree on the ink the two images share.
    """
    image_mask = ink_mask(image)
    target_mask = ink_mask(target)
    union = int(np.count_nonzero(image_mask | target_mask))
    if union == 0:
        # Nothing drawn and nothing to draw: the render is as good as it can be.
        return 0.0

    shared = image_mask & target_mask
    intersection = int(np.count_nonzero(shared))
    iou = intersection / union
    if intersection == 0:
        # No shared ink, so there is no colour to compare.
        colour = 0.0
    else:
        difference = np.abs(image[..., :3][shared].astype(np.int32)
                            - target[..., :3][shared].astype(np.int32))
        colour = 1.0 - float(difference.mean()) / 255.0

    return 1.0 - 0.5 * (iou + colour)


class Evaluator:
    """Turns an individual into a ``(fitness,)`` tuple.

    The rebuilt graph string doubles as a cache key, so elites and repeated genomes do not pay
    for a second render.
    """

    def __init__(self, builder, scene, target):
        self.builder = builder
        self.scene = scene
        self.target = target
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
        return coverage_distance(image, self.target)
