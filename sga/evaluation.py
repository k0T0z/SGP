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

Matching the target means drawing the same channels, in the same places. A candidate is scored
by the mean squared error between its render and the target, taken over every pixel and all
four ``(H, W, 4)`` channels (alpha, red, green, blue), and is minimised:

    0.0  the render is pixel-identical to the target

Because every channel of every pixel contributes, the search has a slope everywhere rather
than only where ink happens to overlap.
"""

import numpy as np

from .runtime import shader_gen_py

# A graph that fails to generate or compile is assigned this. It is far worse than the worst
# real score, so invalid individuals always sort last.
PENALTY = 1e9

def mse(image, target) -> float:
    """Mean squared error over the ARGB channels of two top-down (H, W, 4) uint8 arrays."""
    difference = image.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(difference * difference))


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
        return mse(image, self.target)
