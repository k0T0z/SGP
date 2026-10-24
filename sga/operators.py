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

"""The DEAP toolbox: individuals, selection, crossover and mutation."""

import random

from deap import base, creator, tools


def mutate_parameters(individual, indpb, sigma, reset_probability):
    """Perturb each selected gene locally, occasionally re-drawing it over the full range.

    ``indpb`` is a per-gene probability and is applied to every child, matching the
    granularity shader-gen uses (``ai_agent/mutation.hpp`` draws once per parameter). Each
    selected gene is then either:

    - re-drawn uniformly over [0, 1], shader-gen's operator, which can move a gene anywhere in
      one step and so lets the search leave a converged basin; or
    - nudged by Gaussian noise of standard deviation ``sigma``, which refines a point that is
      already close without throwing away the progress around it.

    With ``reset_probability`` at 1.0 this is exactly shader-gen's mutation.
    """
    for i in range(len(individual)):
        if random.random() <= indpb:
            if random.random() < reset_probability:
                individual[i] = random.random()
            else:
                individual[i] = clamp_gene(individual[i] + random.gauss(0.0, sigma))
    return (individual,)


def clamp_gene(gene: float) -> float:
    """Keep a gene inside [0, 1]."""
    return min(1.0, max(0.0, gene))


class SigmaSchedule:
    """A coarse-to-fine anneal of the mutation sigma across the run.

    The fitness has a wide slope far away from the target and only a very narrow one near it
    (measured around the correct values: median 0.026 at sigma 1e-4, 0.215 at 1e-3, 0.457 at
    1e-2), so a single sigma has to choose between being able to cross the space at all and
    being able to settle inside that basin. Sweeping geometrically from a coarse sigma down to
    a fine one does both: the early generations explore, the late ones refine.

    Setting ``final`` equal to ``initial`` holds the sigma fixed.
    """

    def __init__(self, initial: float, final: float, generations: int):
        self.initial = initial
        self.final = final
        self.generations = max(1, generations)
        self.sigma = initial

    def advance(self, generation: int) -> float:
        """Move to the sigma scheduled for a 1-based ``generation`` and return it."""
        if self.generations == 1 or self.initial <= 0.0 or self.final <= 0.0:
            self.sigma = self.final
        else:
            t = (generation - 1) / (self.generations - 1)
            self.sigma = self.initial * (self.final / self.initial) ** t
        return self.sigma


def seed_population(toolbox, size, known_good, seed_fraction, seed_sigma):
    """Fill a population with individuals jittered around ``known_good`` plus random ones.

    ``known_good`` is the gene vector the graph string already declares; ``seed_fraction`` of
    the population is drawn as that vector plus Gaussian noise, and the rest is uniform random
    so the search still explores away from that point.
    """
    population = []
    if known_good:
        for _ in range(int(seed_fraction * size)):
            genes = [clamp_gene(g + random.gauss(0.0, seed_sigma)) for g in known_good]
            population.append(creator.Individual(genes))
    for _ in range(size - len(population)):
        population.append(toolbox.individual())
    random.shuffle(population)
    return population


def build_toolbox(evaluate, n_genes: int, args, known_good=None):
    """Register every operator the evolution loop needs.

    Genes are floats in [0, 1] and are mapped onto each parameter's declared range by
    :func:`sga.parameters.gene_to_value`.
    """
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_unit", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_unit, n=n_genes)

    def population(size=args.population):
        return seed_population(toolbox, size, known_good, args.seed_fraction, args.seed_sigma)

    toolbox.register("population", population)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=args.blend_alpha)

    # The mutation reads its sigma from the schedule rather than closing over a constant, so
    # the evolution loop can anneal it per generation; the schedule is kept on the toolbox for
    # the loop to advance.
    schedule = SigmaSchedule(args.mutation_sigma, args.mutation_sigma_final, args.generations)

    def mutate(individual):
        return mutate_parameters(individual, args.mutation_probability, schedule.sigma,
                                 args.reset_probability)

    toolbox.register("mutate", mutate)
    toolbox.sigma_schedule = schedule
    toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)
    return toolbox


def clamp(individual):
    """Keep genes inside [0, 1] (blend crossover can overshoot the bounds)."""
    for i, gene in enumerate(individual):
        individual[i] = clamp_gene(gene)
