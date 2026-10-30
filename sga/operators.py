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

from .parameters import gene_to_index


def mutate_parameters(individual, specs, indpb, sigma, reset_probability):
    """Perturb each selected gene, treating continuous and discrete genes differently.

    ``indpb`` is a per-gene probability and is applied to every child, matching the
    granularity shader-gen uses (``ai_agent/mutation.hpp`` draws once per parameter). A
    selected gene is then perturbed according to its kind:

    - continuous: nudged by Gaussian noise of scale ``sigma``, or re-drawn uniformly over
      [0, 1] (shader-gen's operator) with probability ``reset_probability``;
    - discrete: stepped by a single bucket, or re-drawn uniformly over its declared values
      with probability ``reset_probability``. The step is independent of ``sigma``, so a
      discrete gene keeps moving after the anneal has driven ``sigma`` below the bucket width.
    """
    for i, spec in enumerate(specs):
        if random.random() <= indpb:
            if spec["kind"] == "discrete":
                individual[i] = mutate_discrete(spec, individual[i], reset_probability)
            else:
                individual[i] = mutate_continuous(individual[i], sigma, reset_probability)
    return (individual,)


def mutate_continuous(unit, sigma, reset_probability):
    """A continuous gene: full-range re-draw, or a Gaussian nudge clamped to [0, 1]."""
    if random.random() < reset_probability:
        return random.random()
    return clamp_gene(unit + random.gauss(0.0, sigma))


def mutate_discrete(spec, unit, reset_probability):
    """A discrete gene: full-range re-draw over its values, or a single bucket step.

    The result is written at the chosen bucket's centre, ``(index + 0.5) / n``, so decoding it
    is unambiguous and a further step continues from a well-defined position.
    """
    n = len(spec["values"])
    index = gene_to_index(spec, unit)
    if random.random() < reset_probability:
        index = random.randrange(n)
    else:
        index = min(n - 1, max(0, index + random.choice((-1, 1))))
    return (index + 0.5) / n


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


def mate_kinds(ind1, ind2, specs, alpha):
    """Crossover that respects each gene's kind.

    ``cxBlend`` is applied to the whole individual and the continuous genes are read back from
    its result, so the blend stays DEAP's own formula; the discrete genes are instead set to a
    uniform exchange of the two parents' units. A discrete child therefore always inherits one
    parent's value and never lands in a bucket neither parent had.
    """
    parent1, parent2 = ind1[:], ind2[:]
    tools.cxBlend(ind1, ind2, alpha)
    for i, spec in enumerate(specs):
        if spec["kind"] == "discrete":
            if random.random() < 0.5:
                ind1[i], ind2[i] = parent2[i], parent1[i]
            else:
                ind1[i], ind2[i] = parent1[i], parent2[i]
    return ind1, ind2


def build_toolbox(evaluate, specs, args):
    """Register every operator the evolution loop needs.

    Genes are floats in [0, 1]; each is mapped onto its parameter's declared range by
    :func:`sga.parameters.gene_to_value`. Crossover and mutation branch on ``spec['kind']`` so
    discrete genes are inherited and perturbed as discrete values rather than blended.
    """
    n_genes = len(specs)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_unit", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_unit, n=n_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=args.population)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", mate_kinds, specs=specs, alpha=args.blend_alpha)

    # The mutation reads its sigma from the schedule rather than closing over a constant, so
    # the evolution loop can anneal it per generation; the schedule is kept on the toolbox for
    # the loop to advance.
    schedule = SigmaSchedule(args.mutation_sigma, args.mutation_sigma_final, args.generations)

    def mutate(individual):
        return mutate_parameters(individual, specs, args.mutation_probability, schedule.sigma,
                                 args.reset_probability)

    toolbox.register("mutate", mutate)
    toolbox.sigma_schedule = schedule
    toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)
    return toolbox


def clamp(individual):
    """Keep genes inside [0, 1] (blend crossover can overshoot the bounds)."""
    for i, gene in enumerate(individual):
        individual[i] = clamp_gene(gene)


def describe_spec(spec) -> str:
    """A one-line description of a gene's kind and range."""
    if spec["kind"] == "discrete":
        values = spec["values"]
        return f"discrete {values[0]}..{values[-1]} (n={len(values)})"
    return f"continuous [{spec['lo']:g}, {spec['hi']:g}]"


def report_operator_stats(specs, args, trials=20000, seed=0):
    """Measure, on random parents, what the real operators do to each gene.

    Everything is reported in *unit* space, the space the operators actually work in; a
    gene's declared range enters only through ``gene_to_index``, so the numbers describe the
    operators, not the decode. Three sections:

    - the gene table (index, node, field, kind, range);
    - crossover: how often the operator sends a child past both parents, and how often the raw
      result leaves ``[0, 1]`` (i.e. needs :func:`clamp`); for discrete genes also how often a
      child lands in a bucket neither parent had, and how often two same-bucket parents still
      produce a different bucket;
    - mutation: with every gene selected and ``reset_probability`` at 0. Continuous genes
      report the mean unit displacement, which is exactly ``sigma`` scaled by the parameter's
      range; discrete genes report ``P(bucket changes)``, which is independent of ``sigma`` by
      construction, so the two sigma blocks should agree on the discrete rows.

    Parents are drawn uniform on ``[0, 1]`` (a blind start), so the crossover numbers are the
    operator's raw behaviour rather than a snapshot of an evolved population.
    """
    random.seed(seed)
    n = len(specs)

    print("\nGene table:")
    for i, spec in enumerate(specs):
        print(f"  gene {i:>2}  node {spec['node']:>2}  field {spec['field']:>2}  {describe_spec(spec)}")

    overshoot = [0] * n
    clamped = [0] * n
    new_bucket = [0] * n
    same_parent_diff = [0] * n

    for _ in range(trials):
        x1 = [random.random() for _ in range(n)]
        x2 = [random.random() for _ in range(n)]
        c1, c2 = x1[:], x2[:]
        mate_kinds(c1, c2, specs, args.blend_alpha)
        for i, spec in enumerate(specs):
            lo, hi = min(x1[i], x2[i]), max(x1[i], x2[i])
            if c1[i] < lo or c1[i] > hi:
                overshoot[i] += 1
            if c1[i] < 0.0 or c1[i] > 1.0:
                clamped[i] += 1
            if spec["kind"] == "discrete":
                b1 = gene_to_index(spec, x1[i])
                b2 = gene_to_index(spec, x2[i])
                bc = gene_to_index(spec, c1[i])
                if bc not in (b1, b2):
                    new_bucket[i] += 1
                if b1 == b2 and bc != b1:
                    same_parent_diff[i] += 1

    print(f"\nCrossover probe (mate_kinds, alpha={args.blend_alpha}, {trials} random parent pairs):")
    for i, spec in enumerate(specs):
        print(f"  gene {i:>2}  {describe_spec(spec):<20}  "
              f"past-both-parents {100 * overshoot[i] / trials:5.1f}%  "
              f"left [0,1] {100 * clamped[i] / trials:5.1f}%  "
              f"new-bucket {100 * new_bucket[i] / trials:5.1f}%  "
              f"same-parents->different {100 * same_parent_diff[i] / trials:5.1f}%")

    print("\nMutation probe (every gene selected, reset=0; discrete rows are sigma-independent):")
    for sigma in (args.mutation_sigma, args.mutation_sigma_final):
        random.seed(seed)
        bucket_changed = [0] * n
        displacement = [0.0] * n
        for _ in range(trials):
            before = [random.random() for _ in range(n)]
            after = before[:]
            mutate_parameters(after, specs, 1.0, sigma, 0.0)
            for i, spec in enumerate(specs):
                if spec["kind"] == "discrete":
                    if gene_to_index(spec, after[i]) != gene_to_index(spec, before[i]):
                        bucket_changed[i] += 1
                else:
                    displacement[i] += abs(after[i] - before[i])
        print(f"  sigma {sigma:.0e}:")
        for i, spec in enumerate(specs):
            if spec["kind"] == "discrete":
                print(f"    gene {i:>2}  {describe_spec(spec):<20}  "
                      f"P(bucket changes) {100 * bucket_changed[i] / trials:5.1f}%")
            else:
                span = spec["hi"] - spec["lo"]
                print(f"    gene {i:>2}  {describe_spec(spec):<20}  "
                      f"mean |d unit| {displacement[i] / trials:.4f}  "
                      f"(= {displacement[i] / trials * span:.3f} of {span:g})")
