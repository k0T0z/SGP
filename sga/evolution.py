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

"""The generational loop and the fitness plot."""

import random

import numpy as np
from deap import tools

from .operators import clamp


def run_ga(toolbox, evaluate, args, on_generation=None):
    """A tournament GA loop with optional elitism, run straight through to the end.

    Used when nothing is watching the run; see :func:`run_ga_steps` for the steppable version.
    """
    steps = run_ga_steps(toolbox, evaluate, args, on_generation)
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


def run_ga_steps(toolbox, evaluate, args, on_generation=None):
    """A tournament GA loop with optional elitism, one generation at a time.

    Yields ``(generation, record, best)`` after the initial population has been scored and after
    every generation, so a viewer can redraw in between without the loop blocking on it. The
    population, logbook and hall of fame reach the caller as the generator's return value, so
    :func:`run_ga` is just a drain of this.

    ``on_generation``, if given, is called with ``(generation, best)`` before each yield so a
    caller can persist the current best without stepping the loop itself.
    """
    random.seed(args.seed)
    population = toolbox.population()

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "sigma", "min", "avg"

    fitnesses = list(map(evaluate, population))
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness
    hof.update(population)
    if on_generation is not None:
        on_generation(0, hof[0])
    yield 0, stats.compile(population), hof[0]

    n_elites = int(args.elitism * args.population)
    for generation in range(1, args.generations + 1):
        sigma = toolbox.sigma_schedule.advance(generation)
        elites = [toolbox.clone(ind) for ind in tools.selBest(population, n_elites)]
        offspring = toolbox.select(population, len(population) - n_elites)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < args.crossover_probability:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Every offspring is mutated; the probability applies per gene, exactly as shader-gen
        # does (ai_agent_worker.cpp calls mutate() on both children unconditionally and
        # ai-agent/mutation.hpp draws once per parameter).
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        for child in offspring:
            clamp(child)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(evaluate, invalid))
        for individual, fitness in zip(invalid, fitnesses):
            individual.fitness.values = fitness

        population = elites + offspring
        hof.update(population)
        if on_generation is not None:
            on_generation(generation, hof[0])

        record = stats.compile(population)
        logbook.record(gen=generation, nevals=len(invalid), sigma=sigma, **record)
        # Only the new row: logbook.stream would re-print every earlier generation too.
        print(f"gen {generation:>4}/{args.generations}  nevals {len(invalid):>4}  sigma {sigma:.2e}  "
              f"min {record['min']:.6f}  avg {record['avg']:.6f}", flush=True)
        yield generation, record, hof[0]

    return population, logbook, hof


def plot_fitness(logbook, path):
    """Save a best/average fitness vs generation plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generations = logbook.select("gen")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(generations, logbook.select("min"), label="best")
    ax.plot(generations, logbook.select("avg"), label="average")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (1.0 = empty render, 0.0 = identical to target)")
    ax.set_title("SGP - DEAP genetic algorithm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
