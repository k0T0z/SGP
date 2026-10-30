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

"""Command line entry point.

Crossover and elitism default to shader-gen's ``Parameters``. Population, generations and the
mutation operator do not, because the search here runs over a fixed topology: 19 genes, three
of them driving chaotic noise fields, so the fitness surface is rugged and the basin around a
good answer is narrow. The defaults therefore spend more samples (``--population``,
``--generations``) and nudge far more gently than shader-gen's full-range re-draw: a Gaussian
perturbation annealed from ``--mutation-sigma`` down to ``--mutation-sigma-final``, applied at
shader-gen's per-gene ``--mutation-probability``, with a small ``--reset-probability`` leaving
shader-gen's operator reachable.
"""

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

# ``renderer`` lives at the project root, which is not necessarily on ``sys.path`` when the
# package is run as ``python -m sga.cli`` or from another directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import renderer  # noqa: E402
from .evaluation import Evaluator, ink_mask
from .evolution import plot_fitness, run_ga, run_ga_steps
from .graph import SAMPLE_GRAPH, GraphBuilder, parse_graph
from .operators import build_toolbox, report_operator_stats
from .parameters import build_gene_specs
from .runtime import shader_gen_py


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SGP - DEAP genetic algorithm over a fixed graph.")
    parser.add_argument("--graph", type=str, default=SAMPLE_GRAPH,
                        help="Fixed encoded graph topology (default: the graph from sga.graph).")
    parser.add_argument("--target", type=str, default="target.png", help="Target image to match.")
    parser.add_argument("--resolution", type=int, default=256, help="Render size (target is compared at this size).")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent.parent),
                        help="Directory for the saved artifacts.")

    parser.add_argument("--population", type=int, default=200,
                        help="Population size. Doubled over shader-gen's 100: the space is 19 "
                             "dimensional and rugged, so more samples per generation buy a better "
                             "chance of landing near a good basin.")
    parser.add_argument("--generations", type=int, default=1000,
                        help="Number of generations. Doubled over shader-gen's 500 to give the "
                             "sigma anneal room to walk from coarse exploration to the fine end.")
    parser.add_argument("--crossover-probability", type=float, default=0.8, help="Crossover probability (mirrors shader-gen).")
    parser.add_argument("--mutation-probability", type=float, default=0.1,
                        help="Per-gene mutation probability, applied to every offspring. "
                             "shader-gen's own value (ai-agent/parameters.cpp); the previous 0.5 "
                             "re-rolled roughly ten of the nineteen genes per child and threw "
                             "away most of what a candidate had already matched.")
                             
    parser.add_argument("--mutation-sigma", type=float, default=0.1,
                        help="Standard deviation of the local Gaussian gene perturbation at the "
                             "start of the run, as a fraction of each parameter's range. A blind "
                             "start needs steps large enough to cross a range such as the "
                             "0-360 noise offset before the anneal narrows in on a basin.")
    parser.add_argument("--mutation-sigma-final", type=float, default=1e-4,
                        help="Sigma the run anneals to geometrically by the last generation. The "
                             "usable slope around the correct values is only a few 1e-4 wide, so "
                             "the fine end is what lets a candidate settle. Equal to "
                             "--mutation-sigma holds the sigma fixed.")
    parser.add_argument("--reset-probability", type=float, default=0.05,
                        help="Chance a mutated gene is re-drawn over its full range (shader-gen's "
                             "operator) instead of nudged locally; 1.0 is shader-gen exactly. A "
                             "full-range re-draw discards the fine-tuning of the gene it hits, so "
                             "this stays low, but a small non-zero chance is what lets a gene the "
                             "search locked in early escape its basin.")
    parser.add_argument("--elitism", type=float, default=0.2, help="Elite fraction carried over untouched (mirrors shader-gen).")
    parser.add_argument("--tournament-size", type=int, default=3, help="Tournament selection size.")
    parser.add_argument("--blend-alpha", type=float, default=0.5, help="cxBlend alpha.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--stats", action="store_true", default=False,
                        help="Print the gene table and a probe of what crossover and mutation "
                             "do to each gene (bucket changes, overshoot, clamp rate) before "
                             "the run starts.")

    parser.add_argument("--show-renderer", dest="show_renderer", action="store_true", default=False,
                        help="Show the two-widget window while the algorithm runs: one generation "
                             "per frame, the current best beside the target. Off by default, "
                             "which keeps the run headless and unthrottled.")
    return parser.parse_args(argv)


def _print_run_info(args, specs, target_pixels):
    """The run's parameters, and what the target looks like at the size it is compared at."""
    print(f"Optimising {len(specs)} parameters: "
          f"{sum(s['kind'] == 'continuous' for s in specs)} continuous, "
          f"{sum(s['kind'] == 'discrete' for s in specs)} discrete.")
    print(f"Population: {args.population}, generations: {args.generations}, "
          f"target: {args.target} @ {args.resolution}x{args.resolution}.")
    print(f"Target ink: {int(ink_mask(target_pixels).sum())} of "
          f"{args.resolution * args.resolution} pixels at {args.resolution}x{args.resolution} "
          f"(an empty render scores 1.0, a matching silhouette 0.0).")


def _shaders_dir(args):
    """The gitignored folder the per-generation shaders are written to."""
    return Path(args.out_dir) / "shaders"


def _clear_generation_shaders(args):
    """Wipe the per-generation shader folder so every launch starts from a clean slate."""
    shaders_dir = _shaders_dir(args)
    if shaders_dir.is_dir():
        shutil.rmtree(shaders_dir)


def _save_generation_shader(args, generation, glsl):
    """Write one generation's best shader into the gitignored ``shaders`` folder."""
    shaders_dir = _shaders_dir(args)
    shaders_dir.mkdir(parents=True, exist_ok=True)
    (shaders_dir / f"gen_{generation:04d}.glsl").write_text(glsl, encoding="utf-8")


def _save_artifacts(args, builder, scene, result):
    """Write the best graph, shader, render and fitness plot.

    Returns ``(best_graph, best_fitness)`` so the caller can report the winner without
    regenerating anything.
    """
    _, logbook, hof = result
    best = hof[0]
    best_graph = builder(best)
    status, best_glsl = shader_gen_py.generate_shader(best_graph)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_graph.txt").write_text(best_graph + "\n", encoding="utf-8")
    (out_dir / "best_shader.glsl").write_text(best_glsl if status else "", encoding="utf-8")
    if status:
        scene.update_shader(best_glsl)
        Image.fromarray(scene.render_pixels()).save(out_dir / "best_output.png")
    plot_fitness(logbook, out_dir / "fitness.png")
    return best_graph, best.fitness.values[0]


def _report(args, best_graph, best_fitness):
    print(f"\nBest fitness: {best_fitness:.6f}")
    print(f"Best graph: {best_graph}")
    print(f"Artifacts written to {args.out_dir}")


def _run_headless(args, builder, specs) -> int:
    """Score candidates in an offscreen framebuffer with no window, as fast as they can go."""
    # One renderer, used offscreen: it owns the target's pixels and renders each candidate.
    scene = renderer.ShaderRenderer(ctx=renderer.create_context(),
                                    offscreen_size=(args.resolution, args.resolution))
    scene.load_target_image(args.target)
    if scene.target_pixels is None:
        print(f"Could not read the target image {args.target!r}.", file=sys.stderr)
        return 1

    _print_run_info(args, specs, scene.target_pixels)

    evaluator = Evaluator(builder, scene, scene.target_pixels)
    toolbox = build_toolbox(evaluator, specs, args)

    def write_generation(generation, best):
        status, glsl = shader_gen_py.generate_shader(builder(best))
        if status:
            _save_generation_shader(args, generation, glsl)

    result = run_ga(toolbox, evaluator, args, on_generation=write_generation)

    best_graph, best_fitness = _save_artifacts(args, builder, scene, result)
    _report(args, best_graph, best_fitness)

    return 0


class LiveDriver:
    """Steps the GA one generation at a time, handing the current best to the window to draw.

    The window owns the GL context candidates are scored in, so the evaluator and the toolbox are
    built on the first ``advance`` call, once that renderer exists. ``artifacts`` holds the run's
    ``(best_graph, best_fitness)`` once it finishes, and ``failed`` marks a run that stopped
    without a result: an unreadable target, or a winner that will not generate a shader.
    """

    def __init__(self, args, builder, specs):
        self.args = args
        self.builder = builder
        self.specs = specs
        self.steps = None
        self.artifacts = None
        self.failed = False
        self.done = False

    def advance(self, scene):
        """One generation; returns the GLSL of the current best, or ``None`` once finished."""
        if self.done:
            return None

        if self.steps is None:
            if scene.target_pixels is None:
                print(f"Could not read the target image {self.args.target!r}.", file=sys.stderr)
                self.failed = True
                self.done = True
                return None
            _print_run_info(self.args, self.specs, scene.target_pixels)
            evaluator = Evaluator(self.builder, scene, scene.target_pixels)
            self.steps = run_ga_steps(build_toolbox(evaluator, self.specs, self.args),
                                      evaluator, self.args)

        try:
            generation, _, best = next(self.steps)
        except StopIteration as stop:
            self.artifacts = _save_artifacts(self.args, self.builder, scene, stop.value)
            self.done = True
            return None

        status, glsl = shader_gen_py.generate_shader(self.builder(best))
        if not status:
            print("The run's best individual will not generate a shader; stopping.", file=sys.stderr)
            self.failed = True
            self.done = True
            return None
        _save_generation_shader(self.args, generation, glsl)
        return glsl


def _run_live(args, builder, specs) -> int:
    """Show the window and let it drive the loop, one generation per frame.

    The window owns the context the candidates are scored in, so the driver builds its evaluator
    and toolbox on the first frame, once that context exists.
    """
    driver = LiveDriver(args, builder, specs)
    renderer.run_live(driver, target_image_path=args.target, resolution=args.resolution)

    if driver.failed:
        return 1
    if driver.artifacts is None:
        print("Window closed before the run finished; nothing was written.", file=sys.stderr)
        return 1

    best_graph, best_fitness = driver.artifacts
    _report(args, best_graph, best_fitness)
    return 0


def main() -> int:
    args = parse_args()

    nodes, connections = parse_graph(args.graph)
    specs = build_gene_specs(nodes)
    if not specs:
        print("The graph exposes no parameters to optimise.", file=sys.stderr)
        return 1

    builder = GraphBuilder(nodes, connections, specs)
    _clear_generation_shaders(args)

    if args.stats:
        report_operator_stats(specs, args)

    if args.show_renderer:
        return _run_live(args, builder, specs)
    return _run_headless(args, builder, specs)
