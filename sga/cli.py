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

Population size, generations, crossover and elitism default to shader-gen's ``Parameters``.
Mutation does not: shader-gen re-draws a parameter over its full range, which discards any
progress a candidate has made towards the target, so the default here is a Gaussian nudge,
annealed from ``--mutation-sigma`` down to ``--mutation-sigma-final``, and the reset is left
off (set ``--reset-probability 1.0`` for shader-gen's operator verbatim).
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

import renderer
from .evaluation import Evaluator
from .evolution import plot_fitness, run_ga, run_ga_steps
from .graph import SAMPLE_GRAPH, GraphBuilder, parse_graph
from .operators import build_toolbox
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

    parser.add_argument("--population", type=int, default=100, help="Population size (mirrors shader-gen).")
    parser.add_argument("--generations", type=int, default=500, help="Number of generations (mirrors shader-gen).")
    parser.add_argument("--crossover-probability", type=float, default=0.8, help="Crossover probability (mirrors shader-gen).")
    parser.add_argument("--mutation-probability", type=float, default=0.5,
                        help="Per-gene mutation probability, applied to every offspring.")
    parser.add_argument("--mutation-sigma", type=float, default=0.05,
                        help="Standard deviation of the local Gaussian gene perturbation at the "
                             "start of the run, as a fraction of each parameter's range. The "
                             "fitness slope is broad far from the target, so the run starts "
                             "coarse and anneals towards --mutation-sigma-final.")
    parser.add_argument("--mutation-sigma-final", type=float, default=1e-4,
                        help="Sigma the run anneals to geometrically by the last generation. The "
                             "usable slope around the correct values is only a few 1e-4 wide, so "
                             "the fine end is what lets a candidate settle. Equal to "
                             "--mutation-sigma holds the sigma fixed.")
    parser.add_argument("--reset-probability", type=float, default=0.0,
                        help="Chance a mutated gene is re-drawn over its full range (shader-gen's "
                             "operator) instead of nudged locally; 1.0 is shader-gen exactly. A "
                             "full-range re-draw always discards the fine-tuning, so it is off by "
                             "default.")
    parser.add_argument("--elitism", type=float, default=0.2, help="Elite fraction carried over untouched (mirrors shader-gen).")
    parser.add_argument("--tournament-size", type=int, default=3, help="Tournament selection size.")
    parser.add_argument("--blend-alpha", type=float, default=0.5, help="cxBlend alpha.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

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
    print(f"Target ink: {int((target_pixels[..., 3] > 0).sum())} of "
          f"{args.resolution * args.resolution} pixels at {args.resolution}x{args.resolution} "
          f"(an empty render scores 1.0, a pixel-identical render 0.0).")


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
    toolbox = build_toolbox(evaluator, len(specs), args)
    result = run_ga(toolbox, evaluator, args)

    best_graph, best_fitness = _save_artifacts(args, builder, scene, result)
    _report(args, best_graph, best_fitness)

    return 0


def _run_live(args, builder, specs) -> int:
    """Show the window and let it drive the loop, one generation per frame.

    The window owns the context the candidates are scored in, so the evaluator, the toolbox and
    the artifacts are all set up on the first frame, once that context exists.
    """
    state = {}

    def step(scene):
        """One generation; returns the GLSL of the current best, or ``None`` when finished."""
        if "steps" not in state:
            if scene.target_pixels is None:
                print(f"Could not read the target image {args.target!r}.", file=sys.stderr)
                state["failed"] = True
                return None
            _print_run_info(args, specs, scene.target_pixels)
            evaluator = Evaluator(builder, scene, scene.target_pixels)
            toolbox = build_toolbox(evaluator, len(specs), args)
            state["steps"] = run_ga_steps(toolbox, evaluator, args)

        try:
            _, _, best = next(state["steps"])
        except StopIteration as stop:
            state["artifacts"] = _save_artifacts(args, builder, scene, stop.value)
            return None

        status, glsl = shader_gen_py.generate_shader(builder(best))
        return glsl if status else None

    renderer.run_live(step, target_image_path=args.target, resolution=args.resolution)

    if state.get("failed"):
        return 1
    if "artifacts" not in state:
        print("Window closed before the run finished; nothing was written.", file=sys.stderr)
        return 1

    best_graph, best_fitness = state["artifacts"]
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

    if args.show_renderer:
        return _run_live(args, builder, specs)
    return _run_headless(args, builder, specs)
