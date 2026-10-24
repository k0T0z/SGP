# SGP
I aim to solve the Shader Generation Problem using machine learning.

## Layout

- `externals/shader-gen` — git submodule. Only its Qt-free `generator/` sources (plus the
  protobuf schemas in `gui/model/schema/`) are compiled; SGP does not use shader-gen's
  CMakeLists, so no Qt and no GUI code is pulled in.
- `src/bindings.cpp` — the pybind11 module `shader_gen_py`.
- `main.py` — smoke test that generates the GLSL for the graph shader-gen uses in its own
  regression test (`tests/generator/test_vs_generator.cpp`).
- `renderer.py` — owns all GL rendering. One `ShaderRenderer` runs either in the preview
  window or offscreen, and in both cases hands back the pixels of the generated shader and of
  the target, which is what the genetic algorithm scores candidates with.
- `ga.py` — entry point for the genetic algorithm. The implementation lives in `sga/`:
  `parameters` (the ranges shader-gen declares, plus the gene mapping), `graph` (the encoded
  graph format and the fixed topology), `evaluation` (fitness), `operators` (DEAP toolbox),
  `evolution` (generational loop), `cli`, `runtime`.
- `CMakeLists.txt` — standalone build: generates the protos, compiles the 4 generator `.cpp`
  files, links the extension.

## Build

Requires MSVC (the Protobuf/Abseil binaries SGP links against are MSVC-built — the
CMakeLists fails fast if configured with another toolchain), CMake 3.21+, and
`pip install pybind11`.

The exact working invocation on this machine (Debug, dynamic Protobuf/Abseil):

```
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_PREFIX_PATH="C:/Users/saifs/Desktop/work/shader-gen-project/proto-lib-dynamic-debug;C:/Users/saifs/Desktop/work/shader-gen-project/absl-lib-dynamic-debug" `
      -Dpybind11_DIR="C:/Users/saifs/AppData/Local/Programs/Python/Python314/Lib/site-packages/pybind11/share/cmake/pybind11"
cmake --build build --config Debug
python main.py
```

`<protobuf-prefix>` holds `lib/cmake/protobuf` and `lib/cmake/utf8_range`; `<absl-prefix>`
holds `lib/cmake/absl` (`python -m pybind11 --cmakedir` prints the pybind11 dir).

The extension lands in `build/python/` next to the protobuf/abseil DLLs it needs, which
`main.py` registers via `os.add_dll_directory`.

### The shader-gen submodule

`SGP_SHADER_GEN_ROOT` defaults to `externals/shader-gen`, and that default is what the build
uses — no flag is needed. The submodule is pinned to shader-gen `master-qt6` (`0c86b8e`),
which carries the Qt-free refactor the sources require: `generator/graph_string_utils.hpp`,
`proto_node.hpp`, `proto_node_factory.hpp`, `port_type_generator_factory.hpp`, and the
Qt-free `ai-agent/utils/utils.hpp` that `get_continuous_range` / `get_discrete_range` are
re-exported from. The cache variable is still there if you want to build against a different
shader-gen checkout.

## API

```python
import shader_gen_py

graph = "0;0;3,0;1;2;1=2,..."  # encoded raw graph, see below
status, glsl = shader_gen_py.generate_shader(graph)
```

- `graph` — a comma separated list of entities where every entity is semicolon separated:
  - Nodes: `"0;<node_id>;<oneof_value_field_number>;<field_number>=<value>;..."`
  - Connections: `"1;<connection_id>;<from_node_id>;<from_port_index>;<to_node_id>;<to_port_index>"`
- `generate_shader` returns `(status, glsl)`: shader-gen reports its result through a
  `std::string&` out-parameter, which has no Python equivalent, so `status` mirrors the C++
  return value and `glsl` is the buffer.
- `generate_preview_shader(graph, node_id, port)` generates GLSL for a single output port.
- `get_continuous_range(node_type, field_number)` / `get_discrete_range(node_type, field_number)`
  re-export shader-gen's mutation ranges (`ai-agent/utils/utils.hpp`) so the search does not
  keep its own copy of them: a `(lower, upper)` pair or `None`, and a list of values or `[]`.

`main.py` currently prints a shader containing `in vec2 FragCoord;`, `uniform float uTime;`,
the value-noise helpers, and 7 node blocks in the order
`Input:4 → ValueNoise:5 → Input:1 → FloatFunc:2 → FloatOp:3 → FloatOp:6 → FloatFunc:7 →
Output:0` — the same output shader-gen's own generator test asserts.

## Genetic algorithm

    pip install -r requirements.txt
    python ga.py                                 # 100 individuals x 500 generations
    python ga.py --generations 50 --population 40 --no-preview

The topology is fixed and only parameters are searched, each over the range shader-gen
declares for its `(node type, field)` pair — asked for through
`shader_gen_py.get_continuous_range` / `get_discrete_range`, so the ranges live in shader-gen
and are not duplicated here. The one exception is the domain: shader-gen declares
ColorConstant as `(0, 1)`, but its generator emits `vec4(r, g, b, a) / 255.0`, so the fetched
range is scaled to `0..255` for the encoded graph (searching `0..1` renders nothing at all).

Fitness rewards agreement with the target rather than penalising difference, because the
difference metric (`ai-agent/fitness.hpp`) makes an empty canvas the global optimum:

    score   = (IoU of the two ink masks + colour agreement over the shared ink) / 2
    fitness = 1 - score

`0` is a pixel-identical render and `1` is drawing nothing, so drawing nothing is the worst
result and a blind start has a slope to climb.

Mutation nudges each selected gene with Gaussian noise and, with `--reset-probability`, falls
back to shader-gen's per-parameter reset. The reset is off by default because a full-range
re-draw throws away whatever the candidate had already matched.

The sigma is annealed geometrically across the run, from `--mutation-sigma` (`5e-2`) to
`--mutation-sigma-final` (`1e-4`), because the fitness has a usable slope only near the target
values: jittering the correct genes gives a median of `0.026` at sigma `1e-4`, `0.215` at
`1e-3` and `0.457` at `1e-2`, against `0.585` for a uniform random individual. A fixed sigma
therefore has to choose between crossing the space early and settling in that region late.
Equal start and end sigmas hold it fixed. The current sigma is printed per generation.

`--seed-fraction` starts that share of the first generation jittered around the values
the graph string already declares instead of uniformly at random; it is `0` by default, i.e. a
blind random start.

Population size, generations, crossover and elitism default to shader-gen's (`0.8` crossover,
`0.2` elitism) and are all CLI flags. Artifacts are written to `--out-dir` (the project root by
default): `best_graph.txt`, `best_shader.glsl`, `best_output.png`, `fitness.png`.
