# SGP
I aim to solve the Shader Generation Problem using machine learning.

## Layout

- `externals/shader-gen` — git submodule. Only its Qt-free `generator/` sources (plus the
  protobuf schemas in `gui/model/schema/`) are compiled; SGP does not use shader-gen's
  CMakeLists, so no Qt and no GUI code is pulled in.
- `src/bindings.cpp` — the pybind11 module `shader_gen_py`.
- `renderer.py` — owns all GL rendering. One `ShaderRenderer` runs either in the live
  window or offscreen, and in both cases hands back the pixels of the generated shader and of
  the target, which is what the genetic algorithm scores candidates with.
- `ga.py` — entry point for the genetic algorithm. The implementation lives in `sga/`:
  `parameters` (the ranges shader-gen declares, plus the gene mapping), `graph` (the encoded
  graph format and the fixed topology), `evaluation` (fitness), `operators` (DEAP toolbox),
  `evolution` (generational loop), `cli`, `runtime`.
- `CMakeLists.txt` — standalone build: generates the protos, compiles the 4 generator `.cpp`
  files, links the extension.

## Build

Requires MSVC (the Protobuf/Abseil binaries SGP links against are MSVC-built), CMake 3.21+,
and `pip install pybind11`.

The exact working invocation on this machine (Debug, dynamic Protobuf/Abseil):

```
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_PREFIX_PATH="C:/Users/saifs/Desktop/work/shader-gen-project/proto-lib-dynamic-debug;C:/Users/saifs/Desktop/work/shader-gen-project/absl-lib-dynamic-debug" `
      -Dpybind11_DIR="C:/Users/saifs/AppData/Local/Programs/Python/Python314/Lib/site-packages/pybind11/share/cmake/pybind11"
cmake --build build --config Debug
python ga.py
```

`<protobuf-prefix>` holds `lib/cmake/protobuf` and `lib/cmake/utf8_range`; `<absl-prefix>`
holds `lib/cmake/absl` (`python -m pybind11 --cmakedir` prints the pybind11 dir).

The extension lands in `build/python/` next to the protobuf/abseil DLLs it needs, which
`sga/runtime.py` registers via `os.add_dll_directory`.

### The shader-gen submodule

`SGP_SHADER_GEN_ROOT` defaults to `externals/shader-gen`, and that default is what the build
uses — no flag is needed. The submodule is pinned to shader-gen `master-qt6` (`abe102e`),
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
