# SGP
I aim to solve the Shader Generation Problem using machine learning.

## Layout

- `externals/shader-gen` — git submodule. Only its Qt-free `generator/` sources (plus the
  protobuf schemas in `gui/model/schema/`) are compiled; SGP does not use shader-gen's
  CMakeLists, so no Qt and no GUI code is pulled in.
- `src/bindings.cpp` — the pybind11 module `shader_gen_py`.
- `main.py` — smoke test that generates the GLSL for the graph shader-gen uses in its own
  regression test (`tests/generator/test_vs_generator.cpp`).
- `CMakeLists.txt` — standalone build: generates the protos, compiles the 4 generator `.cpp`
  files, links the extension.

## Build

Requires MSVC (the Protobuf/Abseil binaries SGP links against are MSVC-built — the
CMakeLists fails fast if configured with another toolchain), CMake 3.21+, and
`pip install pybind11`.

The exact working invocation on this machine (Debug, dynamic Protobuf/Abseil):

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_PREFIX_PATH="C:/Users/saifs/Desktop/work/shader-gen-project/proto-lib-dynamic-debug;C:/Users/saifs/Desktop/work/shader-gen-project/absl-lib-dynamic-debug" `
      -Dpybind11_DIR="C:/Users/saifs/AppData/Local/Programs/Python/Python313/Lib/site-packages/pybind11/share/cmake/pybind11" `
      -DSGP_SHADER_GEN_ROOT="C:/Users/saifs/Desktop/work/shader-gen-project/shader-gen"
cmake --build build --config Debug
python main.py
```

`<protobuf-prefix>` holds `lib/cmake/protobuf` and `lib/cmake/utf8_range`; `<absl-prefix>`
holds `lib/cmake/absl` (`python -m pybind11 --cmakedir` prints the pybind11 dir).

The extension lands in `build/python/` next to the protobuf/abseil DLLs it needs, which
`main.py` registers via `os.add_dll_directory`.

### Why `SGP_SHADER_GEN_ROOT` is required

It defaults to `externals/shader-gen`, but the submodule is pinned to `25621f8`, which
predates shader-gen's Qt-free refactor — `generator/graph_string_utils.hpp`,
`proto_node.hpp`, `proto_node_factory.hpp` and `port_type_generator_factory.hpp` do not
exist there, so the default build fails. Point the flag at the shader-gen working tree
instead. The refactor sits on shader-gen's local `gen-python-api` branch as unpushed work;
once it is pushed and the submodule pointer is bumped, the flag can be dropped.

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

`main.py` currently prints a shader containing `in vec2 FragCoord;`, `uniform float uTime;`,
the value-noise helpers, and 7 node blocks in the order
`Input:4 → ValueNoise:5 → Input:1 → FloatFunc:2 → FloatOp:3 → FloatOp:6 → FloatFunc:7 →
Output:0` — the same output shader-gen's own generator test asserts.
