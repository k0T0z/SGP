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

"""SGP - the Shader Generation Problem.

Minimal driver for the `shader_gen_py` extension module, which wraps shader-gen's Qt-free
`generator` library. Build the extension first:

    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
          -DCMAKE_PREFIX_PATH="<protobuf-prefix>;<absl-prefix>" ^
          -Dpybind11_DIR="<pybind11-cmake-dir>" ^
          -DSGP_SHADER_GEN_ROOT="<path-to-shader-gen>"
    cmake --build build --config Debug

then run:

    python main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent / "build" / "python"
if _MODULE_DIR.is_dir():
    # The extension ships its protobuf/abseil DLLs next to itself; register the directory so
    # those dependencies resolve on Windows.
    os.add_dll_directory(str(_MODULE_DIR))
    sys.path.insert(0, str(_MODULE_DIR))

import shader_gen_py  # noqa: E402
import renderer
import argparse

def build_sample_graph() -> shader_gen_py.RawVisualShaderGraph:
    """The graph shader-gen uses in its own generator regression test.

    A header is 'entity_type;node_id;oneof_field_number;param=value;...' with entity_type 0
    meaning a node. The adjacency matrix cell [from][to] is '<from_port>,<to_port>'.
    """
    headers = [
        "0;0;3",                # Output          (oneof 3)
        "0;1;2;1=2",            # Input TIME      (oneof 2,  type 2)
        "0;2;16;1=1",           # FloatFunc SIN   (oneof 16, func 1)
        "0;3;12;1=4",           # FloatOp DIV     (oneof 12, op 4)
        "0;4;2;1=1",            # Input UV        (oneof 2,  type 1)
        "0;5;20;1=100.000000",  # ValueNoise      (oneof 20, scale 100)
        "0;6;12;1=2",           # FloatOp SUB     (oneof 12, op 2)
        "0;7;16;1=16",          # FloatFunc ROUND (oneof 16, func 16)
    ]

    node_count = len(headers)
    adj_matrix = [["" for _ in range(node_count)] for _ in range(node_count)]

    adj_matrix[1][2] = "0,0"  # time        -> sin
    adj_matrix[2][3] = "0,0"  # sin         -> div
    adj_matrix[3][6] = "0,1"  # div         -> sub (input 1)
    adj_matrix[4][5] = "0,0"  # uv          -> value_noise
    adj_matrix[5][6] = "0,0"  # value_noise -> sub (input 0)
    adj_matrix[6][7] = "0,0"  # sub         -> round
    adj_matrix[7][0] = "0,0"  # round       -> output

    return shader_gen_py.RawVisualShaderGraph(headers, adj_matrix)


def main() -> int:
    parser = argparse.ArgumentParser(description="SGP - Shader Generation Problem")
    parser.add_argument("--preview", action="store_true", help="Run the shader preview")
    parser.add_argument("--target", type=str, help="Path to the target image for comparison")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    graph = build_sample_graph()

    status, glsl = shader_gen_py.generate_shader(graph)
    if not status:
        print("generate_shader failed", file=sys.stderr)
        return 1

    if args.preview:
        print("Starting preview...")
        renderer.run_preview(glsl, target_image_path=args.target, headless=args.headless)
    else:
        print(glsl)
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
