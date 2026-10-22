/*********************************************************************************/
/*                                                                               */
/*  Copyright (C) 2026 Seif Kandil (k0T0z)                                       */
/*                                                                               */
/*  This file is a part of the ENIGMA Development Environment.                   */
/*                                                                               */
/*                                                                               */
/*  ENIGMA is free software: you can redistribute it and/or modify it under the  */
/*  terms of the GNU General Public License as published by the Free Software    */
/*  Foundation, version 3 of the license or any later version.                   */
/*                                                                               */
/*  This application and its source code is distributed AS-IS, WITHOUT ANY       */
/*  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS    */
/*  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more        */
/*  details.                                                                     */
/*                                                                               */
/*  You should have recieved a copy of the GNU General Public License along      */
/*  with this code. If not, see <http://www.gnu.org/licenses/>                   */
/*                                                                               */
/*  ENIGMA is an environment designed to create games and other programs with a  */
/*  high-level, fully compilable language. Developers of ENIGMA or anything      */
/*  associated with ENIGMA are in no way responsible for its users or            */
/*  applications created by its users, or damages caused by the environment      */
/*  or programs made in the environment.                                         */
/*                                                                               */
/*********************************************************************************/

#include <string>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "generator/visual_shader_generator.hpp"

namespace py = pybind11;

namespace {

// `shader-gen`'s generate_shader reports its result through a `std::string&` out-parameter,
// which has no Python equivalent since strings are immutable. Return the status and the
// generated source together instead.
std::tuple<bool, std::string> generate_shader(const std::string& graph) {
  std::string code;
  const bool status = shadergen_visual_shader_generator::generate_shader(graph, code);
  return {status, code};
}

}  // namespace

PYBIND11_MODULE(shader_gen_py, m) {
  m.doc() = "Python bindings for shader-gen's Qt-free GLSL generator.";

  // `graph` is shader-gen's encoded raw graph: a comma separated list of entities where every
  // entity is semicolon separated.
  //   - Nodes:       "0;<node_id>;<oneof_value_field_number>;<field_number>=<value>;..."
  //   - Connections: "1;<connection_id>;<from_node_id>;<from_port_index>;<to_node_id>;<to_port_index>"
  m.def("generate_shader", &generate_shader, py::arg("graph"),
        "Generate the full GLSL source for `graph`.\n\n"
        "Returns (status, glsl): `status` mirrors the C++ return value and `glsl` is the code "
        "buffer that the C++ API fills through its out-parameter.");

  m.def("generate_preview_shader", &shadergen_visual_shader_generator::generate_preview_shader, py::arg("graph"),
        py::arg("node_id"), py::arg("port"), "Generate GLSL that renders a single node's output port.");
}
