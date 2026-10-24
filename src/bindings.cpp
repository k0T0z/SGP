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
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ai-agent/utils/utils.hpp"
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

// `get_range_for_continuous_field` reports "this (node type, field) pair has no range" as
// `(0, 0)`, which no declared range uses, and returns the bounds as `std::variant<int, float>`
// because shader-gen mixes integer and floating point ranges.
double as_double(const ai_agent_utils::ContinuousRangeVariant& bound) {
  return std::visit([](const auto& value) { return static_cast<double>(value); }, bound);
}

py::object get_continuous_range(const int node_type, const int field_number) {
  const auto range = ai_agent_utils::get_range_for_continuous_field(node_type, field_number);
  const double lower = as_double(range.first);
  const double upper = as_double(range.second);
  if (lower == 0.0 && upper == 0.0) return py::none();
  return py::make_tuple(lower, upper);
}

std::vector<int> get_discrete_range(const int node_type, const int field_number) {
  return ai_agent_utils::get_range_for_discrete_field(node_type, field_number);
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

  // The mutation ranges live in shader-gen (`ai-agent/utils/utils.hpp`); re-exporting them keeps
  // the search ranges in step with the generator instead of duplicated on the Python side.
  m.def("get_continuous_range", &get_continuous_range, py::arg("node_type"), py::arg("field_number"),
        "shader-gen's mutation range for a continuous field, as (lower, upper).\n\n"
        "Returns None when shader-gen declares no range for the (node type, field number) pair.");
  m.def("get_discrete_range", &get_discrete_range, py::arg("node_type"), py::arg("field_number"),
        "shader-gen's mutation values for a discrete field.\n\n"
        "Returns an empty list when shader-gen declares no values for the (node type, field "
        "number) pair.");
}
