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
#include "generator/renderer/shader_renderer.hpp"
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

py::object get_continuous_range(const int node_type, const int field_number, const int sub_field) {
  const auto range = ai_agent_utils::get_range_for_continuous_field(node_type, field_number, sub_field);
  const double lower = as_double(range.first);
  const double upper = as_double(range.second);
  if (lower == 0.0 && upper == 0.0) return py::none();
  return py::make_tuple(lower, upper);
}

std::vector<int> get_discrete_range(const int node_type, const int field_number) {
  return ai_agent_utils::get_range_for_discrete_field(node_type, field_number);
}

// `shader-gen` hands the quad back as a reference to its own static array; pybind11 casts a
// `std::vector<float>` to a list on its own, so copy it out.
std::vector<float> quad_vertices() {
  const auto& vertices = shadergen_renderer::ShaderRenderer::quad_vertices();
  return {vertices.begin(), vertices.end()};
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
        py::arg("sub_field") = -1,
        "shader-gen's mutation range for a continuous field, as (lower, upper).\n\n"
        "`sub_field` identifies a field inside a repeated message (a `ColorGradientStop`), so a "
        "gradient stop is addressed by its node's gradient field number plus the stop's column.\n\n"
        "Returns None when shader-gen declares no range for the (node type, field number) pair.");
  m.def("get_discrete_range", &get_discrete_range, py::arg("node_type"), py::arg("field_number"),
        "shader-gen's mutation values for a discrete field.\n\n"
        "Returns an empty list when shader-gen declares no values for the (node type, field "
        "number) pair.");

  // The renderer contract: the GLSL version, the interface names, the full-screen quad and the
  // shader sources built from them. `renderer.py` reads every one of these from here instead of
  // repeating the values, so it cannot drift from the C++ renderers that draw the same code.
  py::module_ renderer = m.def_submodule(
      "renderer",
      "The shader-interface contract the generated GLSL is written against, mirroring shader-gen's "
      "`shadergen_renderer::ShaderRenderer`.");

  renderer.attr("GLSL_VERSION") = py::str(std::string(shadergen_renderer::ShaderRenderer::kGlslVersion));
  renderer.attr("FRAG_COORD_NAME") = py::str(std::string(shadergen_renderer::ShaderRenderer::kFragCoordName));
  renderer.attr("FRAG_COLOR_NAME") = py::str(std::string(shadergen_renderer::ShaderRenderer::kFragColorName));
  renderer.attr("TIME_UNIFORM_NAME") = py::str(std::string(shadergen_renderer::ShaderRenderer::kTimeUniformName));
  renderer.attr("POSITION_ATTRIBUTE_NAME") =
      py::str(std::string(shadergen_renderer::ShaderRenderer::kPositionAttributeName));
  renderer.attr("FRAG_COORD_ATTRIBUTE_NAME") =
      py::str(std::string(shadergen_renderer::ShaderRenderer::kFragCoordAttributeName));
  renderer.attr("DEFAULT_RESOLUTION") = py::int_(shadergen_renderer::ShaderRenderer::kDefaultResolution);
  renderer.attr("QUAD_VERTEX_COUNT") = py::int_(shadergen_renderer::ShaderRenderer::kQuadVertexCount);

  renderer.def("quad_vertices", &quad_vertices,
               "The full-screen quad as interleaved (x, y, u, v), drawn as a triangle strip.");
  renderer.def(
      "vertex_shader_source", []() { return shadergen_renderer::ShaderRenderer::vertex_shader_source(); },
      "The vertex shader forwarding the quad's texture coordinates to the fragment input.");
  renderer.def(
      "fragment_shader_source",
      [](const std::string& code) { return shadergen_renderer::ShaderRenderer::fragment_shader_source(code); },
      py::arg("code"), "`code` prefixed with the version directive the renderer expects.");
  renderer.def(
      "default_fragment_shader_source",
      []() { return shadergen_renderer::ShaderRenderer::default_fragment_shader_source(); },
      "The fragment shader drawn while there is no code yet.");
  renderer.def(
      "error_fragment_shader_source",
      []() { return shadergen_renderer::ShaderRenderer::error_fragment_shader_source(); },
      "The fragment shader drawn when the generated code fails to compile.");
}
