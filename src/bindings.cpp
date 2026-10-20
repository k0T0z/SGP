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
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "generator/raw_visual_shader_graph.hpp"
#include "generator/visual_shader_generator.hpp"

namespace py = pybind11;

namespace {

// `shader-gen`'s generate_shader reports its result through a `std::string&` out-parameter,
// which has no Python equivalent since strings are immutable. Return the status and the
// generated source together instead.
std::tuple<bool, std::string> generate_shader(const RawVisualShaderGraph& graph) {
  std::string code;
  const bool status = shadergen_visual_shader_generator::generate_shader(graph, code);
  return {status, code};
}

}  // namespace

PYBIND11_MODULE(shader_gen_py, m) {
  m.doc() = "Python bindings for shader-gen's Qt-free GLSL generator.";

  // The STL members are exposed read-only on purpose: pybind11 hands out copies of C++
  // containers to Python, so `graph.adj_matrix[7][0] = ...` would silently edit a throwaway
  // list. Build the graph in the constructor instead.
  py::class_<RawVisualShaderGraph>(m, "RawVisualShaderGraph")
      .def(py::init([](std::vector<std::string> headers, std::vector<std::vector<std::string>> adj_matrix) {
             RawVisualShaderGraph graph;
             graph.headers = std::move(headers);
             graph.adj_matrix = std::move(adj_matrix);
             return graph;
           }),
           py::arg("headers"), py::arg("adj_matrix"),
           "Per-node headers formatted as 'entity_type;node_id;oneof_field_number;param=value;...' "
           "plus adj_matrix[from][to] = '<from_port>,<to_port>' for every connection.")
      .def_readonly("headers", &RawVisualShaderGraph::headers)
      .def_readonly("adj_matrix", &RawVisualShaderGraph::adj_matrix);

  m.def("generate_shader", &generate_shader, py::arg("graph"),
        "Generate the full GLSL source for `graph`.\n\n"
        "Returns (status, glsl): `status` mirrors the C++ return value and `glsl` is the code "
        "buffer that the C++ API fills through its out-parameter.");

  m.def("generate_preview_shader", &shadergen_visual_shader_generator::generate_preview_shader, py::arg("graph"),
        py::arg("node_id"), py::arg("port"), "Generate GLSL that renders a single node's output port.");
}
