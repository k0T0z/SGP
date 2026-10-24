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

"""Parameter ranges and the gene <-> parameter mapping.

The ranges are not listed here: they are read from shader-gen's own
``ai_agent_utils::get_range_for_continuous_field`` / ``get_range_for_discrete_field``
(``ai-agent/utils/utils.hpp``) through the python extension, so the search follows shader-gen
rather than a copy of it that can drift. Fields are addressed by the oneof value field number
of the ``VisualShaderNode`` message, exactly as in the encoded graph format.
"""

from .runtime import shader_gen_py

# shader-gen declares ColorConstant as (0.0, 1.0), but its generator emits
# ``vec4(r, g, b, a) / 255.0`` (``generator/visual_shader_node_generators.cpp``) and so divides
# whatever the encoded graph carries: the encoded domain for a colour is 0..255, not 0..1. A
# render of the target with colour values drawn from 0..1 comes out completely empty (measured:
# 0 ink pixels), which would make the target unreachable. The declared range is therefore
# rescaled into the domain the string format uses.
ENCODED_DOMAIN_SCALE = {8: 255.0}


def get_range(node_type: int, field_number: int):
    """Return ``('discrete', [values])`` or ``('continuous', (lo, hi))`` for a node field.

    Discrete ranges take precedence over continuous ones, mirroring
    ``ai_agent_utils::get_range_for_field``.
    """
    values = shader_gen_py.get_discrete_range(node_type, field_number)
    if values:
        return "discrete", list(values)
    limits = shader_gen_py.get_continuous_range(node_type, field_number)
    if limits is None:
        raise ValueError(
            f"shader-gen declares no mutation range for node type {node_type}, "
            f"field {field_number}."
        )
    scale = ENCODED_DOMAIN_SCALE.get(node_type, 1.0)
    return "continuous", (limits[0] * scale, limits[1] * scale)


def build_gene_specs(nodes):
    """One gene per declared (node, field) pair, in node order.

    ``nodes`` is the node-token list from :func:`sga.graph.parse_graph`. Each spec records
    which token it rewrites (``node``/``offset``), the range it is drawn from, and the index
    of its gene in the individual.
    """
    specs = []
    for node_index, tokens in enumerate(nodes):
        node_type = int(tokens[2])
        for offset, parameter in enumerate(tokens[3:]):
            field_number = int(parameter.split("=")[0])
            kind, limits = get_range(node_type, field_number)
            spec = {"gene": len(specs), "node": node_index, "offset": offset,
                    "field": field_number, "kind": kind}
            if kind == "discrete":
                spec["values"] = limits
            else:
                spec["lo"], spec["hi"] = limits
            specs.append(spec)
    return specs


def gene_to_value(spec, unit: float):
    """Map a gene in [0, 1] onto the range the spec declares."""
    unit = min(1.0, max(0.0, unit))
    if spec["kind"] == "discrete":
        values = spec["values"]
        index = min(int(unit * len(values)), len(values) - 1)
        return values[index]
    return spec["lo"] + unit * (spec["hi"] - spec["lo"])


def value_to_gene(spec, value) -> float:
    """The inverse of :func:`gene_to_value`: map a declared value back onto [0, 1]."""
    if spec["kind"] == "discrete":
        values = spec["values"]
        index = min(range(len(values)), key=lambda i: abs(values[i] - float(value)))
        # gene_to_value uses int(unit * n), so the middle of each index's slot round-trips.
        return (index + 0.5) / len(values)
    lo, hi = spec["lo"], spec["hi"]
    if hi == lo:
        return 0.0
    return min(1.0, max(0.0, (float(value) - lo) / (hi - lo)))


def format_value(spec, value) -> str:
    """Render a value the way the encoded graph expects it (ints stay ints)."""
    if spec["kind"] == "discrete":
        return str(int(value))
    return f"{value:.6f}"
