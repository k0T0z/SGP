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
    return "continuous", (limits[0], limits[1])


def build_gene_specs(nodes):
    """One gene per declared (node id, field) pair, in node order.

    ``nodes`` is the node-token list from :func:`sga.graph.parse_graph`. A spec is identified by
    its node id and field number and records the range it is drawn from; its gene index is its
    position in the returned list.
    """
    specs = []
    for tokens in nodes:
        node_id = int(tokens[1])
        node_type = int(tokens[2])
        for parameter in tokens[3:]:
            field_number = int(parameter.split("=")[0])
            kind, limits = get_range(node_type, field_number)
            spec = {"node": node_id, "field": field_number, "kind": kind}
            if kind == "discrete":
                spec["values"] = limits
            else:
                spec["lo"], spec["hi"] = limits
            specs.append(spec)
    return specs


def gene_to_index(spec, unit: float):
    """The bucket a discrete gene selects, or ``None`` for a continuous gene.

    The single source of truth for the discrete mapping: :func:`gene_to_value` decodes through
    it, so the two can never disagree.
    """
    if spec["kind"] != "discrete":
        return None
    unit = min(1.0, max(0.0, unit))
    return min(int(unit * len(spec["values"])), len(spec["values"]) - 1)


def gene_to_value(spec, unit: float):
    """Map a gene in [0, 1] onto the range the spec declares."""
    if spec["kind"] == "discrete":
        return spec["values"][gene_to_index(spec, unit)]
    unit = min(1.0, max(0.0, unit))
    return spec["lo"] + unit * (spec["hi"] - spec["lo"])


def format_value(spec, value) -> str:
    """Render a value the way the encoded graph expects it (ints stay ints)."""
    if spec["kind"] == "discrete":
        return str(int(value))
    return f"{value:.6f}"
