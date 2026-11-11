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
rather than a copy of it that can drift. Scalar fields are addressed by the oneof value field
number of the ``VisualShaderNode`` message, exactly as in the encoded graph format. A repeated
``ColorGradientStop`` entry is encoded as ``<field_number>:<row>:<column>``; it is addressed by
the field number and column carried in the key itself.
"""

from .runtime import shader_gen_py

# Fields the GA must never mutate. Node type is the ``VisualShaderNode`` oneof value field number
# (visual_shader.proto); the field is the message's own field number. Colour alpha carries no signal
# for the reference image to optimise against, and the Input node's type only offers INPUT_TYPE_UV
# for now.
#
# Every node that resamples its input alongside a ``Matrix3x3`` ``transformation`` carries its nine
# entries as ``<field number>:<column>`` parameters, one field per entry. A transformation is a
# coordinate frame rather than a scalar, so the GA leaves the identity in place.
EXCLUDED_FIELDS = frozenset({
    (8, 4),    # ColorConstant.a
    (11, 4),   # Vec4Constant.w
    (42, 2),   # Blend.opacity
    (42, 3),   # Blend.inv_alpha
    (2, 1),    # Input.type
    (20, 2),   # ValueNoise.transformation
    (21, 10),  # PerlinNoise.transformation
    (44, 3),   # Gradient.transformation
    (45, 4),   # Checker.transformation
    (46, 10),  # Brick.transformation
    (47, 5),   # Tile.transformation
    (48, 7),   # Function.transformation
    (49, 11),  # Cellular.transformation
})

# A ColorGradientStop keeps alpha in its ``a`` channel (field 5), so every gradient node encodes it as
# the ``<field_number>:<row>:5`` column. shader-gen shares one 0..255 range across r/g/b/a, so the
# column is the only thing that tells alpha apart from the colour channels.
ALPHA_GRADIENT_COLUMN = 5


def get_range(node_type: int, field_number: int, sub_field: int = -1):
    """Return ``('discrete', [values])`` or ``('continuous', (lo, hi))`` for a node field.

    Discrete ranges take precedence over continuous ones, mirroring
    ``ai_agent_utils::get_range_for_field``. ``sub_field`` addresses a field inside a repeated
    message (a ``ColorGradientStop``), so a ``<field_number>:<row>:<column>`` key passes the column
    here. Returns ``None`` when shader-gen declares no range, in which case the field is not a gene
    and is carried through the graph unchanged.
    """
    values = shader_gen_py.get_discrete_range(node_type, field_number)
    if values:
        return "discrete", list(values)
    limits = shader_gen_py.get_continuous_range(node_type, field_number, sub_field)
    if limits is None:
        return None
    return "continuous", (limits[0], limits[1])


def build_gene_specs(nodes):
    """One gene per declared parameter, in node order.

    ``nodes`` is the node-token list from :func:`sga.graph.parse_graph`. A spec is identified by
    its node id and its encoded key and records the range it is drawn from; its gene index is its
    position in the returned list. A key is a scalar field number (``5``), a ``Matrix3x3`` entry's
    ``<field_number>:<column>`` pair (``10:5``) or a repeated-field ``<field_number>:<row>:<column>``
    triple (``10:0:1``).

    Parameters for which shader-gen declares no range are skipped and left at their encoded value,
    as are the fields in ``EXCLUDED_FIELDS`` and every gradient's alpha column.
    """
    specs = []
    for tokens in nodes:
        node_id = int(tokens[1])
        node_type = int(tokens[2])
        for parameter in tokens[3:]:
            key = parameter.split("=")[0]
            parts = key.split(":")
            if len(parts) == 3:
                field_number, _row, column = (int(part) for part in parts)
                if column == ALPHA_GRADIENT_COLUMN:
                    continue
                found = get_range(node_type, field_number, column)
            elif len(parts) == 2:
                # A ``Matrix3x3`` entry, keyed ``<field number>:<column>``. A transformation field
                # the exclusion list does not name still reaches ``get_range``, whose sub-field is
                # the column: shader-gen declares no range for those entries, so it would be skipped
                # there instead.
                field_number, column = (int(part) for part in parts)
                if (node_type, field_number) in EXCLUDED_FIELDS:
                    continue
                found = get_range(node_type, field_number, column)
            else:
                if (node_type, int(key)) in EXCLUDED_FIELDS:
                    continue
                found = get_range(node_type, int(key))
            if found is None:
                continue
            kind, limits = found
            spec = {"node": node_id, "key": key, "kind": kind}
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
