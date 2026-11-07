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

"""The encoded graph format and the fixed topology the GA works on.

An encoded graph is a comma separated list of entities, each semicolon separated:
  - Nodes:       ``0;<node_id>;<oneof_value_field_number>;<field>=<value>;...``
  - Connections: ``1;<connection_id>;<from_node_id>;<from_port>;<to_node_id>;<to_port>``

A node parameter's key is a single field number for scalar fields (``5=1``), or a
``<field_number>:<row>:<column>`` triple for a repeated-field entry such as a gradient stop
(``10:0:1=0`` means gradient field 10, row 0, channel 1). Both forms are rebuilt from genes; the
key is carried through the graph string verbatim.
"""

from .parameters import format_value, gene_to_value

# The fixed topology. Nodes and connections are copied verbatim on every evaluation; only the
# parameter values are replaced by the individual's genes.
SAMPLE_GRAPH = (
    "0;0;3,"
    "0;1;49;1=0;2=2;3=0;4=0;5=1;6=0;7=16;8=false;9=true;"
    "10:0:1=0;10:0:2=0;10:0:3=0;10:0:4=0;10:0:5=1;10:1:1=1;10:1:2=1;10:1:3=1;10:1:4=1;10:1:5=1,"
    "0;2;21;1=5;2=5;3=1;4=0.5;5=1;6=8;7=-1;8=true;"
    "9:0:1=0;9:0:2=0;9:0:3=0;9:0:4=0;9:0:5=1;9:1:1=1;9:1:2=1;9:1:3=1;9:1:4=1;9:1:5=1,"
    "0;3;2;1=1,"
    "0;4;42;1=7;2=1;3=false,"
    "0;5;46;1=4;2=6;3=0.5;4=0.10000000149011612;5=0.029999999329447746;"
    "6=0.029999999329447746;7=0.05000000074505806;8=true;"
    "9:0:1=0;9:0:2=0;9:0:3=0;9:0:4=0;9:0:5=1;9:1:1=0.5;9:1:2=0.5;9:1:3=0.5;9:1:4=0.5;9:1:5=1;"
    "9:2:1=1;9:2:2=0;9:2:3=0;9:2:4=1;9:2:5=1,"
    "0;6;42;1=7;2=1;3=false,"
    "1;0;3;0;1;0,"
    "1;1;3;0;2;0,"
    "1;2;1;0;4;0,"
    "1;3;2;0;4;1,"
    "1;4;3;0;5;0,"
    "1;5;4;0;6;0,"
    "1;6;5;0;6;1,"
    "1;7;6;0;0;0"
)


def parse_graph(graph: str):
    """Split an encoded graph into its node and connection token lists."""
    nodes, connections = [], []
    for entity in graph.split(","):
        tokens = entity.split(";")
        if tokens[0] == "0":
            nodes.append(tokens)
        elif tokens[0] == "1":
            connections.append(tokens)
        else:
            raise ValueError(f"Unknown entity type in graph: {entity!r}")
    return nodes, connections


class GraphBuilder:
    """Rebuilds an encoded graph string from an individual's genes."""

    def __init__(self, nodes, connections, specs):
        self.nodes = [list(tokens) for tokens in nodes]
        self.connections = [list(tokens) for tokens in connections]
        self.specs = specs
        self.node_by_id = {int(tokens[1]): index for index, tokens in enumerate(self.nodes)}
        # Keys are the raw parameter keys, so scalar (``5``) and repeated-field (``0:1``) entries
        # both resolve without being coerced to an int.
        self.offset_by_field = {}
        for node_id, index in self.node_by_id.items():
            for offset, parameter in enumerate(self.nodes[index][3:]):
                self.offset_by_field[(node_id, parameter.split("=")[0])] = offset

    def __call__(self, individual) -> str:
        nodes = [list(tokens) for tokens in self.nodes]
        for gene, spec in enumerate(self.specs):
            node_id, key = spec["node"], spec["key"]
            value = gene_to_value(spec, individual[gene])
            offset = self.offset_by_field[(node_id, key)]
            node_index = self.node_by_id[node_id]
            nodes[node_index][3 + offset] = f"{key}={format_value(spec, value)}"
        entities = [";".join(tokens) for tokens in nodes]
        entities += [";".join(tokens) for tokens in self.connections]
        return ",".join(entities)
