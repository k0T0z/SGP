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
  - Nodes:       ``0;<node_id>;<oneof_value_field_number>;<field_number>=<value>;...``
  - Connections: ``1;<connection_id>;<from_node_id>;<from_port>;<to_node_id>;<to_port>``
"""

from .parameters import format_value, gene_to_value, value_to_gene

# The fixed topology. Nodes and connections are copied verbatim on every evaluation; only the
# parameter values are replaced by the individual's genes.
SAMPLE_GRAPH = (
    "0;0;3,"
    "0;1;2;1=1,"
    "0;2;22;1=7;2=50,"
    "0;3;21;1=10,"
    "0;4;2;1=1,"
    "0;5;12;1=10,"
    "0;6;8;1=50;2=100;3=0;4=255,"
    "0;7;22;1=10;2=60,"
    "0;8;8;1=60;2=100;3=0;4=255,"
    "0;9;12;1=5,"
    "0;11;16;1=16,"
    "0;12;12;1=7,"
    "1;0;1;0;2;0,"
    "1;1;4;0;3;0,"
    "1;2;6;0;5;0,"
    "1;3;2;0;5;1,"
    "1;4;5;0;7;0,"
    "1;5;3;0;9;0,"
    "1;6;8;0;9;1,"
    "1;10;11;0;0;0,"
    "1;11;7;0;12;0,"
    "1;12;9;0;12;1,"
    "1;13;12;0;11;0"
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
        self.by_node = {}
        for spec in specs:
            self.by_node.setdefault(spec["node"], []).append(spec)

    def __call__(self, individual) -> str:
        nodes = [list(tokens) for tokens in self.nodes]
        for node_index, node_specs in self.by_node.items():
            tokens = nodes[node_index]
            for spec in node_specs:
                value = gene_to_value(spec, individual[spec["gene"]])
                tokens[3 + spec["offset"]] = f"{spec['field']}={format_value(spec, value)}"
        entities = [";".join(tokens) for tokens in nodes]
        entities += [";".join(tokens) for tokens in self.connections]
        return ",".join(entities)


def declared_genes(nodes, specs) -> list:
    """The gene vector that reproduces the parameter values in ``nodes`` verbatim.

    Used to seed the initial population around the graph's own values, so the search can
    refine a working point instead of rediscovering one from a uniformly random start.
    """
    genes = []
    for spec in specs:
        token = nodes[spec["node"]][3 + spec["offset"]]
        genes.append(value_to_gene(spec, float(token.split("=", 1)[1])))
    return genes
