
# Give the input graph, give an output version that respects the following:
#
#   1. Output is the same regardless of the ordering of triples.
#   2. Nodes are integers from 0 to n.
#   3. Edge types are integers from 0 to k.
#   4. Multiple (a, b) edges are squashed to a single edge with a joint type.
#
# The output is (out_edge_sets, in_edge_sets, edge_types) where:
#
#   A. out_edge_sets is a list mapping a node `n` to a set of nodes `n` points to
#   B. in_edge_sets is a list mapping a node `n` to a set of nodes pointing to `n`
#   C. edge_types is a dict mapping a (squashed) edge to its (squashed) type.
def format_kg_for_JH_special(triples, nodes=None):
    raw_edge_types = set()

    if nodes is None:
        raw_nodes = set()
        for (a, t, b) in triples:
            raw_edge_types.add(t)
            raw_nodes.add(a)
            raw_nodes.add(b)
    else:
        for (a, t, b) in triples:
            raw_edge_types.add(t)
        raw_nodes = nodes

    raw_et_to_int = sorted(list(raw_edge_types))
    raw_et_to_int = {raw_et_to_int[i]: i for i in range(0, len(raw_et_to_int))}
    raw_node_to_int = sorted(list(raw_nodes))
    raw_node_to_int = {raw_node_to_int[i]: i for i in range(0, len(raw_node_to_int))}

    triples = [(raw_node_to_int[a], raw_et_to_int[t], raw_node_to_int[b]) for (a, t, b) in triples]
    nodes = [i for i in range(0, len(raw_nodes))]

    out_edge_sets = [set() for n in nodes]
    in_edge_sets = [set() for n in nodes]

    edge_type_sets = {}
    for (a, t, b) in triples:
        out_edge_sets[a].add(b)
        in_edge_sets[b].add(a)
        if (a, b) not in edge_type_sets:
            edge_type_sets[(a, b)] = set()
        edge_type_sets[(a, b)].add(t)

    edge_type_sets = {edge: tuple(sorted(list(s))) for edge, s in edge_type_sets.items()}
    edge_type_set_to_int = sorted(list(set([s for edge, s in edge_type_sets.items()])))
    edge_type_set_to_int = {edge_type_set_to_int[i]: i for i in range(0, len(edge_type_set_to_int))}
    edge_type_sets = {edge: edge_type_set_to_int[s] for edge, s in edge_type_sets.items()}

    return (out_edge_sets, in_edge_sets, edge_type_sets)  # edge_type_sets is now edge_types
