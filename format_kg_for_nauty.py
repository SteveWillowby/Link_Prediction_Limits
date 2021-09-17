import networkx as nx

# Functions
#
# load_FB15k_237()
#   -- Returns (train, valid, test, id_to_mid, id_to_type)
#       where `train`, `valid`, and `test` are lists of (a, type, b) triples
#       `id_to_mid` is a dict mapping integer node labels to original labels
#       `id_to_type` is a dict mapping integer type labels to original labels
#
# convert_triples_to_NT_graph(triples, nodes=None)
#   -- Given a list of triples, constructs a graph ready to feed to Nauty/Traces
#       If `nodes` is None, then only nodes in the triples are included.
#   -- Returns (graph, bonuse_highlights) where `graph` is a networkx graph and
#       bonus `highlights` are any highlights needed to give types and
#       directions to the edges.


def load_FB15k_237():
    with open("FB15k-237/train.txt", "r") as f:
        train = f.readlines()
    with open("FB15k-237/valid.txt", "r") as f:
        valid = f.readlines()
    with open("FB15k-237/test.txt", "r") as f:
        test = f.readlines()

    train = [line.strip().split("\t") for line in train]
    valid = [line.strip().split("\t") for line in valid]
    test  = [line.strip().split("\t") for line in test]

    next_type_id = 0
    next_mid_id = 0
    type_to_id = {}
    mid_to_id = {}
    for l in [train, valid, test]:
        for triple in l:
            a = triple[0]
            b = triple[2]
            t = triple[1]
            if t not in type_to_id:
                type_to_id[t] = next_type_id
                next_type_id += 1
            if a not in mid_to_id:
                mid_to_id[a] = next_mid_id
                next_mid_id += 1
            if b not in mid_to_id:
                mid_to_id[b] = next_mid_id
                next_mid_id += 1

    train = [(mid_to_id[x[0]],type_to_id[x[1]],mid_to_id[x[2]]) for x in train]
    valid = [(mid_to_id[x[0]],type_to_id[x[1]],mid_to_id[x[2]]) for x in valid]
    test  = [(mid_to_id[x[0]],type_to_id[x[1]],mid_to_id[x[2]]) for x in test]

    return (train, valid, test, {i: mid for mid, i in mid_to_id.items()}, \
                                {i: t for t, i in type_to_id.items()})

def convert_triples_to_NT_graph(triples, nodes=None, nodes_for_direction=True):
    if nodes_for_direction:
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()

    edge_to_types = {}
    get_nodes = nodes is None
    if get_nodes:
        nodes = set()
    else:
        for node in nodes:
            assert type(node) is int

    for (a, t, b) in triples:
        if (a, b) not in edge_to_types:
            edge_to_types[(a, b)] = set()
        edge_to_types[(a, b)].add(t)
        if get_nodes:
            assert type(a) is int
            assert type(b) is int
            nodes.add(a)
            nodes.add(b)

    edge_to_types = {e: tuple(sorted(list(t))) for e,t in edge_to_types.items()}
    highlights = {node: 0 for node in nodes}
    type_combo_to_idx = \
        sorted(list(set([t for e, t in edge_to_types.items()])))
    type_combo_to_idx = \
        {type_combo_to_idx[i]: i for i in range(0, len(type_combo_to_idx))}
    edge_to_types = {e: type_combo_to_idx[t] for e, t in edge_to_types.items()}
    del type_combo_to_idx  # No longer using

    for node in nodes:
        graph.add_node(node)
    for (a, b), type_combo_idx in edge_to_types.items():
        if nodes_for_direction:
            ab1 = "%d-%d-1" % (a, b)
            ab2 = "%d-%d-2" % (a, b)
            graph.add_node(ab1)
            graph.add_node(ab2)
            graph.add_edge(a, b)
            graph.add_edge(a, ab1)
            graph.add_edge(ab1, ab2)
            graph.add_edge(ab2, b)
            highlights[ab1] = 1
            highlights[ab2] = type_combo_idx + 2
        else:
            ab = "%d-%d" % (a, b)
            graph.add_node(ab)
            graph.add_edge(a, ab)
            graph.add_edge(ab, b)
            highlights[ab] = type_combo_idx + 1

    return (graph, highlights)

if __name__ == "__main__":
    print("Loading FB15k-237...")
    (train, valid, test, id_to_mid, id_to_type) = load_FB15k_237()
    train_nodes = set([x[0] for x in train] + [x[2] for x in train])
    valid_nodes = set([x[0] for x in valid] + [x[2] for x in valid])
    test_nodes  = set([x[0] for x in test]  + [x[2] for x in test])

    print("Formatting FB15k-237...")
    (graph, highlights) = \
        convert_triples_to_NT_graph(train + valid, \
                nodes=(train_nodes | valid_nodes | test_nodes))
    print("Number of edge combo types: %d" % \
            (len(set([h for n, h in highlights.items()])) - 2))
