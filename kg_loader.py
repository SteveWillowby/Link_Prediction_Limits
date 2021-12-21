# Functions
#
# load_FB15k_237()
#   -- Returns (train, valid, test, id_to_mid, id_to_type)
#       where `train`, `valid`, and `test` are lists of (a, type, b) triples
#       `id_to_mid` is a dict mapping integer node labels to original labels
#       `id_to_type` is a dict mapping integer type labels to original labels
#
# convert_triples_to_NT_graph(triples, num_nodes=None)
#   -- Given a list of triples, constructs a graph ready to feed to Nauty/Traces
#       If `nodes` is None, then only nodes in the triples are included.
#   -- NOTE: ASSUMES that the nodes are indexed from 0 to num_nodes - 1.
#   -- Returns (neighbors_collections, node_colors) where neighbors_collections
#       is a list of collections of neighbors formatted as the
#       RAMFriendlyNTSession requires.
#       node_colors accounts for self-loops


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

def convert_triples_to_NT_graph(triples, num_nodes=None):
    get_nodes = num_nodes is None
    if get_nodes:
        num_nodes = 1
        for (a, _, b) in triples:
            assert type(a) is int
            assert type(b) is int
            num_nodes = max(num_nodes, max(a, b))
        num_nodes = num_nodes - 1
    else:
        for (a, _, b) in triples:
            assert type(a) is int
            assert type(b) is int

    neighbors_collections = [{} for _ in range(0, num_nodes)]
    node_colors = [[] for _ in range(0, num_nodes)]

    for (a, t, b) in triples:
        if a == b:
            node_colors[a].append(t)
            continue
        nc = neighbors_collections[a]
        if b not in nc:
            nc[b] = []
        nc[b].append(t)

    print("Number of nodes: %d" % num_nodes)
    num_edges = sum([len(d) for d in neighbors_collections])
    print("Number of flattened edges: %d" % num_edges)

    edge_types = set()
    for a in range(0, num_nodes):
        neighbors_collections[a] = \
            {b: tuple(sorted(list(l))) for b, l in \
                neighbors_collections[a].items()}
        for _, t in neighbors_collections[a].items():
            edge_types.add(t)
    edge_types = sorted(list(edge_types))
    edge_types = {edge_types[i]: i for i in range(0, len(edge_types))}
    for a in range(0, num_nodes):
        nc = neighbors_collections[a]
        neighbors = [(b, t) for b, t in nc.items()]
        for (b, t) in neighbors:
            nc[b] = edge_types[t]
    print("Number of distinct edge-type combos: %d" % len(edge_types))
    del edge_types

    for n in range(0, num_nodes):
        node_colors[n] = tuple(sorted(list(node_colors[n])))
    node_types = set(node_colors)
    node_types = sorted(list(node_types))
    node_types = {node_types[i]: i for i in range(0, len(node_types))}
    for n in range(0, num_nodes):
        node_colors[n] = node_types[node_colors[n]]
    print("Number of distinct self-loop-type combos: %d" % (len(node_types) - 1))
    del node_types

    return (neighbors_collections, node_colors)

if __name__ == "__main__":
    print("Loading FB15k-237...")
    (train, valid, test, id_to_mid, id_to_type) = load_FB15k_237()
    train_nodes = set([x[0] for x in train] + [x[2] for x in train])
    valid_nodes = set([x[0] for x in valid] + [x[2] for x in valid])
    test_nodes  = set([x[0] for x in test]  + [x[2] for x in test])

    print("Formatting FB15k-237...")
    (neighbors_collections, node_colors) = \
        convert_triples_to_NT_graph(train + valid, \
                num_nodes=len(train_nodes | valid_nodes | test_nodes))
