
# Read the tuples.

nodes = set()
edge_types = set()
train = []
valid = []
test  = []
train_f = open("train.txt", "r")
for line in train_f.readlines():
    line = line.strip().split("\t")
    if len(line) != 3:
        print(line)
    assert len(line) == 3
    nodes.add(line[0])
    nodes.add(line[2])
    edge_types.add(line[1])
    train.append(tuple(line))
train_f.close()
valid_f = open("valid.txt", "r")
for line in valid_f.readlines():
    line = line.strip().split("\t")
    if len(line) != 3:
        print(line)
    assert len(line) == 3
    nodes.add(line[0])
    nodes.add(line[2])
    edge_types.add(line[1])
    valid.append(tuple(line))
valid_f.close()
test_f = open("test.txt", "r")
for line in test_f.readlines():
    line = line.strip().split("\t")
    if len(line) != 3:
        print(line)
    assert len(line) == 3
    nodes.add(line[0])
    nodes.add(line[2])
    edge_types.add(line[1])
    test.append(tuple(line))
test_f.close()

# Relabel the nodes and edge_types to be integers starting at 0.
#   Perform the renaming in lexicographical order.
nodes = sorted(list(nodes))
nodes = {nodes[i]: i for i in range(0, len(nodes))}
edge_types = sorted(list(edge_types))
edge_types = {edge_types[i]: i for i in range(0, len(edge_types))}

train = [(nodes[a], nodes[b], edge_types[t]) for (a, t, b) in train]
valid = [(nodes[a], nodes[b], edge_types[t]) for (a, t, b) in valid]
test  = [(nodes[a], nodes[b], edge_types[t]) for (a, t, b) in test]

# Write out all the files.
nodes_file = open("FB15k-237_nodes.txt", "w")
for i in range(0, len(nodes)):
    if i > 0:
        nodes_file.write("\n")
    nodes_file.write("%d" % i)
nodes_file.close()

node_relabel_file = open("FB15k-237_node_relabeling.txt", "w")
nodes = {i: n for n, i in nodes.items()}
first = True
for i in range(0, len(nodes)):
    n = nodes[i]
    if not first:
        node_relabel_file.write("\n")
    first = False
    node_relabel_file.write("%d %s" % (i, n))
node_relabel_file.close()

edge_types_relabel_file = open("FB15k-237_triple_type_relabeling.txt", "w")
edge_types = {i: t for t, i in edge_types.items()}
first = True
for i in range(0, len(edge_types)):
    t = edge_types[i]
    if not first:
        edge_types_relabel_file.write("\n")
    first = False
    edge_types_relabel_file.write("%d %s" % (i, t))
edge_types_relabel_file.close()

train_file = open("FB15k-237_train_edges.txt", "w")
first = True
for (a, b, t) in train:
    if not first:
        train_file.write("\n")
    first = False
    train_file.write("%d %d %d" % (a, b, t))
train_file.close()

valid_file = open("FB15k-237_valid_edges.txt", "w")
first = True
for (a, b, t) in valid:
    if not first:
        valid_file.write("\n")
    first = False
    valid_file.write("%d %d %d" % (a, b, t))
valid_file.close()

test_file = open("FB15k-237_test_edges.txt", "w")
first = True
for (a, b, t) in test:
    if not first:
        test_file.write("\n")
    first = False
    test_file.write("%d %d %d" % (a, b, t))
test_file.close()

train_and_valid_file = open("FB15k-237_train_and_valid_edges.txt", "w")
first = True
for l in [train, valid]:
    for (a, b, t) in l:
        if not first:
            train_and_valid_file.write("\n")
        first = False
        train_and_valid_file.write("%d %d %d" % (a, b, t))
train_and_valid_file.close()

all_file = open("FB15k-237_all_edges.txt", "w")
first = True
for l in [train, valid, test]:
    for (a, b, t) in l:
        if not first:
            all_file.write("\n")
        first = False
        all_file.write("%d %d %d" % (a, b, t))
all_file.close()

"""
# Since edges can be repeated, squash the edge types as well so that each edge
#   gets a single label which corresponds to the collection of labels it had.
#
# Edge type set S --> tuple(sorted(list(S)))
#
# This squashing follows a lexicographical order on the sorted tuples.
#
# E.g.
#  (0, 10, 11, 12) < (1, 5, 7) < (2, 3) < (3, 4) < (4, 8) < (4, 10) < (5,)
#
# Edges in train/valid/test are considered distinct edges.

TRAIN = 0
VALID = 1
TEST = 2

type_squashing = {}
for l, marker in [(train, TRAIN), (valid, VALID), (test, TEST)]:
    for (a, b, t) in l:
        if (a, b, marker) not in type_squashing:
            type_squashing[(a, b, marker)] = []
        type_squashing[(a, b, marker)].append((t, marker))

type_squashing = {e: tuple(sorted(l)) for e, l in type_squashing.items()}
type_combos = set()
for _, type_tuple in type_squashing:
    type_combos.add(type_tuple)

type_combos = sorted(list(type_combos))
type_combos = {type_combos[i]: i for i in range(0, len(type_combos))}
type_squashing = {e: type_combos[t] for e, t in type_squashing.items()}

train = []
valid = []
test  = []
for (a, b, marker), t in type_squashing.items():
    if marker == TRAIN:
        train.append((a, b, t))
    elif marker == VALID:
        valid.append((a, b, t))
    else:
        test.append((a, b, t))
"""
