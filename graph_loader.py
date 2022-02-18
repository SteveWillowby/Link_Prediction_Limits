import random

# Functions:
#
# random_coin(p)
#   -- Given a value p from the range (0, 1), returns a lambda function
#       that returns true with probability p.
#
# read_edges(edge_list_filename, directed)
#   returns a list of edges
#
# read_graph(edge_list_filename, directed, 
#            node_list_filename=None,
#            edge_remover=None)
#
# The input graph can be a multigraph, but will be flattened to a regular
#   graph. However, the multiplicity (or collection of edge types) will be
#   accounted for in the resulting "edge types."
#
# Edges and nodes can have types.
#
# All self-loops will be converted to node colors.
#
# IMPORTANT: Assumes all values are integers. Will relabel things to start at
#   zero.
#
# `edge_remover` must be either None or a function that takes an edge and
#   returns True or False. If the function returns true, the edge will be
#   excluded from the graph. Excluded edges are returned in a list. The removed
#   edges take the format of (source, target) or (source, edge_type, target)
#   if the graph has edge types.
#
# NOTE: Removed edges can be self-loops if the input graph contains self-loops.
#
# Returns:
#   ((directed, has_edge_types, old_nodes, neighbors_collections),
#    node_types,
#    removed_edges,
#    hidden_nodes, new_node_color_to_orig_color)
#
#   WHERE `directed` and `has_edge_types` are boolean values, `old_nodes` maps
#    from the indices 0 to len(nodes) to the original node labels, and
#    `neighbors_collections` is a list of containers. `neighbors_collections[i]`
#    contains information about the neighbors of the i'th node. If the graph has
#    edge_types, then `neighbors_collections[i]` is a dictionary where the keys
#    are the i's neighbors and the values are the edge types. If the graph does
#    not have edge types, then `neighbors_collections[i]` is simply the set of
#    i's neighbors.
#    `node_types` is a list giving an integer `color` to each node.
#    `removed_edges` is a list of any edges removed due to `edge_remover`.
#    `hidden_nodes` is a list of (node, orig_color) tuples of nodes whose color
#       has been replaced with a new "blank" color due to `node_label_hider`.
#       Note that if there are self loops, the hidden nodes might have MULTIPLE
#       "blank" colors - one for each type of self-loop found on blank nodes.
#    `new_node_color_to_orig_color` maps final node colors to the colors
#       obtained AFTER some nodes have had their colors hidden.

def read_graph(edge_list_filename, directed, \
               node_list_filename=None, \
               edge_remover=None, \
               node_label_hider=None):

    has_node_types = None
    has_edge_types = None
    edge_type_set = set()
    node_types = {}
    removed_edges = []
    hidden_nodes = []
    new_node_color_to_orig_color = {}

    if node_list_filename is not None:
        nodes = set()
        f = open(node_list_filename, "r")
        for line in f:
            line = line.strip()
            line = line.split(" ")
            if has_node_types is None:
                has_node_types = len(line) == 2

            if (not has_node_types) and node_label_hider is not None:
                raise ValueError("Error! Cannot use `node_label_hider` when" + \
                                 " nodes have no types.")

            if len(line) != 1 + int(has_node_types):
                raise ValueError(\
                        ("Error! All lines in %s" % node_list_filename) + \
                         " must have the same number of integers: 1 or 2.")
            node = int(line[0])
            if node in nodes:
                raise ValueError("Error! Node %d was repeated in file %s." % \
                                    (node, node_list_filename))
            nodes.add(node)
            if has_node_types:
                t = int(line[1])
                node_types[node] = t
        f.close()
        got_nodes = True
    else:
        nodes = set()
        got_nodes = False

        if node_label_hider is not None:
            raise ValueError("Error! Cannot use `node_label_hider` when" + \
                             " nodes have no types.")

    if node_label_hider is not None:
        max_label = max(node_types)
        next_label = max_label + 1
        for n in nodes:
            if node_label_hider(n):
                hidden_nodes.append((n, node_types[n]))
                node_types[n] = next_label

    f = open(edge_list_filename, "r")
    neighbors_collections = {}
    self_loop_info = {}

    for line in f:
        line = line.strip()
        line = line.split(" ")
        if has_edge_types is None:
            has_edge_types = len(line) == 3
        if len(line) != 2 + int(has_edge_types):
            raise ValueError(\
                    ("Error! All lines in %s" % edge_list_filename) + \
                    " must have the same number of integers: 2 or 3.")

        source = int(line[0])
        target = int(line[1])
        if has_edge_types:
            edge_type = int(line[2])

        if not got_nodes:
            nodes.add(source)
            nodes.add(target)

        if edge_remover is not None:
            if has_edge_types:
                edge = (source, edge_type, target)
            else:
                edge = (source, target)
            if edge_remover(edge):
                removed_edges.append(edge)
                continue

        if source == target:
            if has_edge_types:
                if source not in self_loop_info:
                    self_loop_info[source] = []
                self_loop_info[source].append(edge_type)
            else:
                if source not in self_loop_info:
                    self_loop_info[source] = 0
                self_loop_info[source] += 1
            continue

        if directed and has_edge_types:
            if source not in neighbors_collections:
                neighbors_collections[source] = {}
            if target not in neighbors_collections[source]:
                neighbors_collections[source][target] = []
            neighbors_collections[source][target].append(edge_type)
        elif directed:
            if source not in neighbors_collections:
                neighbors_collections[source] = []
            neighbors_collections[source].append(target)
        elif has_edge_types:
            if source not in neighbors_collections:
                neighbors_collections[source] = {}
            if target not in neighbors_collections[source]:
                neighbors_collections[source][target] = []
            neighbors_collections[source][target].append(edge_type)
            if target not in neighbors_collections:
                neighbors_collections[target] = {}
            if source not in neighbors_collections[target]:
                neighbors_collections[target][source] = []
            neighbors_collections[target][source].append(edge_type)
        else:
            if source not in neighbors_collections:
                neighbors_collections[source] = []
            neighbors_collections[source].append(target)
            if target not in neighbors_collections:
                neighbors_collections[target] = []
            neighbors_collections[target].append(source)

    f.close()

    if len(self_loop_info) > 0:
        print("NOTE: Input file %s had %d nodes with 1 or more self-loops." % \
                (edge_list_filename, len(self_loop_info)))

    # Zero-index nodes.
    num_nodes = len(nodes)
    nodes = sorted(list(nodes))

    if nodes[0] != 0 or nodes[-1] != len(nodes) - 1:
        print("Relabeling nodes.")

        node_relabeling = {nodes[i]: i for i in range(0, len(nodes))}
        hidden_nodes = [(node_relabeling[n], t) for (n, t) in hidden_nodes]

        if len(removed_edges) > 0:
            if len(removed_edges[0]) == 3:
                removed_edges = [(node_relabeling[a], t, node_relabeling[b]) \
                                    for (a, t, b) in removed_edges]
            else:
                removed_edges = [(node_relabeling[a], node_relabeling[b]) \
                                    for (a, b) in removed_edges]

        self_loop_info = {node_relabeling[n]: info \
                            for n, info in self_loop_info.items()}

        if has_node_types:
            new_node_types = [None for _ in range(0, len(nodes))]
            for n, i in node_relabeling.items():
                new_node_types[i] = node_types[n]
            node_types = new_node_types

        # Convert from dict to list.
        nc = [{} for _ in range(0, num_nodes)]
        for n, c in neighbors_collections.items():
            nc[node_relabeling[n]] = c
        neighbors_collections = nc

        if has_edge_types:
            neighbors_collections = \
                [{node_relabeling[nbr]: tuple(sorted(list(types))) \
                    for nbr, types in neighbors_collections[n].items()} \
                        for n in range(0, num_nodes)]
        else:
            neighbors_collections = \
                [[node_relabeling[nbr] \
                    for nbr in neighbors_collections[n]] \
                        for n in range(0, num_nodes)]
    else:
        # Convert from dict to list.
        nc = [{} for _ in range(0, num_nodes)]
        for n, c in neighbors_collections.items():
            nc[n] = c
        neighbors_collections = nc

        if has_edge_types:
            for n in range(0, num_nodes):
                neighbors_collections[n] = \
                    {nbr: tuple(sorted(list(types))) \
                        for nbr, types in neighbors_collections[n].items()}

    # Flatten (and 0-index) node types.
    has_node_types = (has_node_types is not None) and has_node_types
    if has_node_types or len(self_loop_info) > 0:
        if not has_node_types:
            if has_edge_types:
                node_types = [tuple([]) for _ in range(0, num_nodes)]
            else:
                node_types = [0 for _ in range(0, num_nodes)]
            for n in range(0, num_nodes):
                if nodes[n] in self_loop_info:
                    if has_edge_types:
                        node_types[n] = \
                            tuple(sorted(list(self_loop_info[nodes[n]])))
                    else:
                        node_types[n] = self_loop_info[nodes[n]]

        elif len(self_loop_info) > 0:
            for n in range(0, num_nodes):
                node = nodes[n]
                if node in self_loop_info:
                    if has_edge_types:
                        node_types[n] = (node_types[n], \
                                         tuple(sorted(list(self_loop_info[node]))))
                    else:
                        node_types[n] = (node_types[n], self_loop_info[node])
                else:
                    if has_edge_types:
                        node_types[n] = (node_types[n], tuple([]))
                    else:
                        node_types[n] = (node_types[n], 0)
        else:
            new_node_color_to_orig_color = \
                {t: t for t in set(node_types)}

        node_type_relabeling = sorted(list(set(node_types)))
        node_type_relabeling = {node_type_relabeling[i]: i \
                                   for i in range(0, len(node_type_relabeling))}

        if not has_node_types:
            new_node_color_to_orig_color = \
                {i: 0 for i in range(0, len(node_type_relabeling))}
        elif len(self_loop_info) > 0:
            for ((orig_color, extra), new_t) in node_type_relabeling.items():
                new_node_color_to_orig_color[new_t] = orig_color

        for n in range(0, num_nodes):
            node_types[n] = node_type_relabeling[node_types[n]]
    else:
        node_types = [0 for _ in range(0, num_nodes)]

    # Flatten (and 0-index) edge types.
    if has_edge_types:
        edge_type_relabeling = set()
        for nc in neighbors_collections:
            for _, t in nc.items():
                edge_type_relabeling.add(t)
        edge_type_relabeling = sorted(list(edge_type_relabeling))
        edge_type_relabeling = {edge_type_relabeling[i]: i \
                                   for i in range(0, len(edge_type_relabeling))}
        for n in range(0, num_nodes):
            neighbors_collections[n] = {nbr: edge_type_relabeling[t] \
                                 for nbr, t in neighbors_collections[n].items()}

    N = num_nodes
    M = sum([len(nc) for nc in neighbors_collections])
    if not directed:
        assert M % 2 == 0
        M = int(M / 2)

    if has_node_types or len(self_loop_info) > 0:
        Nt = len(node_type_relabeling)
    else:
        Nt = 1
    if has_edge_types:
        Mt = len(edge_type_relabeling)
    else:
        Mt = 1

    print(("INFO: Input graph has %d nodes, %d edges, " % (N, M)) + \
          ("%d node types (accounting for self-loops), and %d edge types." % (Nt, Mt)))

    return ((directed, has_edge_types, nodes, neighbors_collections), \
                node_types, removed_edges, \
                hidden_nodes, new_node_color_to_orig_color)

def read_edges(edge_list_filename, directed):
    f = open(edge_list_filename, "r")
    edges = []
    has_edge_types = None

    for line in f:
        line = line.strip()
        line = line.split(" ")
        if has_edge_types is None:
            has_edge_types = len(line) == 3
        if len(line) != 2 + int(has_edge_types):
            raise ValueError(\
                    ("Error! All lines in %s" % edge_list_filename) + \
                    " must have the same number of integers: 2 or 3.")

        source = int(line[0])
        target = int(line[1])
        if not directed:
            a = min(source, target)
            b = max(source, target)
            source = a
            target = b

        if has_edge_types:
            edge_type = int(line[2])
            edges.append((source, edge_type, target))
        else:
            edges.append((source, target))

    f.close()
    return edges

def random_coin(p):
    if p <= 0.0 or p >= 1.0:
        raise ValueError("Error! p must be in the range (0, 1) for " + \
                         "random_coin(). It was %f" % p)

    return (lambda x: (lambda y: random.random() < x))(p)

if __name__ == "__main__":
    print("Loading FB15k-237...")
    edgefile = "real_world_graphs/FB15k-237/FB15k-237_train_and_valid_edges.txt"
    nodefile = "real_world_graphs/FB15k-237/FB15k-237_nodes.txt"
    directed = True
    ((directed, has_edge_types, nodes, neighbors_collections), node_coloring) =\
        read_graph(edgefile, directed, node_list_filename=nodefile)
