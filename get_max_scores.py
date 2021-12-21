from kg_loader import load_FB15k_237, convert_triples_to_NT_graph
import math
from ram_friendly_NT_session import RAMFriendlyNTSession

# Functions:
#
# get_class_info_for_target_triples(formatted_graph, node_colors, triples)
#   -- Returns (orbit_type, class size, num positives) triples for the classes
#       in which at least one of the (a, t, b) edges in `triples` falls.
#
# get_canonical_k_hop_info(formatted_graph, highlights, entity, k)
#   -- Not yet implemented.


def get_class_info_for_target_triples(neighbors_collections, node_colors, triples):
    # A triple (a, t, b)'s class is also a triple (i, j, t).
    #   `i` is the index of the automorphism orbit of a.
    #   `j` is the index of the automorphism orbit of b given a is highlighted.
    #   `t` is the same.
    triple_classes = {}
    class_sizes = {}
    positives_in_class = {}

    num_nodes = len(neighbors_collections)
    # TODO: Update

    print("  Overall ISO...")
    coloring_list = list(node_colors)
    session = RAMFriendlyNTSession(directed=True, \
                                   has_edge_types=True, \
                                   neighbors_collections=neighbors_collections, \
                                   kill_py_graph=True, \
                                   only_one_call=False)
    session.set_colors_by_coloring(node_colors)
    base_orbits = session.get_automorphism_orbits()
    session.run()
    base_orbits = base_orbits.get()
    print("  Done with Overall ISO.")

    base_orbit_colors = [0 for _ in range(0, num_nodes)]
    for i in range(0, len(base_orbits)):
        for n in base_orbits[i]:
            base_orbit_colors[n] = i

    main_cell_sizes = [len(o) for o in base_orbits]

    next_color = len(base_orbits)
    print("  Found %d base orbits for %d nodes" % (next_color, num_nodes))

    triple_classes = set()
    class_sizes = {}
    positives_in_class = {}

    percent_done = 0
    for i in range(0, len(triples)):
        percent = int((100 * i) / len(triples))
        if percent > percent_done:
            percent_done = percent
            print("%d percent done" % percent_done)

        (a, t, b) = triples[i]
        main_a_color = base_orbit_colors[a]
        main_b_color = base_orbit_colors[b]
        a_cell_size = main_cell_sizes[main_a_color]
        b_cell_size = main_cell_sizes[main_b_color]
        if a_cell_size == 1 or b_cell_size == 1:
            orbit = (main_a_color, main_b_color, t, True)
            orbit_size = a_cell_size * b_cell_size
        else:
            base_orbits[main_a_color].remove(a)
            base_orbits.append([a])
            session.set_colors_by_partitions(base_orbits)
            sub_orbits = session.get_automorphism_orbits()
            session.run()
            sub_orbits = sub_orbits.get()
            base_orbits[main_a_color].append(a)
            base_orbits.pop()

            sub_cell_sizes = [len(o) for o in sub_orbits]

            for j in range(0, len(sub_orbits)):
                if b in sub_orbits[j]:
                    sub_b_color = j
                    break

            b_sub_cell_size = sub_cell_sizes[sub_b_color]
            orbit = (main_a_color, sub_b_color, t, False)
            orbit_size = a_cell_size * b_sub_cell_size

        if orbit not in triple_classes:
            triple_classes.add(orbit)
            class_sizes[orbit] = orbit_size
            positives_in_class[orbit] = 0
        positives_in_class[orbit] += 1

    session.end_session()

    return [(tc, class_sizes[tc], positives_in_class[tc]) for tc in triple_classes]


# If you only look at the nodes within k hops of (head, tail),
#   then what are the classes?
#
# Warning! This can run up to num_nodes^2 times.
#   It can be slow and use LOTS of ram.
#
# It makes one call per edge class.
#
# NOTE: true_edges can be a container of (a, b) pairs or (a, t, b) triples
#   depending on whether or not the graph has edge types.
def get_k_hop_info_classes_for_link_pred(neighbors_collections, node_colors, \
                                         directed, \
                                         has_edge_types, \
                                         true_edges, k):
    assert type(node_colors[0]) is int or type(node_colors[0]) is list

    if type(true_edges) is not set:
        true_edges = set(true_edges)

    if has_edge_types:
        if directed:
            edge_is_true = set([(a, b) for (a, t, b) in true_edges])
        else:
            edge_is_true = \
                set([(min(a, b), max(a, b)) for (a, t, b) in true_edges])
    else:
        if directed:
            edge_is_true = true_edges
        else:
            edge_is_true = set([(min(a,b), max(a,b)) for (a, b) in true_edges])

    num_nodes = len(neighbors_collections)

    if directed:
        if has_edge_types:
            neighbors = [set([n for n, _ in d.items()]) \
                            for d in neighbors_collections]
            for a in range(0, num_nodes):
                for b, _ in neighbors_collections[a].items():
                    neighbors[b].add(a)
        else:
            neighbors = [set(nc) for nc in neighbors_collections]
            for a in range(0, num_nodes):
                for b in neighbors_collections[a]:
                    neighbors[b].add(a)
    else:
        if has_edge_types:
            neighbors = [set([n for n, _ in d.items()]) \
                            for d in neighbors_collections]
        else:
            neighbors = [set(nc) for nc in neighbors_collections]

    if type(node_colors[0]) is int:
        colors = sorted(list(set(node_colors)))
        colors = {colors[i]: i for i in range(0, colors)}
        nc = [[] for c in colors]
        for n in range(0, num_nodes):
            nc[colors[node_colors[n]]].append(n)
        node_colors = nc

    # Note: Make sure to use the _external_, _base_ colors when making a
    #   canonical representation of a subgraph.
    main_session = \
        RAMFriendlyNTSession(directed=directed, \
                             has_edge_types=has_edge_types, \
                             neighbors_collections=neighbors_collections, \
                             only_one_call=False, \
                             kill_py_graph=False, \
                             mode="Traces")
    main_session.set_colors_by_partitions(node_colors)
    base_orbits = main_session.get_automorphism_orbits()
    main_session.run()
    base_orbits = base_orbits.get()

    base_orbit_colors = [0 for _ in range(0, num_nodes)]
    for i in range(0, len(base_orbits)):
        for n in base_orbits[i]:
            base_orbit_colors[n] = i

    main_cell_sizes = [len(o) for o in base_orbits]

    print("  Found %d base orbits for %d nodes" % \
            (len(base_orbits), num_nodes))

    k_hop_edge_classes = set()
    class_sizes = {}
    positives_in_class = {}

    t = None
    percent_done = 0
    for i in range(0, len(base_orbits)):
        percent = int((100 * i) / len(true_edges))
        if percent > percent_done:
            percent_done = percent
            print("%d percent done" % percent_done)

        a = min(base_orbits[i])
        main_a_color = i
        a_cell_size = main_cell_sizes[main_a_color]
        a_b_candidates = []
        if len(base_orbits[i]) == 1:
            if directed:
                start = 0
            else:
                start = i + 1
            for j in range(start, len(base_orbits)):
                if j == i:
                    continue
                b = base_orbits[j][0]
                main_b_color = j
                b_cell_size = main_cell_sizes[main_b_color]
                b_candidates.append((b, main_b_color, b_cell_size))
        else:
            base_orbits.append([a])
            base_orbits[main_a_color].remove(a)
            main_session.set_colors_by_partition(base_orbits)
            base_orbits[main_a_color].append(a)
            base_orbits.pop()
            sub_orbits = main_session.get_automorphism_orbits()
            main_session.run()
            sub_orbits = sub_orbits.get()
            for j in range(0, len(sub_orbits)):
                b = min(sub_orbits[j])
                if b == a:
                    continue
                if directed or a < b:
                    for k in range(0, len(base_orbits)):
                        if b in base_orbits[k]:
                            main_b_color = k
                            break
                    b_candidates.append((b, main_b_color, \
                                         a_cell_size * len(sub_orbits[j])))

        for (b, main_b_color, main_size) in a_b_candidates:
            b_cell_size = main_cell_sizes[main_b_color]

            # TODO: Set aside colors for a and b to make sure they are treated
            #   as special within the k-hop subgraph.
            if main_a_color == main_b_color and a_cell_size == 2:
                pass
            elif a_cell_size > 1 and b_cell_size > 1:
                pass
            elif a_cell_size > 1:
                pass
            elif b_cell_size > 1:
                pass

            # TODO: Set k_hop_class via a canonicalization around (a, b).

            # TODO: Restore colors for a and b.
            if main_a_color == main_b_color and a_cell_size == 2:
                pass
            elif a_cell_size > 1 and b_cell_size > 1:
                pass
            elif a_cell_size > 1:
                pass
            elif b_cell_size > 1:
                pass

            if k_hop_class not in k_hop_edge_classes:
                k_hop_edge_classes.add(k_hop_class)
                class_sizes[k_hop_class] = 0

                # TODO: Figure out how to get positives in class via the code above.
                positives_in_class[k_hop_class] = 0
            class_sizes[k_hop_class] += main_size

    session.end_session()

def __k_hop_nodes__(neighbors, k, init_nodes):
    visited = set(init_nodes)
    frontier = set(init_nodes)
    for _ in range(0, k):
        new_frontier = set()
        for n in frontier:
            new_frontier |= neighbors[n] - visited
        visited |= new_frontier
        frontier = new_frontier
    return visited

def __induced_subgraph__(neighbors_collections, node_colors, \
                         nodes, has_edge_types):
    num_nodes = len(nodes)
    nodes_list = sorted(list(nodes))
    nodes = {nodes_list[i]: i for i in range(0, num_nodes)}

    # Get a partitioning corresponding to the new colors.
    new_colors = [node_colors[n] for n in nodes_list]
    new_colors = sorted(list(set(new_colors)))
    ncs = {new_colors[i]: i for i in range(0, len(new_colors))}
    new_colors = [[] for _ in range(0, len(new_colors)]
    for i in range(0, num_nodes):
        new_colors[ncs[node_colors[nodes[i]]]].append(i)
    del ncs

    if has_edge_types:
        observed_edge_types = set()
        new_neighbors_collections = [{} for _ in range(0, num_nodes)]
        for a in range(0, num_nodes):
            old_a = nodes_list[a]
            for old_b, t in neighbors_collections[old_a].items():
                if old_b in nodes:
                    observed_edge_types.add(t)
                    new_neighbors_collections[a][nodes[old_b]] = t

        observed_edge_types = tuple(sorted(list(observed_edge_types)))

        if observed_edge_types[-1] == len(observed_edge_types) - 1:
            # If there is effectively no relabeling, don't bother to relabel.
            return (nodes_list, new_neighbors_collections, new_colors, None)

        et_relabeling = {observed_edge_types[i]: i \
                            for i in range(0, len(observed_edge_types))}
        for a in range(0, num_nodes):
            new_neighbors_collections[a] = {b: et_relabeling[t] for \
                            b, t in new_neighbors_collections[a].items()}
        del et_relabeling

        return (nodes_list, new_neighbors_collections, \
                new_colors, observed_edge_types)
    else:
        new_neighbors_collections = [[] for _ in range(0, num_nodes)]
        for n in range(0, num_nodes):
            old_n = nodes_list[n]
            for old_neighbor neighbors_collections[old_n]:
                if old_neighbor in nodes:
                    new_neighbors_collections[n].append(nodes[old_neighbor])
        return (nodes_list, new_neighbors_collections, new_colors, None)

def __canon_rep__(new_node_to_old, g, new_colors, old_colors, \
                  et_relabeling, directed, has_edge_types):
    num_nodes = len(g)

    session = RAMFriendlyNTSession(directed=directed, \
                                   has_edge_types, \
                                   neighbors_collections=g, \
                                   only_one_call=True, \
                                   kill_py_graph=False, \
                                   mode="Traces")
    session.set_colors_by_partitions(new_colors)
    node_order = session.get_canonical_order()
    session.run()
    session.end_session()
    node_order = node_order.get()
    node_to_order = {node_order[i]: i for i in range(0, num_nodes)}

    if has_edge_types:
        edge_list = \
            tuple([tuple(sorted([(node_to_order[n2], t) for n2, t in
                    g[n].items()])) for n in node_order])
    else:
        edge_list = \
            tuple([tuple(sorted([node_to_order[n2] for n2 in \
                    g[n]])) for n in node_order])

    old_colors_in_order = \
        tuple([old_colors[new_node_to_old[n]] for n in old_nodes_sorted])

    return (num_nodes, et_relabeling, edge_list, old_colors_in_order)

# class_info is a collection of triples:
#   (class_label, class_size, num_positives_in_class)
def get_max_AUPR(class_info):
    class_info = [(float(x[1]) / x[2], x[2], x[1]) for x in class_info]
    class_info.sort()
    class_info = [(x[1], x[2]) for x in class_info]  # Positives, Total Size
    P = sum([x[0] for x in class_info])
    T = sum([x[1] for x in class_info])
    N = T - P

    b = 0.0
    d = 0.0
    AUPR = 0.0
    for (a, c) in class_info:
        a = float(a)
        c = float(c)
        if a == 0.0:
            addition = 0.0
        elif d == 0.0:
            # Only occurs once.
            addition = (a * a) / c
        else:
            addition = ((a * a) / c) * (1.0 + ((b / a) - (d / c)) * math.log((d + c) / d))

        assert addition >= 0.0
        # print("a: %f, b: %f, c: %f, d: %f -----> %f" % (a, b, c, d, addition))
        AUPR += addition

        b += a
        d += c
    AUPR /= float(P)
    return AUPR

def get_max_ROC(class_info):
    class_info = [(float(x[1]) / x[2], x[2], x[1]) for x in class_info]
    class_info.sort()
    class_info = [(x[1], x[2]) for x in class_info]  # Positives, Total Size
    P = sum([x[0] for x in class_info])
    T = sum([x[1] for x in class_info])
    N = T - P
    print("T: %d, P: %d, N: %d" % (T, P, N))
    n_acc = 0
    p_acc = 0
    TPR = []  # Goes up from 0 to 1
    FPR = []  # Goes up from 0 to 1
    for (p, t) in class_info:
        p_acc += p
        n_acc += t - p
        TPR.append(float(p_acc) / P)
        FPR.append(float(n_acc) / N)
    ROC = 0.0
    for i in range(1, len(TPR)):
        tpr_a = TPR[i - 1]
        tpr_b = TPR[i]
        fpr_a = FPR[i - 1]
        fpr_b = FPR[i]
        addition = (fpr_b - fpr_a) * ((tpr_a + tpr_b) / 2.0)
        assert addition >= 0.0
        ROC += addition
    return ROC

if __name__ == "__main__":
    print("Loading FB15k-237...")
    (train, valid, test, id_to_mid, id_to_type) = load_FB15k_237()
    train_nodes = set([x[0] for x in train] + [x[2] for x in train])
    valid_nodes = set([x[0] for x in valid] + [x[2] for x in valid])
    test_nodes  = set([x[0] for x in test]  + [x[2] for x in test])
    all_nodes = train_nodes | valid_nodes | test_nodes

    print("  New Formatting...")
    (g, nc) = convert_triples_to_NT_graph(train + valid, \
                                          num_nodes=len(all_nodes))

    print("Computing Triple Class Sizes...")
    class_info = get_class_info_for_target_triples(neighbors_collections=g, \
                                                   node_colors=nc, \
                                                   triples=test)

    print("Num Triples: %d" % len(test))
    print("Num Classes: %d" % len(class_info))
    print("Average Class Size: %f" % (float(sum([x[1] for x in class_info])) / len(class_info)))
    print("T/P: %f" % (float(sum([x[1] for x in class_info])) / sum([x[2] for x in class_info])))

    print("Max ROC: %f" % get_max_ROC(class_info))
    print("Max AUPR: %f" % get_max_AUPR(class_info))
