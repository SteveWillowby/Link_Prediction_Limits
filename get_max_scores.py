from format_kg_for_nauty import *
from format_kg_for_JH_special import *
from hopeful_canonicalizer import hopeful_canonicalizer
import math
from nauty_traces_session import NautyTracesSession
from weisfeiler_lehman import WL, zero_indexed_and_map 

# Functions:
#
# get_class_info_for_target_triples(formatted_graph, node_colors, triples)
#   -- Returns (orbit_type, class size, num positives) triples for the classes
#       in which at least one of the (a, t, b) edges in `triples` falls.
#
# get_canonical_k_hop_info(formatted_graph, highlights, entity, k)
#   -- Not yet implemented.


def get_class_info_for_target_triples(graph, node_colors, triples):
    # A triple (a, t, b)'s class is also a triple (i, j, t).
    #   `i` is the index of the automorphism orbit of a.
    #   `j` is the index of the automorphism orbit of b given a is highlighted.
    #   `t` is the same.
    triple_classes = {}
    class_sizes = {}
    positives_in_class = {}

    assert type(graph) is tuple
    assert len(graph) == 3
    (out_edge_sets, in_edge_sets, edge_types) = graph
    num_nodes = len(out_edge_sets)

    print("  Overall ISO...")
    coloring_list = list(node_colors)
    hopeful_canonicalizer(out_edge_sets, coloring_list, edge_types=edge_types, \
                          in_neighbor_sets=in_edge_sets, return_canon_order=False, \
                          print_info=False, k_hop_graph_collections=None)
    print("  Done with Overall ISO.")

    main_cell_sizes = {}
    for color in coloring_list:
        if color not in main_cell_sizes:
            main_cell_sizes[color] = 0
        main_cell_sizes[color] += 1

    next_color = max(coloring_list) + 1
    print("%d orbits for %d nodes" % (next_color, num_nodes))

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
        main_a_color = coloring_list[a]
        main_b_color = coloring_list[b]
        a_cell_size = main_cell_sizes[main_a_color]
        b_cell_size = main_cell_sizes[main_b_color]
        if a_cell_size == 1 or b_cell_size == 1:
            orbit = (main_a_color, main_b_color, t, True)
            orbit_size = a_cell_size * b_cell_size
        else:
            sub_coloring = list(coloring_list)
            sub_coloring[a] = next_color
            hopeful_canonicalizer(out_edge_sets, sub_coloring, edge_types=edge_types, \
                                  in_neighbor_sets=in_edge_sets, return_canon_order=False, \
                                  print_info=False, k_hop_graph_collections=None)
            sub_cell_sizes = {}
            for color in coloring_list:
                if color not in sub_cell_sizes:
                    sub_cell_sizes[color] = 0
                sub_cell_sizes[color] += 1

            sub_b_color = sub_coloring[b]
            b_sub_cell_size = sub_cell_sizes[sub_b_color]
            orbit = (main_a_color, sub_b_color, t, False)
            orbit_size = a_cell_size * b_sub_cell_size

        if orbit not in triple_classes:
            triple_classes.add(orbit)
            class_sizes[orbit] = orbit_size
            positives_in_class[orbit] = 0
        positives_in_class[orbit] += 1

    return [(tc, class_sizes[tc], positives_in_class[tc]) for tc in triple_classes]

def get_canonical_k_hop_info(formatted_graph, highlights, entity, k):
    pass

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

"""
def get_canonical_node_orbits(canon_node_order, orbits, filter_tuples=True):
    node_order_by_node = {}
    for i in range(0, len(canon_node_order)):
        node = canon_node_order[i]
        if filter_tuples and type(node) is tuple:
            continue
        node_order_by_node[node] = i
    orbits_by_node = {}
    orbit_labels = []
    orbit_sizes = {}
    for orbit in orbits:
        if filter_tuples and type(orbit[0]) is tuple:
            continue
        canonical_orbit_label = min([node_order_by_node[n] for n in orbit])
        orbit_labels.append(canonical_orbit_label)
        orbit_sizes[canonical_orbit_label] = len(orbit)
        for node in orbit:
            orbits_by_node[node] = canonical_orbit_label

    orbit_labels.sort()
    orbit_labels = {orbit_labels[i]: i for i in range(0, len(orbit_labels))}
    orbits_by_node = {n: orbit_labels[l] for n, l in orbits_by_node.items()}
    orbit_sizes = {orbit_labels[l]: s for l, s in orbit_sizes.items()}
    return (orbits_by_node, orbit_sizes)
"""

if __name__ == "__main__":
    print("Loading FB15k-237...")
    (train, valid, test, id_to_mid, id_to_type) = load_FB15k_237()
    train_nodes = set([x[0] for x in train] + [x[2] for x in train])
    valid_nodes = set([x[0] for x in valid] + [x[2] for x in valid])
    test_nodes  = set([x[0] for x in test]  + [x[2] for x in test])

    print("  New Formatting...")
    graph = format_kg_for_JH_special(train + valid, \
                                     nodes=(train_nodes | valid_nodes | test_nodes))

    print("Computing Triple Class Sizes...")
    node_colors = [0 for _ in range(0, len(graph[0]))]
    class_info = get_class_info_for_target_triples(graph, node_colors, test)

    print("Num Triples: %d" % len(test))
    print("Num Classes: %d" % len(class_info))
    print("Average Class Size: %f" % (float(sum([x[1] for x in class_info])) / len(class_info)))
    print("T/P: %f" % (float(sum([x[1] for x in class_info])) / sum([x[2] for x in class_info])))

    print("Max ROC: %f" % get_max_ROC(class_info))
    print("Max AUPR: %f" % get_max_AUPR(class_info))
