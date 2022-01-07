from graph_loader import read_graph, random_edge_remover
from link_pred_class_info import get_k_hop_info_classes_for_link_pred
from max_score_functions import get_max_AUPR, get_max_ROC
import sys

if __name__ == "__main__":

    test_edge_fraction = 0.1

    for t in [("karate.g", False), \
              ("eucore.g", True), \
              ("college-temporal.g", True), \
              ("citeseer.g", True), \
              ("cora.g", True), \
              ("FB15k-237/FB15k-237_train_and_valid_edges.txt", \
                    "FB15k-237/FB15k-237_nodes.txt", True), \
              ("wiki-en-additions.g", True)]:

        # if ("karate" not in t[0] and "eucore" not in t[0]):
        #     continue
        if ("wiki" in t[0]):
            continue

        if len(t) == 2:
            (name, directed) = t
            edge_list = "real_world_graphs/%s" % name
            node_list = None
        elif len(t) == 3:
            (name, node_name, directed) = t
            edge_list = "real_world_graphs/%s" % name
            node_list = "real_world_graphs/%s" % node_name

        print("Loading %s" % edge_list)
        ((directed, has_edge_types, nodes, neighbors_collections), \
            node_coloring, removed_edges) = \
                read_graph(edge_list, directed, node_list_filename=node_list, \
                           edge_remover=random_edge_remover(test_edge_fraction))
        sys.stdout.flush()

        true_edges = removed_edges

        class_info = get_k_hop_info_classes_for_link_pred(\
                        neighbors_collections=neighbors_collections, \
                        orig_colors=node_coloring, \
                        directed=directed, \
                        has_edge_types=has_edge_types, \
                        true_edges=true_edges, \
                        k=1, \
                        num_processes=6, \
                        num_threads_per_process=1)
        sys.stdout.flush()

        if len(true_edges) == 0:
            print("There were not test given edges for this graph.")
            sys.stdout.flush()
            continue

        print("Num True Edges: %d" % len(true_edges))
        print("Num Classes: %d" % len(class_info))
        print("Average Class Size: %f" % (float(sum([x[1] for x in class_info])) / len(class_info)))
        print("T/P: %f" % (float(sum([x[1] for x in class_info])) / sum([x[2] for x in class_info])))

        print("Max ROC: %f" % get_max_ROC(class_info))
        print("Max AUPR: %f" % get_max_AUPR(class_info))
        sys.stdout.flush()

    exit()

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
    # class_info = get_class_info_for_target_triples(neighbors_collections=g, \
    #                                                node_colors=nc, \
    #                                                triples=test)

    class_info = get_k_hop_info_classes_for_link_pred(neighbors_collections=g, \
                                                      orig_colors=nc, \
                                                      directed=True, \
                                                      has_edge_types=True, \
                                                      true_edges=test, \
                                                      k=1)

    print("Num Triples: %d" % len(test))
    print("Num Classes: %d" % len(class_info))
    print("Average Class Size: %f" % (float(sum([x[1] for x in class_info])) / len(class_info)))
    print("T/P: %f" % (float(sum([x[1] for x in class_info])) / sum([x[2] for x in class_info])))

    print("Max ROC: %f" % get_max_ROC(class_info))
    print("Max AUPR: %f" % get_max_AUPR(class_info))
