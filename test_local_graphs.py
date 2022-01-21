from graph_loader import read_graph, read_edges, random_edge_remover
from link_pred_class_info import get_k_hop_info_classes_for_link_pred
from max_score_functions import get_max_AUPR, get_max_ROC
import sys

def __argstr_parser__(s):
    if "=" not in s:
        return s

    for i in range(0, len(s)):
        if s[i] == "=":
            return s[i + 1:]

if __name__ == "__main__":
    argv = [__argstr_parser__(s) for s in sys.argv]

    n_args = len(argv) - 1
    if n_args != 6:
        raise ValueError("Error! Must pass five arguments:\n" + \
                "number of processes, number of threads per process, " + \
                "k, py_iso, number of runs, graph name.\n" + \
                "options for graph name are:\n" + \
                "karate, eucore, college, citeseer, cora, FB15k, and wiki")

    TEST_EDGE_FRACTION = 0.1

    np = int(argv[1])
    ntpp = int(argv[2])
    k = argv[3]
    if k != "inf" and k != "all":
        k = int(k)
    py_iso = argv[4]
    assert py_iso in ["1", "0", "true", "false", "True", "False"]
    py_iso = py_iso in ["1", "true", "True"]
    num_runs = int(argv[5])
    graph_name = argv[6]

    assert graph_name in ["karate", "eucore", "college", "citeseer", \
                          "cora", "FB15k", "wiki"]

    graph_info = {"karate": ("karate.g", False), \
                  "eucore": ("eucore.g", True), \
                  "college": ("college-temporal.g", True), \
                  "citeseer": ("citeseer.g", True), \
                  "cora": ("cora.g", True), \
                  "FB15k": ("FB15k-237/FB15k-237_train_and_valid_edges.txt", \
                            "FB15k-237/FB15k-237_nodes.txt", \
                            "FB15k-237/FB15k-237_test_edges.txt", True), \
                  "wiki": ("wiki-en-additions.g", True)}[graph_name]


    if len(graph_info) == 2:
        (name, directed) = graph_info
        edge_list = "real_world_graphs/%s" % name
        node_list = None
        test_edge_list = None
    elif len(graph_info) == 4:
        (name, node_name, test_name, directed) = graph_info
        edge_list = "real_world_graphs/%s" % name
        node_list = "real_world_graphs/%s" % node_name
        test_edge_list = "real_world_graphs/%s" % test_name
    else:
        assert "Error is" == "not here"

    if test_edge_list is not None:
        print("Loading %s" % edge_list)
        ((directed, has_edge_types, nodes, neighbors_collections), \
            node_coloring, removed_edges) = \
                read_graph(edge_list, directed, node_list_filename=node_list, \
                           edge_remover=None)

        true_edges = read_edges(test_edge_list, directed)

    for _ in range(0, num_runs):

        if test_edge_list is None:
            print("(Re)Loading %s and randomly removing edges." % edge_list)
            ((directed, has_edge_types, nodes, neighbors_collections), \
                node_coloring, removed_edges) = \
                    read_graph(edge_list, directed, node_list_filename=node_list, \
                               edge_remover=random_edge_remover(TEST_EDGE_FRACTION))
            true_edges = removed_edges

        sys.stdout.flush()

        if k == "all":
            assert len(true_edges) > 0
            # First do exact computations for k=inf and k=1.
            sub_k = "inf"
            class_info = get_k_hop_info_classes_for_link_pred(\
                            neighbors_collections=neighbors_collections, \
                            orig_colors=node_coloring, \
                            directed=directed, \
                            has_edge_types=has_edge_types, \
                            true_edges=true_edges, \
                            k=sub_k, \
                            num_processes=np, \
                            num_threads_per_process=ntpp, \
                            use_py_iso=py_iso, \
                            print_progress=False)
            sys.stdout.flush()
            print("k = %s" % sub_k)
            print("Num True Edges: %d" % len(true_edges))
            print("Num Classes: %d" % len(class_info))
            print("Average Class Size: %f" % (float(sum([x[1] for x in class_info])) / len(class_info)))
            print("T/P: %f" % (float(sum([x[1] for x in class_info])) / sum([x[2] for x in class_info])))
            inf_ROC = get_max_ROC(class_info)
            inf_AUPR = get_max_AUPR(class_info)
            print("Max ROC: %f" % inf_ROC)
            print("Max AUPR: %f" % inf_AUPR)
            sys.stdout.flush()

            sub_k = 1
            class_info = get_k_hop_info_classes_for_link_pred(\
                            neighbors_collections=neighbors_collections, \
                            orig_colors=node_coloring, \
                            directed=directed, \
                            has_edge_types=has_edge_types, \
                            true_edges=true_edges, \
                            k=sub_k, \
                            num_processes=np, \
                            num_threads_per_process=ntpp, \
                            use_py_iso=py_iso, \
                            hash_subgraphs=False, \
                            print_progress=False)
            sys.stdout.flush()
            print("k = %s" % sub_k)
            print("Num True Edges: %d" % len(true_edges))
            print("Num Classes: %d" % len(class_info))
            print("Average Class Size: %f" % (float(sum([x[1] for x in class_info])) / len(class_info)))
            print("T/P: %f" % (float(sum([x[1] for x in class_info])) / sum([x[2] for x in class_info])))
            k1_ROC = get_max_ROC(class_info)
            k1_AUPR = get_max_AUPR(class_info)
            print("K1 ROC: %f" % k1_ROC)
            print("K1 AUPR: %f" % k1_AUPR)

            # Second, interpolate using the hashed subgraphs.
            print("-- Now Hashing Subgraphs --")
            MARGIN = 0.001
            sub_k = 1
            k_ROC = None
            k_AUPR = None
            while k_ROC is None or k_ROC < (inf_ROC - MARGIN) or \
                                   k_AUPR < (inf_AUPR - MARGIN):

                class_info = get_k_hop_info_classes_for_link_pred(\
                                neighbors_collections=neighbors_collections, \
                                orig_colors=node_coloring, \
                                directed=directed, \
                                has_edge_types=has_edge_types, \
                                true_edges=true_edges, \
                                k=sub_k, \
                                num_processes=np, \
                                num_threads_per_process=ntpp, \
                                use_py_iso=py_iso, \
                                hash_subgraphs=True, \
                                print_progress=False)
                sys.stdout.flush()
                print("k = %s" % sub_k)
                print("Num True Edges: %d" % len(true_edges))
                print("Num Classes: %d" % len(class_info))
                print("Average Class Size: %f" % (float(sum([x[1] for x in class_info])) / len(class_info)))
                print("T/P: %f" % (float(sum([x[1] for x in class_info])) / sum([x[2] for x in class_info])))
                k_ROC = get_max_ROC(class_info)
                k_AUPR = get_max_AUPR(class_info)
                print("Max ROC: %f" % k_ROC)
                print("Max AUPR: %f" % k_AUPR)
                sys.stdout.flush()

                sub_k += 1

        else:
            class_info = get_k_hop_info_classes_for_link_pred(\
                            neighbors_collections=neighbors_collections, \
                            orig_colors=node_coloring, \
                            directed=directed, \
                            has_edge_types=has_edge_types, \
                            true_edges=true_edges, \
                            k=k, \
                            num_processes=np, \
                            num_threads_per_process=ntpp, \
                            use_py_iso=py_iso, \
                            hash_subgraphs=True, \
                            print_progress=True)
            sys.stdout.flush()

            if len(true_edges) == 0:
                print("There were no test given edges for this graph.")
                sys.stdout.flush()
                continue

            print("Num True Edges: %d" % len(true_edges))
            print("Num Classes: %d" % len(class_info))
            print("Average Class Size: %f" % (float(sum([x[1] for x in class_info])) / len(class_info)))
            print("T/P: %f" % (float(sum([x[1] for x in class_info])) / sum([x[2] for x in class_info])))

            print("Max ROC: %f" % get_max_ROC(class_info))
            print("Max AUPR: %f" % get_max_AUPR(class_info))
            sys.stdout.flush()
