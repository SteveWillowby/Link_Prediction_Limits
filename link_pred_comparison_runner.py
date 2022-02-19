from graph_loader import read_graph, read_edges, random_coin
from link_pred_comparison import k_hop_info_and_alternate_pred_agent
from link_pred_class_info import get_k_hop_info_classes_for_link_pred
from node_pred_class_info import get_k_hop_info_classes_for_node_pred
from max_score_functions import get_max_AUPR, get_max_ROC, estimate_min_frac_for_AUPR, get_AUPR, get_ROC
import random
import sys
import time

def __argstr_parser__(s):
    if "=" not in s:
        return s

    for i in range(0, len(s)):
        if s[i] == "=":
            return s[i + 1:]

if __name__ == "__main__":
    argv = [__argstr_parser__(s) for s in sys.argv]

    n_args = len(argv) - 1
    if n_args != 9:
        raise ValueError("Error! Must pass nine arguments:\n" + \
                "number of processes, number of threads per process, " + \
                "k, py_iso,\npercent of removed/hidden entities, " + \
                "percent of (non)entities looked at,\n" + \
                "number of runs, hash endpoints, graph name.\n" + \
                "options for graph name are:\n" + \
                "karate, eucore, college_10_predict_end, " + \
                "college_10_predict_any, college_28_predict_end,\n" + \
                "college_28_predict_any, citeseer, cora, highschool," + \
                "convote, FB15k, celegans_m,\nfoodweb, innovation, wiki" + \
                " and polblogs")

    # STOP_MARGIN is how close the k-hop performance has to be to the observed
    #   k-inf performance in order to stop.
    STOP_MARGIN = 0.005
    # DESIRED_STDEV is the maximum expected stdev of the measured AUPR from the
    #   real AUPR for a given edge set. Making it larger allows looking at
    #   fewer non-edges.
    DESIRED_STDEV = 0.025  # -- only relevant if k=all and percent_non_edges=auto

    AUTO_ESTIMATES = 3

    np = int(argv[1])
    ntpp = int(argv[2])
    k = argv[3]
    if k != "inf" and k != "all":
        k = int(k)
    py_iso = argv[4]
    assert py_iso in ["1", "0", "true", "false", "True", "False"]
    py_iso = py_iso in ["1", "true", "True"]

    if argv[5] == "auto":
        fraction_of_removed_edges = 0.1
    else:
        fraction_of_removed_edges = float(argv[5])
        if fraction_of_removed_edges == 100.0:
            fraction_of_removed_edges = 1.0
        else:
            fraction_of_removed_edges = fraction_of_removed_edges / 100.0
        assert 0.0 < fraction_of_removed_edges and fraction_of_removed_edges < 1.0

    if argv[6] == "auto":
        if k == "all":
            fraction_of_entities = "auto"
        else:
            fraction_of_entities = 1.0
    else:
        fraction_of_entities = float(argv[6])

        if fraction_of_entities == 100.0:
            fraction_of_entities = 1.0
        else:
            fraction_of_entities = fraction_of_entities / 100.0

        assert 0.0 < fraction_of_entities and fraction_of_entities <= 1.0

    num_runs = int(argv[7])

    hash_endpoints = argv[8]
    assert hash_endpoints in ["1", "0", "true", "false", "True", "False"]
    hash_endpoints = hash_endpoints in ["1", "true", "True"]

    graph_name = argv[9]

    assert graph_name in ["karate", "eucore", "citeseer", \
                          "cora", "FB15k", "wiki", "highschool", \
                          "convote", "celegans_m", "foodweb", "innovation", \
                          "college_10_predict_end", "college_10_predict_any", \
                          "college_28_predict_end", "college_28_predict_any", \
                          "polblogs"]

    graph_info = {"karate": ("karate.g", False), \
                  "eucore": ("eucore.g", True), \
                  "college_10_predict_end": \
                        ("college-temporal_10-periods_train-and-valid.txt", \
                         "college-temporal_nodes.txt", \
                         "college-temporal_10-periods_test.txt", True), \
                  "college_28_predict_end": \
                        ("college-temporal_28-weeks_train-and-valid.txt", \
                         "college-temporal_nodes.txt", \
                         "college-temporal_28-weeks_test.txt", True), \
                  "college_10_predict_any": \
                        ("college-temporal_10-periods.g", True), \
                  "college_28_predict_any": \
                        ("college-temporal_28-weeks.g", True), \
                  "citeseer": ("citeseer.g", True), \
                  "cora": ("cora.g", True), \
                  "FB15k": ("FB15k-237/FB15k-237_train_and_valid_edges.txt", \
                            "FB15k-237/FB15k-237_nodes.txt", \
                            "FB15k-237/FB15k-237_test_edges.txt", True), \
                  "wiki": ("wiki-en-additions.g", True), \
                  "convote": ("convote.g", True), \
                  "highschool": ("moreno_highschool.g", True), \
                  "celegans_m": ("celegans_metabolic.g", False), \
                  "foodweb": ("maayan-foodweb.g", True), \
                  "innovation": ("moreno_innovation.g", True), \
                  "polblogs": ("pol_blogs.g", \
                               "pol_blogs_node_labels.txt", True)}[graph_name]

    if len(graph_info) == 4:
        fraction_of_removed_edges = "NA"

    raw_output_filename = "test_results/%s_k-%s_ref-%s_nef-%s_he-%s_raw.txt" % \
                            (graph_name, k, fraction_of_removed_edges, \
                             fraction_of_entities, str(hash_endpoints).lower())
    raw_output_file = open(raw_output_filename, "w")

    mode = "Link Pred"
    main_function = k_hop_info_and_alternate_pred_agent

    if len(graph_info) == 2:
        (name, directed) = graph_info
        edge_list = "real_world_graphs/%s" % name
        node_list = None
        test_edge_list = None

    elif len(graph_info) == 3:
        (name, node_name, directed) = graph_info
        edge_list = "real_world_graphs/%s" % name
        node_list = "real_world_graphs/%s" % node_name
        test_edge_list = None

        mode = "Node Classification"
        main_function = get_k_hop_info_classes_for_node_pred
        assert fraction_of_entities != "auto"
        assert mode != "Node Classification"

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
            node_coloring, removed_edges, \
            hidden_nodes, new_node_color_to_orig_color) = \
                read_graph(edge_list, directed, node_list_filename=node_list, \
                           edge_remover=None)

        true_entities = read_edges(test_edge_list, directed)
        ente = len(true_entities)

    if fraction_of_entities == "auto":
        if test_edge_list is not None:
            print("Why are you doing auto estimates with a specified test " + \
                  "edge list? Oh well. Here goes!")
            AUTO_ESTIMATES = 1

        fraction_of_entities = []

        for i in range(0, AUTO_ESTIMATES):
            if mode == "Node Classification":
                print("(Re)Loading %s and randomly hiding node labels." % edge_list)
                ((directed, has_edge_types, nodes, neighbors_collections), \
                    node_coloring, removed_edges, \
                    hidden_nodes, new_node_color_to_orig_color) = \
                        read_graph(edge_list, directed, \
                                   node_list_filename=node_list, \
                                   node_label_hider=\
                                     random_coin(fraction_of_removed_edges))

                true_entities = hidden_nodes  # techincally only half are "true entities"
                assert set([t for (n, t) in hidden_nodes]) == set([0, 1])
                
            elif test_edge_list is None:
                print("(Re)Loading %s and randomly removing edges." % edge_list)
                ((directed, has_edge_types, nodes, neighbors_collections), \
                    node_coloring, removed_edges, \
                    hidden_nodes, new_node_color_to_orig_color) = \
                        read_graph(edge_list, directed, \
                                   node_list_filename=node_list, \
                                   edge_remover=\
                                     random_coin(fraction_of_removed_edges))

                true_entities = {}
                for edge in removed_edges:
                    if edge not in true_entities:
                        true_entities[edge] = 0
                    true_entities[edge] += 1
                if len(removed_edges[0]) == 2:
                    if set([c for _, c in true_entities.items()]) != set([1]):
                        print("NOTE: True edges were repeated. This " + \
                              "repitition is being ignored.")
                    true_entities = [(a, b) for (a, b), c in true_entities.items()]
                else:
                    true_entities = [(a, (t, c), b) for (a, t, b), c in true_entities.items()]

            (class_info, full_T, OE, alt_classes) = main_function(\
                                neighbors_collections=neighbors_collections, \
                                orig_colors=node_coloring, \
                                directed=directed, \
                                has_edge_types=has_edge_types, \
                                true_entities=true_entities, \
                                k=1, \
                                fraction_of_entities=1.0, \
                                num_processes=np, \
                                num_threads_per_process=ntpp, \
                                use_py_iso=py_iso, \
                                hash_reps=True, \
                                print_progress=False, \
                                report_only_classes_with_positives=False)

            if mode != "Node Classification":
                if len(true_entities) != sum([p for (t, p) in class_info]):
                    print("%d vs %d" % (len(true_entities), sum([p for (t, p) in class_info])))
                    print("All vs %d" % (len(set([(a, b) for (a, t, b) in true_entities]))))
                assert len(true_entities) == sum([p for (t, p) in class_info])

            print("Completed estimate run %d of %d." % (i + 1, AUTO_ESTIMATES))
            sys.stdout.flush()
            fraction_of_entities.append(\
                estimate_min_frac_for_AUPR(class_info, \
                                           desired_stdev=DESIRED_STDEV))
            del class_info

        fnes = fraction_of_entities
        fraction_of_entities = sum(fraction_of_entities) / float(AUTO_ESTIMATES)

        print(("Based on initial estimates (%s), \n" % (fnes)) + \
                "choosing to look at " + \
                "%f percent of all non-edges." % (fraction_of_entities * 100))
        if fraction_of_entities >= 0.9:
            fraction_of_entities = 1.0
            print("...but rounding up to 100% so as to avoid coin tosses.")
        sys.stdout.flush()

        raw_output_file.write(("Based on initial estimate (%s)" % (fnes)) + \
                                (",\nusing %f percent of all non-edges.\n" % \
                                    (fraction_of_entities * 100)))
    else:
        raw_output_file.write("Using %f percent of all non-edges.\n" % \
                                    (fraction_of_entities * 100))


    for _ in range(0, num_runs):

        if mode == "Node Classification":
            print("(Re)Loading %s and randomly hiding node labels." % edge_list)
            ((directed, has_edge_types, nodes, neighbors_collections), \
                node_coloring, removed_edges, \
                hidden_nodes, new_node_color_to_orig_color) = \
                    read_graph(edge_list, directed, \
                               node_list_filename=node_list, \
                               node_label_hider=\
                                 random_coin(fraction_of_removed_edges))

            # hidden_nodes = [(n, new_node_color_to_orig_color[t]) \
            #                     for n, t in hidden_nodes]
            true_entities = hidden_nodes  # techincally only half are "true entities"
            ente = sum([t for (n, t) in hidden_nodes])
            assert set([t for (n, t) in hidden_nodes]) == set([0, 1])

        elif test_edge_list is None:
            print("(Re)Loading %s and randomly removing edges." % edge_list)
            ((directed, has_edge_types, nodes, neighbors_collections), \
                node_coloring, removed_edges, \
                hidden_nodes, new_node_color_to_orig_color) = \
                    read_graph(edge_list, directed, \
                               node_list_filename=node_list, \
                               edge_remover=\
                                 random_coin(fraction_of_removed_edges))

            true_entities = {}
            for edge in removed_edges:
                if edge not in true_entities:
                    true_entities[edge] = 0
                true_entities[edge] += 1
            if len(removed_edges[0]) == 2:
                if set([c for _, c in true_entities.items()]) != set([1]):
                    print("NOTE: True edges were repeated. This " + \
                          "repitition is being ignored.")
                true_entities = [(a, b) for (a, b), c in true_entities.items()]
            else:
                true_entities = [(a, (t, c), b) for (a, t, b), c in true_entities.items()]

            ente = len(true_entities)

        ente = ente * fraction_of_entities

        sys.stdout.flush()

        if k == "all":

            base_seed = time.time()
            print("Using base_seed %s" % (base_seed))
                
            assert len(true_entities) > 0
            # First do exact computations for k=inf and k=1.
            sub_k = "inf"
            (class_info, full_T, OE, alt_classes) = main_function(\
                            neighbors_collections=neighbors_collections, \
                            orig_colors=node_coloring, \
                            directed=directed, \
                            has_edge_types=has_edge_types, \
                            true_entities=true_entities, \
                            k=sub_k, \
                            fraction_of_entities=fraction_of_entities, \
                            base_seed=base_seed, \
                            num_processes=np, \
                            num_threads_per_process=ntpp, \
                            use_py_iso=py_iso, \
                            print_progress=False, \
                            hash_reps=hash_endpoints, \
                            report_only_classes_with_positives=False)
            sys.stdout.flush()
            print("k = %s" % sub_k)
            print("Expected Num of Observed True Entities: %f" % ente)
            print("Num Classes: %d" % len(class_info))
            print("Average Class Size: %f" % (float(sum([x[0] for x in class_info])) / len(class_info)))
            print("PT/P: %f" % (float(sum([x[0] for x in class_info])) / sum([x[1] for x in class_info])))
            inf_ROC = get_max_ROC(class_info, observed_edges=OE)
            inf_AUPR = get_max_AUPR(class_info)
            print("Max ROC: %f" % inf_ROC)
            print("Max AUPR: %f" % inf_AUPR)
            sys.stdout.flush()

            raw_output_file.write("################ New Run ###############\n")
            raw_output_file.write("full_T=%d\n" % full_T)
            raw_output_file.write("observed_T=%d\n" % OE)

            raw_output_file.write("k=inf\n")

            condensed_class_info = {}
            for CI in class_info:
                if CI not in condensed_class_info:
                    condensed_class_info[CI] = 0
                condensed_class_info[CI] += 1
            condensed_class_info = sorted([(CI, c, CI[0] * c) for CI, c in condensed_class_info.items()])
            raw_output_file.write("raw_classes=%s\n" % (condensed_class_info))

            if not hash_endpoints:
                sub_k = 1
                (class_info, full_T, OE, alt_classes) = main_function(\
                                neighbors_collections=neighbors_collections, \
                                orig_colors=node_coloring, \
                                directed=directed, \
                                has_edge_types=has_edge_types, \
                                true_entities=true_entities, \
                                k=sub_k, \
                                fraction_of_entities=fraction_of_entities, \
                                base_seed=base_seed, \
                                num_processes=np, \
                                num_threads_per_process=ntpp, \
                                use_py_iso=py_iso, \
                                hash_reps=False, \
                                print_progress=False, \
                                report_only_classes_with_positives=False)
                sys.stdout.flush()
                print("k = %s" % sub_k)
                print("Expected Num of Observed True Entities: %f" % ente)
                print("Num Classes: %d" % len(class_info))
                print("Average Class Size: %f" % (float(sum([x[0] for x in class_info])) / len(class_info)))
                print("PT/P: %f" % (float(sum([x[0] for x in class_info])) / sum([x[1] for x in class_info])))
                k1_ROC = get_max_ROC(class_info, observed_edges=OE)
                k1_AUPR = get_max_AUPR(class_info)
                print("K1 ROC: %f" % k1_ROC)
                print("K1 AUPR: %f" % k1_AUPR)
                sys.stdout.flush()

                raw_output_file.write("k=1\n")

                condensed_class_info = {}
                for CI in class_info:
                    if CI not in condensed_class_info:
                        condensed_class_info[CI] = 0
                    condensed_class_info[CI] += 1
                condensed_class_info = sorted([(CI, c, CI[0] * c) for CI, c in condensed_class_info.items()])
                raw_output_file.write("raw_classes=%s\n" % (condensed_class_info))

            # Second, interpolate using the hashed subgraphs.
            print("-- Now Hashing Subgraphs --")
            sub_k = 1
            k_ROC = None
            k_AUPR = None
            while k_ROC is None or k_AUPR < (inf_AUPR - STOP_MARGIN):

                (class_info, full_T, OE, alt_classes) = main_function(\
                                neighbors_collections=neighbors_collections, \
                                orig_colors=node_coloring, \
                                directed=directed, \
                                has_edge_types=has_edge_types, \
                                true_entities=true_entities, \
                                k=sub_k, \
                                fraction_of_entities=fraction_of_entities, \
                                base_seed=base_seed, \
                                num_processes=np, \
                                num_threads_per_process=ntpp, \
                                use_py_iso=py_iso, \
                                hash_reps=True, \
                                print_progress=False, \
                                report_only_classes_with_positives=False)
                sys.stdout.flush()
                print("k = %s" % sub_k)
                print("Expected Num of Observed True Entities: %f" % ente)
                print("Num Classes: %d" % len(class_info))
                print("Average Class Size: %f" % (float(sum([x[0] for x in class_info])) / len(class_info)))
                print("PT/P: %f" % (float(sum([x[0] for x in class_info])) / sum([x[1] for x in class_info])))
                k_ROC = get_max_ROC(class_info, observed_edges=OE)
                k_AUPR = get_max_AUPR(class_info)
                print("Max ROC: %f" % k_ROC)
                print("Max AUPR: %f" % k_AUPR)
                sys.stdout.flush()

                raw_output_file.write("k=%s\n" % sub_k)

                condensed_class_info = {}
                for CI in class_info:
                    if CI not in condensed_class_info:
                        condensed_class_info[CI] = 0
                    condensed_class_info[CI] += 1
                condensed_class_info = sorted([(CI, c, CI[0] * c) for CI, c in condensed_class_info.items()])
                raw_output_file.write("raw_classes=%s\n" % (condensed_class_info))

                if sub_k == 1 and hash_endpoints:
                    raw_output_file.write("k=%s\n" % sub_k)
                    raw_output_file.write("raw_classes=%s\n" % (condensed_class_info))

                sub_k += 1

        else:
            (class_info, full_T, OE, alt_classes) = main_function(\
                            neighbors_collections=neighbors_collections, \
                            orig_colors=node_coloring, \
                            directed=directed, \
                            has_edge_types=has_edge_types, \
                            true_entities=true_entities, \
                            k=k, \
                            fraction_of_entities=fraction_of_entities, \
                            num_processes=np, \
                            num_threads_per_process=ntpp, \
                            use_py_iso=py_iso, \
                            hash_reps=True, \
                            print_progress=True, \
                            report_only_classes_with_positives=False)
            sys.stdout.flush()

            if len(true_entities) == 0:
                print("There were no test given edges for this graph.")
                sys.stdout.flush()
                continue

            condensed_class_info = {}
            for CI in class_info:
                if CI not in condensed_class_info:
                    condensed_class_info[CI] = 0
                condensed_class_info[CI] += 1
            condensed_class_info = sorted([(CI, c, CI[0] * c) for CI, c in condensed_class_info.items()])
            uniqueness_classes = []
            prev_size = 0
            for ((t, p), c, _) in condensed_class_info:
                if t > prev_size:
                    prev_size = t
                    uniqueness_classes.append([0, 0])
                uniqueness_classes[-1][0] += t * c
                uniqueness_classes[-1][1] += p * c
            uniqueness_classes = [(x[0], x[1]) for x in uniqueness_classes]
            Uniqueness_ROC = get_ROC(uniqueness_classes, observed_edges=OE)
            Alternate_ROC = get_ROC(alt_classes, observed_edges=OE)
            Uniqueness_AUPR = get_AUPR(uniqueness_classes)
            Alternate_AUPR = get_AUPR(alt_classes)
            print("\n\nROC:  Ours vs. Triadic Closure: %f vs %f" % (Uniqueness_ROC, Alternate_ROC))
            print("AUPR: Ours vs. Triadic Closure: %f vs %f\n\n" % (Uniqueness_AUPR, Alternate_AUPR))

            print("Expected Num of Observed True Entities: %f" % ente)
            print("Num Classes: %d" % len(class_info))
            print("Average Class Size: %f" % (float(sum([x[0] for x in class_info])) / len(class_info)))
            print("PT/P: %f" % (float(sum([x[0] for x in class_info])) / sum([x[1] for x in class_info])))
            print("Max ROC: %f" % get_max_ROC(class_info, observed_edges=OE))
            print("Max AUPR: %f" % get_max_AUPR(class_info))
            sys.stdout.flush()

    raw_output_file.close()
