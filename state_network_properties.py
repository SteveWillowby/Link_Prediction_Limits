from graph_loader import read_graph, read_edges, random_coin
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
    if n_args != 2:
        raise ValueError("Error! Must pass two arguments: the graph name and FRE.\n" + \
                "options for graph name are:\n\n" + \
                "karate, eucore, college_10_predict_end, " + \
                "college_10_predict_any, college_28_predict_end,\n" + \
                "college_28_predict_any, citeseer, cora, highschool, " + \
                "convote, FB15k, celegans_m,\nfoodweb, innovation, wiki," + \
                " powergrid, polblogs, y2h_ppi, collins_yeast,\n" + \
                "faculty_business, faculty_comp_sci, faculty_history, " + \
                "GR_coauth, jazz_collab,\nlinux_calls, mysql_calls, " + \
                "roget_thesaurus, roman_roads, roman_roads_p,\n" + \
                "roman_roads_u, species_1_brain, US_airports_2010, " + \
                "US_airports_2010_l, US_airports_2010_u,\n" + \
                "US_500_airports, US_500_airports_l, US_500_airports_u,\n" + \
                "ER_<n>_<m>" \
                )

    graph_name = argv[1]
    fraction_of_removed_edges = float(argv[2])

    generate_graph = graph_name[:2] == "ER"

    mode = "Link Pred"

    if generate_graph:
        properties = graph_name.split("_")[1:]
        GEN_n = int(properties[0])
        GEN_m = int(properties[1])
        directed = False
        has_edge_types = False
        has_self_loops = False
        node_coloring = [0 for _ in range(0, GEN_n)]

        test_edge_list = None
    else:
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
                  "powergrid": ("opsahl-powergrid.g", False), \
                  "polblogs": ("pol_blogs.g", \
                               "pol_blogs_node_labels.txt", True), \
                  "y2h_ppi": ("CCSB_Y2H_PPI_Network.g", False), \
                  "collins_yeast": ("collins_yeast_interactome.g", False), \
                  "faculty_business": ("faculty_hiring_business.g", True), \
                  "faculty_comp_sci": ("faculty_hiring_comp_sci.g", True), \
                  "faculty_history": ("faculty_hiring_history.g", True), \
                  "GR_coauth": ("GR_coauthorship.g", False), \
                  "jazz_collab": ("jazz_collaboration.g", False), \
                  "linux_calls": ("linux_call_graph.g", True), \
                  "mysql_calls": ("mysql_call_graph.g", True), \
                  "roget_thesaurus": ("roget_thesaurus.g", True), \
                  "roman_roads": ("roman_roads_1999.g", True), \
                  "roman_roads_p": ("roman_roads_1999_partitioned.g", True), \
                  "roman_roads_u": ("roman_roads_1999_unweighted.g", True), \
                  "species_1_brain": ("species_brain_1.g", True), \
                  "US_airports_2010": ("US_airports_2010.g", True), \
                  "US_airports_2010_l": ("US_airports_2010_log2_weights.g", True), \
                  "US_airports_2010_u": ("US_airports_2010_unweighted.g", True), \
                  "US_500_airports": ("US_top_500_airports_2002.g", True), \
                  "US_500_airports_l": ("US_top_500_airports_2002_log2_weights.g", True), \
                  "US_500_airports_u": ("US_top_500_airports_2002_unweighted.g", True) \
                    }

        assert graph_name in graph_info
        graph_info = graph_info[graph_name]

    if not generate_graph:
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

        elif len(graph_info) == 4:
            (name, node_name, test_name, directed) = graph_info
            edge_list = "real_world_graphs/%s" % name
            node_list = "real_world_graphs/%s" % node_name
            test_edge_list = "real_world_graphs/%s" % test_name
        else:
            assert "Error is" == "not here"

    if mode == "Node Classification":
        ((directed, has_edge_types, nodes, neighbors_collections), \
            node_coloring, removed_edges, \
            hidden_nodes, new_node_color_to_orig_color) = \
                read_graph(edge_list, directed, \
                           node_list_filename=node_list, \
                           node_label_hider=\
                             random_coin(fraction_of_removed_edges))

    elif test_edge_list is not None:
        print("Loading %s" % edge_list)
        ((directed, has_edge_types, nodes, neighbors_collections), \
            node_coloring, removed_edges, \
            hidden_nodes, new_node_color_to_orig_color) = \
                read_graph(edge_list, directed, node_list_filename=node_list, \
                           edge_remover=None)

    elif test_edge_list is None:
        print("Loading %s" % edge_list)
        ((directed, has_edge_types, nodes, neighbors_collections), \
            node_coloring, removed_edges, \
            hidden_nodes, new_node_color_to_orig_color) = \
                read_graph(edge_list, directed, \
                           node_list_filename=node_list, \
                           edge_remover=\
                             random_coin(fraction_of_removed_edges))

    elif generate_graph:
        print("Generating an %d, %d ER graph." % (GEN_n, GEN_m))
        nodes = [i for i in range(0, GEN_n)]
        (neighbors_collections, removed_edges, node_coloring) = \
            ER(GEN_n, GEN_m, frac_hidden=fraction_of_removed_edges, \
               directed=directed, has_self_loops=has_self_loops)
