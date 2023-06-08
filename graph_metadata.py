from graph_loader import read_graph
import networkx as nx
import sys

if __name__ == "__main__":

    # Maps graph name to (edgelist, <optional> nodelist, directed)
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
                  "gcn_cora": ("gcn_cora_nosl_edgelist.txt", False), \
                  "gcn_citeseer": ("gcn_citeseer_nosl_edgelist.txt", \
                                   "gcn_citeseer_nosl_nodelist.txt", False), \
                  "gcn_pubmed": ("gcn_pubmed_nosl_edgelist.txt", False), \
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
                  "polblogs": ("pol_blogs.g", True), \
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

    if len(sys.argv) < 2:
        print("Error! Must pass graph name.")
        graph_name = ""
    else:
        graph_name = sys.argv[1]

    if graph_name not in graph_info:
        print("Error! Graph name \"%s\" not in list of graphs." % graph_name)
        print("    Your options are:")
        for (name, _) in graph_info.items():
            print("\t" + name)
        exit(0)

    graph_info = graph_info[graph_name]

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

        # mode = "Node Classification"
        # main_function = get_k_hop_info_classes_for_node_pred
        # assert fraction_of_entities != "auto"
        # if Null_Model_gen:
        #     raise ValueError("Error! Currently unprepared for null " + \
        #                      "model gen on node classification graphs.")

    elif len(graph_info) == 4:
        (name, node_name, test_name, directed) = graph_info
        edge_list = "real_world_graphs/%s" % name
        node_list = "real_world_graphs/%s" % node_name
        test_edge_list = "real_world_graphs/%s" % test_name
    else:
        assert "Graph name not in dict" == "false"





    if test_edge_list is not None:
        ((directed, has_edge_types, nodes, neighbors_collections), \
            node_coloring, removed_edges, \
            hidden_nodes, new_node_color_to_orig_color) = \
                read_graph(edge_list, directed, \
                           node_list_filename=node_list, \
                           edge_remover=None)
    else:
        ((directed, has_edge_types, nodes, neighbors_collections), \
            node_coloring, removed_edges, \
            hidden_nodes, new_node_color_to_orig_color) = \
                read_graph(edge_list, directed, \
                           node_list_filename=node_list, \
                           edge_remover=None)

    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    num_nodes = len(nodes)
    num_edges = 0

    for n in range(0, num_nodes):
        graph.add_node(n)
    assert len(nodes) == len(neighbors_collections)
    for n in range(0, num_nodes):
        s = neighbors_collections[n]
        num_edges += len(s)
        for nbr in s:
            graph.add_edge(n, nbr)

    if not directed:
        num_edges = int(num_edges / 2)

    print("Avg. Degree: \t%f" % (2 * num_edges / num_nodes))
    print("Avg. Clustering Coefficient: \t%f" % \
          nx.algorithms.cluster.average_clustering(graph))

    path_lengths = dict(nx.shortest_path_length(graph))
    diameter = 0
    total = 0
    num_pairs = 0
    for node, sub_dict in path_lengths.items():
        for other_node, value in sub_dict.items():
            num_pairs += 1
            total += value
            if value > diameter:
                diameter = value
    print("Diameter: \t%d" % diameter)
    print("Avg. Shortest Path Length: \t%f" % (total / num_pairs))
