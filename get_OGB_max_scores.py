import numpy as np
from ogb.lsc import MAG240MDataset
from ogb.lsc import WikiKG90Mv2Dataset
from ogb.lsc import PCQM4Mv2Dataset
# from Practical_Isomorphism_Alg.views import GraphView
# from Practical_Isomorphism_Alg.coloring import Coloring
# from Practical_Isomorphism_Alg.main_algorithm import hopeful_canonicalizer, canonical_representation
from hopeful_canonicalizer import hopeful_canonicalizer

# Class Info Format:
#
# [(class_ID, class_size, positives_in_class)]

dataset_base = "/data/datasets/open_graph_benchmark_LSC_2021"

graph_classification_root = dataset_base + "/PCQM4M"
link_pred_root = dataset_base + "/WikiKG90M"
node_classification_root = dataset_base + "/MAG240M"

def link_pred_dataset():
    dataset = WikiKG90Mv2Dataset(root=link_pred_root)

    train_hrt = dataset.train_hrt
    (num_triples, _) = train_hrt.shape

    nodes = set()

    # Get edges and flatten edge types.
    print("Loading edges...")
    edges = set([(train_hrt[i,0], train_hrt[i,2]) for i in range(0, num_triples)])
    edge_types = {edge: set() for edge in edges}
    print("Flattening edge types...")
    for i in range(0, num_triples):
        edge = (train_hrt[i,0], train_hrt[i,2])
        t = train_hrt[i,1]
        edge_types[edge].add(t)

    next_type_id = 0
    edge_type_combo_dict = {}
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        types = sorted(tuple(edge_types[edge]))
        if types not in edge_type_combo_dict:
            edge_type_combo_dict[types] = next_type_id
            next_type_id += 1
        edge_types[edge] = edge_type_combo_dict[types]
    print("  ... %d total flattened edge types." % len(edge_type_combo_dict))

    valid_task = dataset.valid_dict['h,r->t']
    hr = valid_task['hr']
    t_candidate = valid_task['t_candidate']
    t_correct_index = valid_task['t_correct_index']
    (num_predictions, _) = hr.shape

    print("Num nodes before adding validation nodes: %d" % len(nodes))
    for i in range(0, num_predictions):
        nodes.add(hr[i,0])
        for j in range(0, 1001):
            nodes.add(t_candidate[i,j])

    assert max(nodes) == len(nodes) - 1
    assert min(nodes) == 0
    print("  ... obtained %d nodes" % len(nodes))

    directed = True

    # New Way:
    # graph = GraphView(nodes, edges, directed, edge_types=edge_types)
    # node_coloring = Coloring([0 for _ in nodes])

    # Old Way:
    ons = {n: set() for n in nodes}
    ins = {n: set() for n in nodes}
    for (a, b) in edges:
        ons[a].add(b)
        ins[b].add(a)
    graph = (ons, ins, directed, edge_types)
    node_coloring = [0 for _ in nodes]

    return (graph, node_coloring, hr, t_candidate, t_correct_index)

def node_classifier_dataset():
    dataset = MAG240MDataset(root=node_classification_root)

    print("  Dataset Info:")
    print("  %d" % dataset.num_papers) # number of paper nodes
    print("  %d" % dataset.num_authors) # number of author nodes
    print("  %d" % dataset.num_institutions) # number of institution nodes
    print("  %d" % dataset.num_paper_features) # dimensionality of paper features
    print("  %d" % dataset.num_classes) # number of subject area classes

    edge_index_writes = dataset.edge_index('author', 'paper')
    edge_index_cites = dataset.edge_index('paper', 'paper')
    edge_index_affiliated_with = dataset.edge_index('author', 'institution')

    # Combine all indices.
    author_node_offset = int(dataset.num_papers)
    institution_node_offset = int(dataset.num_papers + dataset.num_authors)

    node_colors = [2 for _ in range(0, int(dataset.num_papers))] + \
                  [1 for _ in range(0, int(dataset.num_authors))] + \
                  [0 for _ in range(0, int(dataset.num_institutions))]
    edges = []
    edge_types = None  # Can be inferred from node types.

    next_paper_type_id = 3  # 2 is preserved for validation nodes
    paper_type_map = {}

    validation_node_labels = {}

    # Author-Paper Edges
    for i in range(0, edge_index_writes.shape[1]):
        author = edge_index_writes[0,i] + author_node_offset
        paper = edge_index_writes[1,i]

        paper_type = (dataset.paper_label[paper], dataset.paper_year[paper])
        if int(paper_type[1]) != 2019 or np.isnan(paper_type[0]):
            if paper_type not in paper_type_map:
                paper_type_map[paper_type] = next_paper_type_id
                next_paper_type_id += 1
            node_colors[paper] = paper_type_map[paper_type]
        else:
            validation_node_labels[paper] = int(paper_type[0])

        edges.append((author, paper))

    # Paper-Paper Edges
    for i in range(0, edge_index_cites.shape[1]):
        paper_A = edge_index_cites[0,i]
        paper_B = edge_index_cites[1,i]

        for paper in [paper_A, paper_B]:
            paper_type = (dataset.paper_label[paper], dataset.paper_year[paper])
            if int(paper_type[1]) != 2019:
                if paper_type not in paper_type_map:
                    paper_type_map[paper_type] = next_paper_type_id
                    next_paper_type_id += 1
                node_colors[paper] = paper_type_map[paper_type]
            else:
                validation_node_labels[paper] = int(paper_type[0])

        edges.append((paper_A, paper_B))

    # Author-Institution Edges
    for i in range(0, edge_index_affiliated_with.shape[1]):
        author = edge_index_affiliated_with[0,i] + author_node_offset
        institution = edge_index_affiliated_with[1,i] + institution_node_offset

        edges.append((author, institution))

    directed = True
    print("  ...Constructing GraphView...")
    # New Way:
    # node_colors = Coloring(node_colors)
    # graph = GraphView(nodes, edges, directed, edge_types)
    # Old Way:
    ons = {n: set() for n in range(0, len(node_colors))}
    ins = {n: set() for n in range(0, len(node_colors))}
    for (a, b) in edges:
        ons[a].add(b)
        ins[b].add(a)
    graph = (ons, ins, directed, edge_types)

    return (graph, node_colors, validation_node_labels)

def get_max_score_for_node_classification(graph, node_colors, validation_node_labels):
    print("  Getting Automorphism Orbits...")
    # New Way:
    # (new_colors, _) = hopeful_canonicalizer(graph, node_colors, \
    #                                         return_canonical_order=False)
    # Old Way:
    (out_neighbor_sets, in_neighbor_sets, _, edge_types) = graph
    hopeful_canonicalizer(out_neighbor_sets, node_colors, edge_types=edge_types, \
                          in_neighbor_sets=in_neighbor_sets, return_canon_order=False, \
                          print_info=False, k_hop_graph_collections=None)
    new_colors = node_colors
    print("    ...Obtained Automorphism Orbits.")

    if new_colors.num_singletons() == graph.num_nodes():
        print("ALL NODES ARE SINGLETONS! MAX PERFORMANCE IS OPTIMAL!")
        return

    print("  Computing Max Accuracy Validation Score...")
    observed_cells = {}
    for node, label in validation_node_labels.items():
        node_color = new_colors[node]
        if node_color in observed_cells:
            continue

        cell = new_colors.get_cell(node_color)
        labels_in_cell = {}
        for n in cell:
            label = validation_node_labels[n]
            if label not in labels_in_cell:
                labels_in_cell[label] = 0
            labels_in_cell[label] += 1
        labels_in_cell = [count for label, count in labels_in_cell.items()]
        m = max(labels_in_cell)
        s = sum(labels_in_cell)

        observed_cells[node_color] = (m, s)

    M = 0
    S = 0
    for _, (m, s) in observed_cells.items():
        M += m
        S += s
    print("  Max Possible Validation Accuracy is: %f   (%d / %d)" % (float(M) / float(S), M, S))

if __name__ == "__main__":
    task = "Link Pred"  # "Link Pred", "Node Classification", and "Graph Classification"
    if task == "Link Pred":
        (graph, node_colors, hr, t_candidate, t_correct_idx) = link_pred_dataset()

    elif task == "Node Classification":
        print("Loading MAG Graph (Node Classification Graph)...")
        (graph, node_colors, validation_node_labels) = node_classifier_dataset()
        print("...Graph Loaded")
        print("Getting Max Possible Validation Score...")
        get_max_score_for_node_classification(graph, node_colors, validation_node_labels)
    elif task == "Graph Classification":
        pass
    else:
        raise ValueError("Error! Unknown task '%s'" % task)