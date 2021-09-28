from ogb.lsc import MAG240MDataset
from ogb.lsc import WikiKG90MDataset
from ogb.lsc import PCQM4MDataset
from Practical_Isomorphism_Alg.views import GraphView
from Practical_Isomorphism_Alg.coloring import Coloring
from Practical_Isomorphism_Alg.main_algorithm import hopeful_canonicalizer, canonical_representation

# Class Info Format:
#
# [(class_ID, class_size, positives_in_class)]

dataset_base = "/data/datasets/WAT"

graph_classification_root = dataset_base + "/PCQM4M"
link_pred_root = dataset_base + "/WikiKG90M"
node_classification_root = dataset_base + "/MAG240M"

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
        if int(paper_type[1]) != 2019:
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

    node_colors = Coloring(node_colors)
    directed = True
    print("  ...Constructing GraphView...")
    graph = GraphView(nodes, edges, directed, edge_types)

    return (graph, node_colors, validation_node_labels)

def get_max_score_for_node_classification(graph, node_colors, validation_node_labels):
    print("  Getting Automorphism Orbits...")
    (new_colors, _) = hopeful_canonicalizer(graph, node_colors, \
                                            return_canonical_order=False)
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
    print("Loading MAG Graph (Node Classification Graph)...")
    (graph, node_colors, validation_node_labels) = node_classifier_dataset()
    print("...Graph Loaded")
    print("Getting Max Possible Validation Score...")
    get_max_score_for_node_classification(graph, node_colors, validation_node_labels)
