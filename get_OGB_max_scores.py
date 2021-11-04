import numpy as np
from ogb.lsc import MAG240MDataset
from ogb.lsc import WikiKG90Mv2Dataset
from ogb.lsc import PCQM4Mv2Dataset
import os
import sys

# Choose between the following two and update ISO_MODE() accordingly.
res = os.system("cp Practical_Isomorphism_Alg/*.py ./")
if res != 0:
    print(res)
    exit(0)

from coloring import Coloring
from views import GraphView
from main_algorithm import hopeful_canonicalizer, canonical_representation

exit(0)

def ISO_MODE():
    return "new"  # "old" or "new"

# Class Info Format:
#
# [(class_ID, class_size, positives_in_class)]

def orbits(graph, coloring):
    if ISO_MODE() == "old":
        (ons, ins, edge_types) = graph
        hopeful_canonicalizer(ons, coloring, edge_types=edge_types, \
                              in_neighbor_sets=ins, return_canon_order=False, \
                              print_info=False, k_hop_graph_collections=None)
    else:
        (coloring, _) = hopeful_canonicalizer(graph, coloring, \
                                              return_canonical_order=False)

    return coloring

def canonical_form(graph, coloring):
    if ISO_MODE() == "old":
        (ons, ins, edge_types) = graph
        init_coloring = list(coloring)
        canon_order = hopeful_canonicalizer(ons, coloring, \
                                            edge_types=edge_types, \
                                            in_neighbor_sets=ins, \
                                            return_canon_order=True)
        return node_order_to_representation(canon_order, ons, ins is not None, \
                                            external_colors=init_coloring, \
                                            edge_types=edge_types)
    else:
        init_coloring = Coloring(coloring)
        (_, canon_order) = hopeful_canonicalizer(graph, coloring, \
                                                 return_canonical_order=True)
        return canonical_representation(graph, canon_order, init_coloring)

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
    print_flush("Loading edges...")
    edges = list(set([(train_hrt[i,0], train_hrt[i,2]) for i in range(0, num_triples)]))
    print_flush("Edges loaded. %d raw edges vs. %d flattened edges." % \
                    (num_triples, len(edges)))
    print_flush("Initializing edge_types dict...")
    edge_types = {edge: set() for edge in edges}
    print_flush("Flattening edge types...")
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
    print_flush("  ... %d total flattened edge types." % len(edge_type_combo_dict))

    valid_task = dataset.valid_dict['h,r->t']
    hr = valid_task['hr']
    t = valid_task['t']
    (num_predictions, _) = hr.shape
    assert t.shape == (num_predictions,1)

    print_flush("Num nodes before adding validation nodes: %d" % len(nodes))
    for i in range(0, num_predictions):
        nodes.add(hr[i,0])
        nodes.add(t[i,0])

    assert max(nodes) == len(nodes) - 1
    assert min(nodes) == 0
    print_flush("  ... obtained %d nodes" % len(nodes))

    directed = True

    print_flush("Constructing graph...")
    # New Way:
    if ISO_MODE() == "new":
        graph = GraphView(nodes, edges, directed, edge_types=edge_types)
        node_coloring = Coloring([0 for _ in nodes])
    else:
        # Old Way:
        ons = {n: set() for n in nodes}
        ins = {n: set() for n in nodes}
        for (a, b) in edges:
            ons[a].add(b)
            ins[b].add(a)
        graph = (ons, ins, directed, edge_types)
        node_coloring = [0 for _ in nodes]
    print_flush(" ... graph constructed.")

    return (graph, node_coloring, hr, t)

def get_max_score_for_link_pred(graph, node_coloring, hr, t):
    print_flush("Getting ORBITS...")
    base_orbits = orbits(graph, node_coloring)
    print_flush("  ...obtained orbits.")
    if ISO_MODE() == "old":
        base_orbits = Coloring(base_orbits)

    (num_predictions, _) = hr.shape
    hr_classes = {}
    
    print_flush("    Getting classes raw info")
    for i in range(0, num_predictions):
        h = hr[i,0]
        r = hr[i,1]

        h_type = base_orbits[h]
        if (h_type, r) not in hr_classes:
            hr_classes[(h_type, r)] = []
        hr_classes[(h_type, r)].append((h, t))

    print_flush("Now to look at classes...")

    num_unique_tasks = 0
    multi_target_tasks = []
    messy_tasks = []
    for (h_type, r), tasks in hr_classes.items():
        if len(tasks) == 1 and \
                len(base_orbits.get_cell(base_orbits[tasks[0][1]])) == 1:
            num_unique_tasks += 1
            continue
        elif len(tasks) == 1:
            (h, t) = tasks[0]
            l = len(base_orbits.get_cell(base_orbits[h]))
            if l == 1:
                multi_target_tasks.append(len(base_orbits.get_cell(base_orbits[t])))
                continue
            print_flush("Running a sub-iso call.")
            sub_orbits = Coloring(base_orbits)
            sub_orbits.make_singleton(h)
            if ISO_MODE() == "old":
                sub_orbits = list(sub_orbits.__list__)
            sub_orbits = orbits(graph, sub_orbits)
            multi_target_tasks.append(len(sub_orbits.get_cell(sub_orbits[t])))
            print_flush("   ...finished the sub-iso call.")
        else:
            messy_tasks.append((h_type, r), tasks)
    del hr_classes

    if len(messy_tasks) == 0 and len(multi_target_tasks) == 0:
        print_flush("The Max Possible MRR is a Perfect 1 on the Validation Task.")
        return

    print_flush("Code is not yet ready to handle the rest of the situation.")

    if len(messy_tasks) == 0:
        sum_of_expected_RR = 0.0
        for v in multi_target_tasks:
            if v > 10:
                pass
            sum_of_expected_RR += 2.0 / v

def node_classifier_dataset():
    dataset = MAG240MDataset(root=node_classification_root)

    print_flush("  Dataset Info:")
    print_flush("  %d" % dataset.num_papers) # number of paper nodes
    print_flush("  %d" % dataset.num_authors) # number of author nodes
    print_flush("  %d" % dataset.num_institutions) # number of institution nodes
    print_flush("  %d" % dataset.num_paper_features) # dimensionality of paper features
    print_flush("  %d" % dataset.num_classes) # number of subject area classes

    edge_index_writes = dataset.edge_index('author', 'paper')
    edge_index_cites = dataset.edge_index('paper', 'paper')
    edge_index_affiliated_with = dataset.edge_index('author', 'institution')

    print_flush("  Loaded base data matrices.")
    print_flush("  Setting blank node colors...")

    # Combine all indices.
    author_node_offset = int(dataset.num_papers)
    institution_node_offset = int(dataset.num_papers + dataset.num_authors)

    node_colors = [2 for _ in range(0, int(dataset.num_papers))] + \
                  [1 for _ in range(0, int(dataset.num_authors))] + \
                  [0 for _ in range(0, int(dataset.num_institutions))]
    edges = []
    edge_types = None  # Can be inferred from node types.

    PAPER_TYPE_BASE = 3  # 2 is preserved for validation nodes

    validation_node_labels = {}

    print_flush("  Loading paper labels and years...")

    percent = 0
    paper_years = dataset.paper_year
    paper_labels = dataset.paper_label
    (year_min, year_max) = (int(paper_years.min()), int(paper_years.max()))
    print("Year  min/max: %d/%d" % (year_min, year_max))
    (label_min, label_max) = (int(np.nanmin(paper_labels)), int(np.nanmax(paper_labels)))

    num_years = int(year_max - year_min) + 1
    num_labels = int(label_max - label_min) + 1 + 1  # The extra +1 is because of NaN

    paper_type_map = [[False for _ in range(0, num_labels)] for _ in range(0, num_years)]

    print("Label min/max: %d/%d" % (label_min, label_max))
    size = paper_years.shape[0]
    for i in range(0, paper_years.shape[0]):
        if int((100 * i) / size) > percent:
            percent = int((100 * i) / size)
            print_flush("    ... %d percent done" % percent) 

        label = paper_labels[i]
        year = int(paper_years[i]) - year_min
        if np.isnan(label):
            label = num_labels - 1
        else:
            label = int(label) - label_min

        paper_type_map[year][label] = True
        node_colors[i] = PAPER_TYPE_BASE + year * num_years + label
        if year == 2019 and not label == num_labels - 1:
            validation_node_labels[i] = label

    next_label = int(PAPER_TYPE_BASE)
    relabel_map = {i: i for i in range(0, PAPER_TYPE_BASE)}
    for y_idx in range(0, num_years):
        for l_idx in range(0, num_labels):
            if not paper_type_map[y_idx][l_idx]:
                continue
            old_label = PAPER_TYPE_BASE + y_idx * num_years + l_idx
            relabel_map[old_label] = next_label
            next_label += 1

    print("    %d total labels." % len(relabel_map))

    for n in range(0, paper_years.shape[0]):
        old_color = node_colors[n]
        if old_color not in relabel_map:
            print("%d of %d" % (n, paper_years.shape[0]))
            print("Missing Colors for year, label: %d, %d" % \
                (int((old_color - PAPER_TYPE_BASE) / num_years), \
                 int((old_color - PAPER_TYPE_BASE) % num_years)))
            print("old_color: %d" % old_color)
            exit(0)
        node_colors[n] = relabel_map[old_color]

    print("    %d total papers." % paper_years.shape[0])
    print("    %d validation papers." % len(validation_node_labels))
        

    print_flush("  Getting author-paper edges...")

    # Author-Paper Edges
    percent = 0
    size = edge_index_writes.shape[1]
    paper_observed = [False for _ in range(0, dataset.num_papers)]
    for i in range(0, size):
        if int((100 * i) / size) > percent:
            percent = int((100 * i) / size)
            print_flush("    ... %d percent done" % percent) 

        author = edge_index_writes[0,i] + author_node_offset
        paper = edge_index_writes[1,i]
        if not paper_observed[paper]:
            paper_observed[paper] = True

        edges.append((author, paper))

    print_flush("  Getting paper-paper edges...")
    # Paper-Paper Edges
    percent = 0
    size = edge_index_cites.shape[1]
    for i in range(0, size):
        if int((100 * i) / size) > percent:
            percent = int((100 * i) / size)
            print_flush("    ... %d percent done" % percent) 

        paper_A = edge_index_cites[0,i]
        paper_B = edge_index_cites[1,i]

        for paper in [paper_A, paper_B]:
            if not paper_observed[paper]:
                paper_observed[paper] = True

        edges.append((paper_A, paper_B))

    print_flush("  There are a total of %d validation nodes." % \
                    len(validation_node_labels))

    if False in paper_observed:
        print_flush("  !!!! There is at least one paper with no edges !!!!")
    del paper_observed

    print_flush("  Getting author-institution edges...")
    # Author-Institution Edges
    percent = 0
    size = edge_index_affiliated_with.shape[1]
    for i in range(0, size):
        if int((100 * i) / size) > percent:
            percent = int((100 * i) / size)
            print_flush("    ... %d percent done" % percent) 

        author = edge_index_affiliated_with[0,i] + author_node_offset
        institution = edge_index_affiliated_with[1,i] + institution_node_offset

        edges.append((author, institution))

    directed = True
    print_flush("  ...Constructing GraphView...")
    # New Way:
    if ISO_MODE() == "new":
        node_colors = Coloring(node_colors)
        graph = GraphView(nodes, edges, directed, edge_types)
    else:
        # Old Way:
        ons = {n: set() for n in range(0, len(node_colors))}
        ins = {n: set() for n in range(0, len(node_colors))}
        for (a, b) in edges:
            ons[a].add(b)
            ins[b].add(a)
        graph = (ons, ins, edge_types)

    return (graph, node_colors, validation_node_labels)

def get_max_score_for_node_classification(graph, node_colors, validation_node_labels):
    print_flush("  Getting Automorphism Orbits...")
    new_colors = orbits(graph, node_colors, iso_mode=ISO_MODE)
    print_flush("    ...Obtained Automorphism Orbits.")

    if new_colors.num_singletons() == graph.num_nodes():
        print_flush("ALL NODES ARE SINGLETONS! MAX PERFORMANCE IS OPTIMAL!")
        return

    print_flush("  Computing Max Accuracy Validation Score...")
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
    print_flush("  Max Possible Validation Accuracy is: %f   (%d / %d)" % (float(M) / float(S), M, S))

def print_flush(s):
    print(s)
    sys.stdout.flush()

if __name__ == "__main__":
    task = "Node Classification"  # "Link Pred", "Node Classification", and "Graph Classification"
    if task == "Link Pred":
        (graph, node_colors, hr, t) = link_pred_dataset()
        print_flush("Graph Loaded!!!!!! Now to process...")
        exit()
        get_max_score_for_link_pred(graph, node_colors, hr, t)

    elif task == "Node Classification":
        print_flush("Loading MAG Graph (Node Classification Graph)...")
        (graph, node_colors, validation_node_labels) = node_classifier_dataset()
        print_flush("...Graph Loaded")
        print_flush("Getting Max Possible Validation Score...")
        get_max_score_for_node_classification(graph, node_colors, validation_node_labels)
    elif task == "Graph Classification":
        pass
    else:
        raise ValueError("Error! Unknown task '%s'" % task)
