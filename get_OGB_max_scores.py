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

from basic_container_types import default_set, default_dict, Set, Dict, \
                                  set_default_set_type, set_default_dict_type
from coloring import Coloring
from list_containers import ListSet, ListDict
from main_algorithm import hopeful_canonicalizer, canonical_representation
from sampling import set_default_sample_set_type, SampleListSet, SampleSet
from views import GraphView

# Class Info Format:
#
# [(class_ID, class_size, positives_in_class)]

def orbits(graph, coloring):
    (coloring, _) = hopeful_canonicalizer(graph, coloring, \
                                          return_canonical_order=False)

    return coloring

def canonical_form(graph, coloring):
    init_coloring = Coloring(coloring)
    (_, canon_order) = hopeful_canonicalizer(graph, coloring, \
                                             return_canonical_order=True)
    return canonical_representation(graph, canon_order, init_coloring)

dataset_base = "/nfs/datasets/open_graph_benchmark_LSC_2021"

graph_classification_root = dataset_base + "/PCQM4M"
link_pred_root = dataset_base + "/WikiKG90M"
node_classification_root = dataset_base + "/MAG240M"

def link_pred_dataset():
    dataset = WikiKG90Mv2Dataset(root=link_pred_root)

    train_hrt = dataset.train_hrt
    (num_triples, _) = train_hrt.shape

    # Get edges and flatten edge types.
    print_flush("Loading edges...")
    edge_types = default_dict()
    for i in range(0, num_triples):
        edge = (train_hrt[i,0], train_hrt[i,2])
        if edge in edge_types:
            edge_types[edge].add(train_hrt[i,1])
        else:
            edge_types[edge] = default_set([train_hrt[i,1]])

    del train_hrt

    print_flush("Edges loaded. %d raw edges vs. %d flattened edges." % \
                    (num_triples, len(edge_types)))
    print_flush("Flattening edge types...")

    next_type_id = 0
    edge_type_combo_dict = default_dict()
    for edge, types_set in edge_types.items():
        types = tuple(sorted(list(types_set)))
        if types not in edge_type_combo_dict:
            edge_type_combo_dict[types] = next_type_id
            next_type_id += 1
    print_flush("  ... %d total flattened edge types." % len(edge_type_combo_dict))

    for edge in edge_types:
        edge_types[edge] = \
            edge_type_combo_dict[tuple(sorted(list(edge_types[edge])))]
    print_flush("  ...Edge Types Flattened")

    exit(0)

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
    graph = GraphView(directed=directed, nodes=nodes, edges=edges, \
                      edge_types=edge_types)
    print_flush("    ...Graph Constructed")
    print_flush("Making Base Coloring...")
    node_coloring = Coloring([0 for _ in nodes])
    print_flush("    ...Made Base Coloring")

    return (graph, node_coloring, hr, t)

def get_max_score_for_link_pred(graph, node_coloring, hr, t):
    print_flush("Getting ORBITS...")
    base_orbits = orbits(graph, node_coloring)
    print_flush("  ...obtained orbits.")

    (num_predictions, _) = hr.shape
    hr_classes = default_dict()
    
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
    edge_types = None  # Can be inferred from node types.

    PAPER_TYPE_BASE = 3  # 2 is preserved for validation nodes

    validation_node_labels = default_dict()

    print_flush("  Loading paper labels and years...")

    percent = 0
    paper_years = dataset.paper_year
    paper_labels = dataset.paper_label
    (year_min, year_max) = (int(paper_years.min()), int(paper_years.max()))
    print("Year  min/max: %d/%d" % (year_min, year_max))
    (label_min, label_max) = (int(np.nanmin(paper_labels)), int(np.nanmax(paper_labels)))

    num_papers = int(paper_years.shape[0])
    num_years = int(year_max - year_min) + 1
    num_labels = int(label_max - label_min) + 1 + 1  # The extra +1 is because of NaN

    paper_type_map = [[False for _ in range(0, num_labels)] for _ in range(0, num_years)]

    print("Label min/max: %d/%d" % (label_min, label_max))
    for i in range(0, num_papers):
        # if int((100 * i) / num_papers) > percent:
        #     percent = int((100 * i) / num_papers)
        #     print_flush("    ... %d percent done" % percent) 

        label = paper_labels[i]
        year = int(paper_years[i]) - year_min
        if np.isnan(label):
            label = num_labels - 1
        else:
            label = int(label) - label_min

        if year == (2019 - year_min) and not (label == num_labels - 1):
            # Already has color 2 which indicates it is a validation node.
            validation_node_labels[i] = label
        else:
            paper_type_map[year][label] = True
            node_colors[i] = PAPER_TYPE_BASE + year * num_years + label

    del paper_years
    del paper_labels

    next_label = int(PAPER_TYPE_BASE)
    relabel_map = default_dict()
    for i in range(0, PAPER_TYPE_BASE):
        relabel_map[i] = i

    for y_idx in range(0, num_years):
        for l_idx in range(0, num_labels):
            if not paper_type_map[y_idx][l_idx]:
                continue
            old_label = PAPER_TYPE_BASE + y_idx * num_years + l_idx
            relabel_map[old_label] = next_label
            next_label += 1

    print("    %d total labels." % len(relabel_map))

    for n in range(0, num_papers):
        old_color = node_colors[n]
        if old_color not in relabel_map:
            print("%d of %d" % (n, num_papers))
            print("Missing Colors for year, label: %d, %d" % \
                (int((old_color - PAPER_TYPE_BASE) / num_years), \
                 int((old_color - PAPER_TYPE_BASE) % num_years)))
            print("old_color: %d" % old_color)
            exit(0)
        node_colors[n] = relabel_map[old_color]

    del relabel_map

    print("    %d total papers." % num_papers)
    print("    %d validation papers." % len(validation_node_labels))
        

    neighbors_list = [default_set() for _ in node_colors]
    print_flush("  Getting author-paper edges...")

    # Author-Paper Edges
    percent = 0
    size = edge_index_writes.shape[1]
    paper_observed = [False for _ in range(0, dataset.num_papers)]
    for i in range(0, size):
        if int((100 * i) / size) > percent:
            percent = int((100 * i) / size)
            if percent % 10 == 0:
                print_flush("    ... %d percent done" % percent) 

        author = edge_index_writes[0,i] + author_node_offset
        paper = edge_index_writes[1,i]
        if not paper_observed[paper]:
            paper_observed[paper] = True

        neighbors_list[author].add(paper)

    del edge_index_writes

    print_flush("  Getting paper-paper edges...")
    # Paper-Paper Edges
    percent = 0
    size = edge_index_cites.shape[1]
    for i in range(0, size):
        if int((100 * i) / size) > percent:
            percent = int((100 * i) / size)
            if percent % 10 == 0:
                print_flush("    ... %d percent done" % percent) 

        paper_A = edge_index_cites[0,i]
        paper_B = edge_index_cites[1,i]

        for paper in [paper_A, paper_B]:
            if not paper_observed[paper]:
                paper_observed[paper] = True

        neighbors_list[paper_A].add(paper_B)

    del edge_index_cites

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
            if percent % 10 == 0:
                print_flush("    ... %d percent done" % percent) 

        author = edge_index_affiliated_with[0,i] + author_node_offset
        institution = edge_index_affiliated_with[1,i] + institution_node_offset

        neighbors_list[author].add(institution)

    del edge_index_affiliated_with

    directed = True
    print_flush("   ...Converting node_colors to type Coloring...")
    node_colors = Coloring(node_colors)
    print_flush("   ...Conversion complete")
    print_flush("  ...Constructing GraphView...")
    graph = GraphView(directed=directed, neighbors_list=neighbors_list, \
                      edge_types=edge_types)
    print_flush("  ...GraphView constructed.")

    return (graph, node_colors, validation_node_labels)

def get_max_score_for_node_classification(graph, node_colors, validation_node_labels):
    print_flush("  Getting Automorphism Orbits...")
    new_colors = orbits(graph, node_colors, iso_mode=ISO_MODE)
    print_flush("    ...Obtained Automorphism Orbits.")

    if new_colors.num_singletons() == graph.num_nodes():
        print_flush("ALL NODES ARE SINGLETONS! MAX PERFORMANCE IS OPTIMAL!")
        return

    print_flush("  Computing Max Accuracy Validation Score...")
    observed_cells = default_dict()
    for node, label in validation_node_labels.items():
        node_color = new_colors[node]
        if node_color in observed_cells:
            continue

        cell = new_colors.get_cell(node_color)
        labels_in_cell = default_dict()
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
    set_default_set_type(ListSet)
    set_default_dict_type(Dict)
    set_default_sample_set_type(SampleListSet)

    task = "Link Pred"  # "Link Pred", "Node Classification", and "Graph Classification"
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
