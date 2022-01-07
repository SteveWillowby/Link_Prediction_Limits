from multiprocessing import Pool
from ram_friendly_NT_session import RAMFriendlyNTSession
import sys
from threading import Thread

# Answers the questions: If you only look at the nodes within k hops of
#   (head, tail), then what are the classes?
#
# Warning! This can run up to num_nodes^2 times.
#   It can be slow and use LOTS of ram.
#
# It makes one call _per_ pair of nodes.
#
# NOTE: true_edges can be a container of (a, b) pairs or (a, t, b) triples
#   depending on whether or not the graph has edge types.
#
# NOTE: This version treats true edges of different kinds and/or different
#   multiplicity as different prediction tasks (separate classes).
def get_k_hop_info_classes_for_link_pred(neighbors_collections, orig_colors, \
                                         directed, \
                                         has_edge_types, \
                                         true_edges, k, \
                                         num_processes=1, \
                                         num_threads_per_process=1):

    assert type(orig_colors[0]) is int or type(orig_colors[0]) is list
    if type(orig_colors[0]) is list:
        orig_partitions = orig_colors
        next_orig_color = len(orig_partitions)

        orig_colors = [0 for _ in range(0, num_nodes)]
        for i in range(0, orig_partitions):
            for n in orig_partitions[i]:
                orig_colors[n] = i
    else:
        next_orig_color = max(orig_colors) + 1
        orig_partitions = [[] for _ in range(0, next_orig_color)]
        for n in range(0, len(orig_colors)):
            orig_partitions[orig_colors[n]].append(n)

    # While the _graph_ is flattened, `true_edges` might not be. Flatten it.
    self_loops_in_true_edges = False
    has_repeat_edges = False
    if has_edge_types:
        if directed:
            te = {}
            for (source, edge_type, target) in true_edges:
                if (not has_repeat_edges) \
                        and target in neighbors_collections[source]:
                    has_repeat_edges = True

                edge = (source, target)
                self_loops_in_true_edges |= source == target
                if edge not in te:
                    te[edge] = {}
                if edge_type not in te[edge]:
                    te[edge][edge_type] = 0
                te[edge][edge_type] += 1
            te = {edge: [(et, c) for et, c in d.items()] for edge, d in te.items()}
            true_edges = te
        else:
            te = {}
            for (source, edge_type, target) in true_edges:
                if (not has_repeat_edges) \
                        and target in neighbors_collections[source]:
                    has_repeat_edges = True

                edge = (min(source, target), max(source, target))
                self_loops_in_true_edges |= source == target
                if edge not in te:
                    te[edge] = {}
                if edge_type not in te[edge]:
                    te[edge][edge_type] = 0
                te[edge][edge_type] += 1
            te = {edge: [(et, c) for et, c in d.items()] for edge, d in te.items()}
            true_edges = te
    else:
        if directed:
            te = {}
            for (source, target) in true_edges:
                if (not has_repeat_edges) \
                        and target in neighbors_collections[source]:
                    has_repeat_edges = True

                edge = (source, target)
                self_loops_in_true_edges |= source == target
                if edge not in te:
                    te[edge] = 0
                te[edge] += 1
            true_edges = te
        else:
            te = {}
            for (source, target) in true_edges:
                if (not has_repeat_edges) \
                        and target in neighbors_collections[source]:
                    has_repeat_edges = True

                edge = (min(source, target), max(source, target))
                self_loops_in_true_edges |= source == target
                if edge not in te:
                    te[edge] = 0
                te[edge] += 1
            true_edges = te

    num_nodes = len(neighbors_collections)

    # Get sets of neighbors.
    if directed:
        if has_edge_types:
            neighbors = [set([n for n, _ in d.items()]) \
                            for d in neighbors_collections]
            for a in range(0, num_nodes):
                for b, _ in neighbors_collections[a].items():
                    neighbors[b].add(a)
        else:
            neighbors = [set(nc) for nc in neighbors_collections]
            for a in range(0, num_nodes):
                for b in neighbors_collections[a]:
                    neighbors[b].add(a)
    else:
        if has_edge_types:
            neighbors = [set([n for n, _ in d.items()]) \
                            for d in neighbors_collections]
        else:
            neighbors = [set(nc) for nc in neighbors_collections]

    total_iterations = int((num_nodes * (num_nodes - 1)) / 2) + \
                       int(self_loops_in_true_edges) * num_nodes
    if directed:
        total_edges = num_nodes * (num_nodes - 1) + \
                      int(self_loops_in_true_edges) * num_nodes
    else:
        total_edges = total_iterations

    # Get the orbits for the base graph in case k = "inf" or in case all the
    #   nodes within k hops form the entire graph.
    session = RAMFriendlyNTSession(directed=directed, \
                                   has_edge_types=has_edge_types, \
                                   neighbors_collections=neighbors_collections, \
                                   kill_py_graph=False, \
                                   only_one_call=True, \
                                   mode="Traces")
    session.set_colors_by_partitions(orig_partitions)
    orbit_partitions = session.get_automorphism_orbits()
    session.run()
    session.end_session()
    orbit_partitions = orbit_partitions.get()
    orbit_colors = [None for _ in range(0, num_nodes)]
    for i in range(0, len(orbit_partitions)):
        for n in orbit_partitions[i]:
            orbit_colors[n] = i

    if num_processes > 1:

        args = [(i, k, neighbors_collections, neighbors, directed, has_edge_types, \
                 int(total_iterations / (num_processes * num_threads_per_process)), \
                 true_edges, num_nodes, \
                 orig_colors, orig_partitions, next_orig_color, \
                 orbit_colors, orbit_partitions, \
                 self_loops_in_true_edges, has_repeat_edges, \
                 num_processes, num_threads_per_process, {}, {}) \
                        for i in range(0, num_processes)]
 
        process_pool = Pool(num_processes)
        result = process_pool.map(__parallel_proc_func__, args, \
                                    chunksize=1)
        process_pool.close()
        process_pool.join()
        (basic_edge_classes, positives_in_edge_class) = \
            __parallel_aggregator__(result)

    else:
        assert num_processes > 0
        # num_processes = 1. Avoid copying data.
        arg = (0, k, neighbors_collections, neighbors, directed, has_edge_types, \
                 int(total_iterations / num_threads_per_process), \
                 true_edges, num_nodes, \
                 orig_colors, orig_partitions, next_orig_color, \
                 orbit_colors, orbit_partitions, \
                 self_loops_in_true_edges, has_repeat_edges, \
                 1, num_threads_per_process, {}, {})

        (basic_edge_classes, positives_in_edge_class) = \
            __parallel_proc_func__(arg)


    print("#")
    print("#  %d Edge Classes for %d Total Edges" % \
                    (len(basic_edge_classes), total_edges))
    print("#")
    # for (ec, count) in basic_edge_classes.items():
    #     print("%s -- %d" % (ec, count))

    full_edge_classes = []
    next_int_label = 0
    for label, c in positives_in_edge_class.items():
        (EC, _) = label
        full_edge_classes.append((next_int_label, basic_edge_classes[EC], c))
        next_int_label += 1

    return full_edge_classes

def __parallel_proc_func__(arg):
    (proc_idx, k, neighbors_collections, neighbors, directed, has_edge_types, \
     avg_num_tasks, true_edges, num_nodes, \
     orig_colors, orig_partitions, next_orig_color, \
     orbit_colors, orbit_partitions, \
     self_loops_in_true_edges, has_repeat_edges, \
     num_processes, num_threads_per_process, _, __) = arg

    if num_threads_per_process == 1:
        return __parallel_collection_function__(arg)

    args = [\
     (proc_idx + num_processes * i, k, \
      neighbors_collections, neighbors, directed, has_edge_types, \
      avg_num_tasks, true_edges, num_nodes, \
      list(orig_colors), [list(o) for o in orig_partitions], \
      next_orig_color, \
      list(orbit_colors), [list(o) for o in orbit_partitions], \
      self_loops_in_true_edges, has_repeat_edges, \
      num_processes, num_threads_per_process, \
      {}, {}) \
        for i in range(0, num_threads_per_process)]

    # TODO: Figure out threads.
    threads = [Thread((__parallel_collection_function__, args[i])) \
                for i in range(0, num_threads_per_process)]
    for t in thread:
        t.start()
    for t in thread:
        t.join()

    result = [(args[i][-2], args[i][-1]) \
                for i in range(0, num_threads_per_process)]
    return __parallel_aggregator__(result)

def __parallel_collection_function__(arg):

    (proc_thread_idx, k, neighbors_collections, neighbors, directed, has_edge_types, \
     avg_num_tasks, true_edges, num_nodes, \
     orig_colors, orig_partitions, next_orig_color, \
     orbit_colors, orbit_partitions, \
     self_loops_in_true_edges, has_repeat_edges, \
     num_processes, num_threads_per_process, \
     basic_edge_classes, positives_in_edge_class) = arg

    parallelism = num_processes * num_threads_per_process

    session = RAMFriendlyNTSession(directed=directed, \
                                   has_edge_types=has_edge_types, \
                                   neighbors_collections=neighbors_collections, \
                                   kill_py_graph=False, \
                                   only_one_call=False, \
                                   tmp_file_augment="%d" % proc_thread_idx, \
                                   mode="Traces")

    iteration = 0
    percent_done = 0
    for a in range(0, num_nodes):
        if a % parallelism != proc_thread_idx:
            continue
        for b in range(a + int(not self_loops_in_true_edges), num_nodes):
            # Only print progress if you are the first process.
            if proc_thread_idx == 0 and \
                    int((iteration * 100) / avg_num_tasks) > percent_done:
                percent_done = int((iteration * 100) / avg_num_tasks)
                if percent_done < 5 or \
                        (percent_done <= 30 and percent_done % 5 == 0) or \
                        (percent_done <= 100 and percent_done % 10 == 0):
                    print("    Roughly %d percent done." % percent_done)
                if percent_done in [5, 30]:
                    print("    ...")
                sys.stdout.flush()
            iteration += 1

            if k == "inf":
                new_neighbors_collections = None
                observed_edge_types = None
                new_node_to_old = None
            else:
                k_hop_nodes = __k_hop_nodes__(neighbors, k, [a, b])
                if len(k_hop_nodes) < num_nodes:
                    (new_node_to_old, new_neighbors_collections, \
                        observed_edge_types) = \
                            __induced_subgraph__(neighbors_collections, \
                                                 k_hop_nodes, has_edge_types)
                else:
                    new_neighbors_collections = None
                    new_node_to_old = None
                    observed_edge_types = None

            old_a_color = orig_colors[a]
            old_b_color = orig_colors[b]
            if directed and a != b:
                ab_pairs = [(a, b), (b, a)]
            else:
                ab_pairs = [(a, b)]

            for (c, d) in ab_pairs:
                if d in neighbors_collections[c] and not has_repeat_edges:
                    continue

                if k == "inf" or len(k_hop_nodes) == num_nodes:
                    c_color = orbit_colors[c]
                    d_color = orbit_colors[d]
                    c_partition = orbit_partitions[c_color]
                    d_partition = orbit_partitions[d_color]

                    if directed:
                        if len(c_partition) == 1 or len(d_partition) == 1 or \
                                (len(c_partition) == 2 and d in c_partition):
                            EC = (False, c_color, d_color, False)
                        else:
                            orbit_partitions.append([c])
                            c_partition.remove(c)

                            session.set_colors_by_partitions(orbit_partitions)
                            suborbit_partitions = session.get_automorphism_orbits()
                            session.run()
                            suborbit_partitions = suborbit_partitions.get()

                            d_subcolor = None
                            for i in range(0, len(suborbit_partitions)):
                                if d in suborbit_partitions[i]:
                                    d_subcolor = i
                                    break
                            assert d_subcolor is not None

                            c_partition.append(c)
                            orbit_partitions.pop()
                            EC = (False, c_color, d_subcolor, True)
                    else:
                        if len(c_partition) == 1 or len(d_partition) == 1 or \
                                (len(c_partition) == 2 and d in c_partition):
                            EC = (False, min(c_color, d_color), \
                                         max(c_color, d_color), False)
                        else:
                            min_color = min(c_color, d_color)
                            if min_color == c_color:
                                highlighted_node = c
                                changed_partition = c_partition
                                alt_node = d
                            else:
                                highlighted_node = d
                                changed_partition = d_partition
                                alt_node = c

                            orbit_partitions.append([highlighted_node])
                            changed_partition.remove(highlighted_node)

                            session.set_colors_by_partitions(orbit_partitions)
                            suborbit_partitions = session.get_automorphism_orbits()
                            session.run()
                            suborbit_partitions = suborbit_partitions.get()

                            alt_subcolor = None
                            for i in range(0, len(suborbit_partitions)):
                                if alt_node in suborbit_partitions[i]:
                                    alt_subcolor = i
                                    break
                            assert alt_subcolor is not None

                            changed_partition.append(highlighted_node)
                            orbit_partitions.pop()
                            EC = (False, min_color, alt_subcolor, True)
                else:
                    # The canonicalizing code does not require that all colors in
                    #   orig_colors be in the range 0 - max_C
                    if directed:
                        orig_colors[c] = next_orig_color
                        orig_colors[d] = next_orig_color + 1
                    else:
                        orig_colors[c] = next_orig_color
                        orig_colors[d] = next_orig_color

                    new_colors = \
                        __new_color_partitioning__(new_node_to_old, orig_colors)

                    EC = __canon_rep__(new_node_to_old, new_neighbors_collections, \
                                       new_colors, orig_colors, \
                                       observed_edge_types, \
                                       directed, has_edge_types)

                orig_colors[a] = old_a_color
                orig_colors[b] = old_b_color

                if EC not in basic_edge_classes:
                    basic_edge_classes[EC] = 0
                basic_edge_classes[EC] += 1

                if (c, d) in true_edges:
                    labels = []
                    if has_edge_types:
                        for type_val in true_edges[(c, d)]:
                            labels.append((EC, type_val))
                    else:
                        labels.append((EC, true_edges[(c, d)]))

                    for label in labels:
                        if label not in positives_in_edge_class:
                            positives_in_edge_class[label] = 0
                        positives_in_edge_class[label] += 1

    session.end_session()
    return (basic_edge_classes, positives_in_edge_class)

# NOTE: Destroys `results`
def __parallel_aggregator__(results):
    (basic_edge_classes, positives_in_edge_class) = results[0]
    results[0] = None

    for i in range(1, len(results)):
        (proc_bec, proc_piec) = results[i]
        results[i] = None

        for label, c in proc_piec.items():
            if label not in positives_in_edge_class:
                positives_in_edge_class[label] = c
            else:
                positives_in_edge_class[label] += c
        del proc_piec

        for EC, c in proc_bec.items():
            if EC not in basic_edge_classes:
                basic_edge_classes[EC] = c
            else:
                basic_edge_classes[EC] += c
        del proc_bec
    return (basic_edge_classes, positives_in_edge_class)


def __k_hop_nodes__(neighbors, k, init_nodes):
    visited = set(init_nodes)
    frontier = set(init_nodes)
    for _ in range(0, k):
        new_frontier = set()
        for n in frontier:
            new_frontier |= neighbors[n] - visited
        visited |= new_frontier
        frontier = new_frontier
    return visited

def __induced_subgraph__(neighbors_collections, \
                         nodes, has_edge_types):
    num_nodes = len(nodes)
    nodes_list = sorted(list(nodes))
    nodes = {nodes_list[i]: i for i in range(0, num_nodes)}

    if has_edge_types:
        observed_edge_types = set()
        new_neighbors_collections = [{} for _ in range(0, num_nodes)]
        for a in range(0, num_nodes):
            old_a = nodes_list[a]
            for old_b, t in neighbors_collections[old_a].items():
                if old_b in nodes:
                    observed_edge_types.add(t)
                    new_neighbors_collections[a][nodes[old_b]] = t

        observed_edge_types = sorted(list(observed_edge_types))

        if observed_edge_types[-1] == len(observed_edge_types) - 1:
            # If there is effectively no relabeling, don't bother to relabel.
            return (nodes_list, new_neighbors_collections, None)

        observed_edge_types = tuple(observed_edge_types)

        et_relabeling = {observed_edge_types[i]: i \
                            for i in range(0, len(observed_edge_types))}
        for a in range(0, num_nodes):
            new_neighbors_collections[a] = {b: et_relabeling[t] for \
                            b, t in new_neighbors_collections[a].items()}
        del et_relabeling

        return (nodes_list, new_neighbors_collections, \
                observed_edge_types)
    else:
        new_neighbors_collections = [[] for _ in range(0, num_nodes)]
        for n in range(0, num_nodes):
            old_n = nodes_list[n]
            for old_neighbor in neighbors_collections[old_n]:
                if old_neighbor in nodes:
                    new_neighbors_collections[n].append(nodes[old_neighbor])
        return (nodes_list, new_neighbors_collections, None)

def __new_color_partitioning__(new_node_to_old, node_colors):
    # Get a partitioning corresponding to the new colors.
    num_nodes = len(new_node_to_old)
    new_colors = [node_colors[new_node_to_old[i]] for i in range(0, num_nodes)]
    new_colors = sorted(list(set(new_colors)))
    ncs = {new_colors[i]: i for i in range(0, len(new_colors))}
    new_colors = [[] for _ in range(0, len(new_colors))]
    for i in range(0, num_nodes):
        new_colors[ncs[node_colors[new_node_to_old[i]]]].append(i)
    del ncs
    return new_colors

class __IDENTITY_ACCESS__:

    def __init__(self):
        pass

    def __getitem__(i):
        return i

def __canon_rep__(new_node_to_old, g, new_colors, old_colors, \
                  observed_edge_types, directed, has_edge_types):
    num_nodes = len(g)

    session = RAMFriendlyNTSession(directed=directed, \
                                   has_edge_types=has_edge_types, \
                                   neighbors_collections=g, \
                                   only_one_call=True, \
                                   kill_py_graph=False, \
                                   mode="Traces")
    session.set_colors_by_partitions(new_colors)
    node_order = session.get_canonical_order()
    session.run()
    session.end_session()
    node_order = node_order.get()
    node_to_order = {node_order[i]: i for i in range(0, num_nodes)}

    if has_edge_types:
        edge_list = \
            tuple([tuple(sorted([(node_to_order[n2], t) for n2, t in
                    g[n].items()])) for n in node_order])
    else:
        edge_list = \
            tuple([tuple(sorted([node_to_order[n2] for n2 in \
                    g[n]])) for n in node_order])

    old_colors_in_order = \
        tuple([old_colors[new_node_to_old[n]] for n in range(0, len(node_order))])

    return (True, num_nodes, observed_edge_types, edge_list, old_colors_in_order)