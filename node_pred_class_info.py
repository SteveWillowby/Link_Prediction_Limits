from hashlib import blake2b
from multiprocessing import Pool
from py_NT_session import PyNTSession
from ram_friendly_NT_session import RAMFriendlyNTSession
import random
import sys
from threading import Thread

# Answers the questions: If you only look at the nodes within k hops of
#   a node, then what are the classes?
#
# It makes one call _per_ node.

__USE_RF_FOR_FULL_GRAPH__ = True

def get_k_hop_info_classes_for_node_pred(neighbors_collections, orig_colors, \
                                         directed, \
                                         has_edge_types, \
                                         true_entities, k, \
                                         fraction_of_entities=1.0, \
                                         base_seed=None, \
                                         num_processes=1, \
                                         num_threads_per_process=1, \
                                         use_py_iso=True, \
                                         hash_reps=False, \
                                         print_progress=False, \
                                         report_only_classes_with_positives=True):

    assert type(orig_colors[0]) is int or type(orig_colors[0]) is list
    node_percent = fraction_of_entities
    hidden_nodes = [n for (n, t) in true_entities]
    true_nodes = set()
    for (n, t) in true_entities:
        assert t == 0 or t == 1
        if t == 1:
            true_nodes.add(n)

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


    num_nodes = len(neighbors_collections)

    num_edges = sum([len(nc) for nc in neighbors_collections])
    if not directed:
        assert num_edges % 2 == 0
        num_edges = int(num_edges / 2)

    # Get sets of neighbors -- used to get k-hop subgraphs.
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
    graph = (neighbors_collections, neighbors)

    total_iterations = len(hidden_nodes)
    full_observed_nodes = total_iterations

    if use_py_iso and not __USE_RF_FOR_FULL_GRAPH__:
        session = PyNTSession(directed=directed, \
                              has_edge_types=has_edge_types, \
                              neighbors_collections=neighbors_collections, \
                              dir_augment=True)
    else:
        # Get the orbits for the base graph in case k = "inf" or in case all the
        #   nodes within k hops form the entire graph.

        session = RAMFriendlyNTSession(directed=directed, \
                                       has_edge_types=has_edge_types, \
                                       neighbors_collections=neighbors_collections, \
                                       kill_py_graph=False, \
                                       only_one_call=True, \
                                       mode="Traces")

    session.set_colors_by_partitions(orig_partitions)
    del orig_partitions
    orbit_partitions = session.get_automorphism_orbits()
    session.run()
    session.end_session()
    orbit_partitions = orbit_partitions.get()
    orbit_colors = [None for _ in range(0, num_nodes)]
    for i in range(0, len(orbit_partitions)):
        for n in orbit_partitions[i]:
            orbit_colors[n] = i

    ntpp = num_threads_per_process
    np = num_processes

    if num_processes > 1:

        args = [(i, k, graph, directed, has_edge_types, \
                 int((total_iterations + np * ntpp - 1) / (np * ntpp)), \
                 true_nodes, hidden_nodes, num_nodes, \
                 orig_colors, next_orig_color, \
                 orbit_colors, orbit_partitions, \
                 use_py_iso, hash_reps, \
                 num_processes, num_threads_per_process, {}, {}, \
                 print_progress, node_percent, base_seed) \
                        for i in range(0, num_processes)]
 
        process_pool = Pool(num_processes)
        result = process_pool.map(__parallel_proc_func__, args, \
                                    chunksize=1)
        if print_progress:
            print("Got result. Closing pool...")
            sys.stdout.flush()
        process_pool.close()
        if print_progress:
            print("Pool closed. Joining pool...")
            sys.stdout.flush()
        process_pool.join()
        if print_progress:
            print("Pool joined. Aggregating results...")
            sys.stdout.flush()
        (basic_node_classes, positives_in_node_class, observed_nodes) = \
            __parallel_aggregator__(result)
        if print_progress:
            print("Results aggregated.")
            sys.stdout.flush()

    else:
        assert num_processes > 0
        # num_processes = 1. Avoid copying data.
        arg = (0, k, graph, directed, has_edge_types, \
                 int((total_iterations + ntpp - 1) / ntpp), \
                 true_nodes, hidden_nodes, num_nodes, \
                 orig_colors, next_orig_color, \
                 orbit_colors, orbit_partitions, \
                 use_py_iso, hash_reps, \
                 1, num_threads_per_process, {}, {}, \
                 print_progress, node_percent, base_seed)

        (basic_node_classes, positives_in_node_class, observed_nodes) = \
            __parallel_proc_func__(arg)


    print("#")
    print("#  %d Node Classes for %d Observed Nodes" % \
                    (len(basic_node_classes), observed_nodes))
    print("#")
    # for (ec, count) in basic_node_classes.items():
    #     print("%s -- %d" % (ec, count))

    EC_with_positives = set()
    full_node_classes = []
    for label, c in positives_in_node_class.items():
        (EC, _) = label
        full_node_classes.append((basic_node_classes[EC], c))
        if not report_only_classes_with_positives:
            EC_with_positives.add(EC)

    if not report_only_classes_with_positives:
        for EC, c in basic_node_classes.items():
            if EC not in EC_with_positives:
                full_node_classes.append((c, 0))

    # `full_observed_nodes` is the number of nodes that could have been
    #   looked at (i.e. the number of coin flips)
    # `observed_nodes` is the number of non-edges actually looked at (i.e. the
    #   number of "heads" tossed)
    return (full_node_classes, full_observed_nodes, observed_nodes)

def __parallel_proc_func__(arg):
    (proc_idx, k, graph, directed, has_edge_types, \
     avg_num_tasks, true_nodes, hidden_nodes, num_nodes, \
     orig_colors, next_orig_color, \
     orbit_colors, orbit_partitions, \
     use_py_iso, hash_reps, \
     num_processes, num_threads_per_process, _, __, \
     print_progress, node_percent, base_seed) = arg

    if num_threads_per_process == 1:
        return __parallel_collection_function__(arg)

    args = [\
        (proc_idx + num_processes * i, k, \
         graph, directed, has_edge_types, \
         avg_num_tasks, true_nodes, hidden_nodes, num_nodes, \
         list(orig_colors), \
         next_orig_color, \
         list(orbit_colors), [list(o) for o in orbit_partitions], \
         use_py_iso, hash_reps, \
         num_processes, num_threads_per_process, \
         {}, {}, print_progress, node_percent, base_seed) \
            for i in range(0, num_threads_per_process)]

    result = [None for i in range(0, num_threads_per_process)]
    funcs = [(lambda i: (lambda: __thread_collection_overlay__(args, result, i)))(j) \
                for j in range(0, num_threads_per_process)]

    threads = [Thread(target=funcs[i]) \
                for i in range(0, num_threads_per_process)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return __parallel_aggregator__(result)

def __thread_collection_overlay__(arg_array, result_array, idx):
    result_array[idx] = __parallel_collection_function__(arg_array[idx])

def __parallel_collection_function__(arg):

    (proc_thread_idx, k, graph, directed, has_edge_types, \
     avg_num_tasks, true_nodes, hidden_nodes, num_nodes, \
     orig_colors, next_orig_color, \
     orbit_colors, orbit_partitions, \
     use_py_iso, hash_reps, \
     num_processes, num_threads_per_process, \
     basic_node_classes, positives_in_node_class, \
     print_progress, node_percent, base_seed) = arg

    if base_seed is not None:
        local_random = random.Random(base_seed + proc_thread_idx)
    else:
        local_random = random

    if hash_reps:
        HASH_BYTES = 64  # Can be anywhere between 1 and 64

    (neighbors_collections, neighbors) = graph
    if use_py_iso and not __USE_RF_FOR_FULL_GRAPH__:
        session = PyNTSession(directed=directed, \
                              has_edge_types=has_edge_types, \
                              neighbors_collections=neighbors_collections, \
                              dir_augment=True)
    else:
        session = RAMFriendlyNTSession(directed=directed, \
                                  has_edge_types=has_edge_types, \
                                  neighbors_collections=neighbors_collections, \
                                  kill_py_graph=False, \
                                  only_one_call=False, \
                                  tmp_file_augment="%d" % proc_thread_idx, \
                                  mode="Traces")

    parallelism = num_processes * num_threads_per_process

    observed_nodes = 0
    iteration = 0
    percent_done = 0
    for a in hidden_nodes:
        if a % parallelism != proc_thread_idx:
            continue

        # Only print progress if you are the first process.
        if proc_thread_idx == 0 and \
                int((iteration * 100) / avg_num_tasks) > percent_done:
            percent_done = int((iteration * 100) / avg_num_tasks)
            if percent_done < 5 or \
                    (percent_done <= 30 and percent_done % 5 == 0) or \
                    (percent_done <= 100 and percent_done % 10 == 0):
                if print_progress:
                    print("    Roughly %d percent done." % percent_done)
            if percent_done in [5, 30]:
                if print_progress:
                    print("    ...")
            sys.stdout.flush()
        iteration += 1

        if node_percent < 1.0 and local_random.random() >= node_percent:
            continue

        if k == "inf":
            NC = orbit_colors[a]
        else:
            # cfc is True if we are absolutely certain that the connected
            #   component(s) is/are maximal.
            #
            # i.e. cfc --> maximal component
            (k_hop_nodes, cfc) = __k_hop_nodes__(neighbors, k, [a])

            if (not cfc) and len(k_hop_nodes) < num_nodes:
                (new_node_to_old, new_neighbors_collections, \
                    observed_edge_types) = \
                        __induced_subgraph__(neighbors_collections, \
                                             k_hop_nodes, has_edge_types)

                old_a_color = orig_colors[a]
                orig_colors[a] = next_orig_color

                new_colors = \
                    __new_color_partitioning__(new_node_to_old, orig_colors)

                NC = __canon_rep__(new_node_to_old, new_neighbors_collections, \
                                   new_colors, orig_colors, old_a_color, \
                                   observed_edge_types, \
                                   directed, has_edge_types, \
                                   use_py_iso)

                orig_colors[a] = old_a_color
            else:
                NC = (orbit_colors[a], True)

        observed_nodes += 1

        if hash_reps:
            # h = blake2b(digest_size=HASH_BYTES)
            # h.update(bytes(str(NC), 'ascii'))
            # NC = __ALREADY_HASHED__(int.from_bytes(h.digest(), "big"))
            NC = __ALREADY_HASHED__(NC.__hash__())

        if NC not in basic_node_classes:
            basic_node_classes[NC] = 0
        basic_node_classes[NC] += 1

        if a in true_nodes:
            if NC not in positives_in_node_class:
                # The `None` is added as a reminder in case I choose to allow
                #   augmenting the info with extra info about the _type_ of
                #   label the classifier is predicting in this case.
                #
                # For instance, in link pred you might want to predict both
                #   "is a green edge here?" and "is a red edge here?"
                positives_in_node_class[(NC, None)] = 0
            positives_in_node_class[(NC, None)] += 1

    session.end_session()

    if print_progress:
        print("Process-thread idx %d finished collecting." % proc_thread_idx)
        sys.stdout.flush()
    return (basic_node_classes, positives_in_node_class, observed_nodes)

# NOTE: Destroys `results`
def __parallel_aggregator__(results):
    (basic_node_classes, positives_in_node_class, observed_nodes) = results[0]
    results[0] = None

    for i in range(1, len(results)):
        (proc_bec, proc_piec, on) = results[i]
        observed_nodes += on
        results[i] = None

        for label, c in proc_piec.items():
            if label not in positives_in_node_class:
                positives_in_node_class[label] = c
            else:
                positives_in_node_class[label] += c
        del proc_piec

        for NC, c in proc_bec.items():
            if NC not in basic_node_classes:
                basic_node_classes[NC] = c
            else:
                basic_node_classes[NC] += c
        del proc_bec
    return (basic_node_classes, positives_in_node_class, observed_nodes)


def __k_hop_nodes__(neighbors, k, init_nodes):
    certainly_full_component = False

    visited = set(init_nodes)
    frontier = set(init_nodes)
    for _ in range(0, k):
        new_frontier = set()
        for n in frontier:
            new_frontier |= neighbors[n] - visited
        if len(new_frontier) == 0:
            certainly_full_component = True
            break
        visited |= new_frontier
        frontier = new_frontier
    return (visited, certainly_full_component)

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

        if len(observed_edge_types) == 0 or \
                observed_edge_types[-1] == len(observed_edge_types) - 1:
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

class __ALREADY_HASHED__:

    def __init__(self, v):
        self.__v__ = v

    def __hash__(self):
        return self.__v__

    def __eq__(self, other):
        return (type(other) is __ALREADY_HASHED__) and self.__v__ == other.__v__

def __canon_rep__(new_node_to_old, g, new_colors, old_colors, orig_color, \
                  observed_edge_types, directed, has_edge_types, \
                  use_py_iso):
    num_nodes = len(g)

    if use_py_iso:
        session = PyNTSession(directed=directed, \
                              has_edge_types=has_edge_types, \
                              neighbors_collections=g, \
                              dir_augment=True)
    else:
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
        tuple([old_colors[new_node_to_old[n]] for n in node_order])

    return (True, orig_color, num_nodes, \
            observed_edge_types, edge_list, old_colors_in_order)

if __name__ == "__main__":
    NC_triangle = (True, (0, 0), ((0, 0, 0, 0), ((0, 1), (0, 2), (1, 2))), False)
    NC_triangle_copy = tuple(NC_triangle)
    NC_3_chain = (True, (0, 0), ((0, 0, 0, 0), ((0, 1), (1, 2), (2, 3))), False)

    b = blake2b(digest_size=64)
    b.update(bytes(str(NC_triangle), 'ascii'))
    print(b.hexdigest())
    b = blake2b(digest_size=64)
    b.update(bytes(str(NC_triangle_copy), 'ascii'))
    print(b.hexdigest())
    b = blake2b(digest_size=64)
    b.update(bytes(str(NC_3_chain), 'ascii'))
    print(b.hexdigest())

    d = {}
    d[__ALREADY_HASHED__(5)] = 5
    d[5] = hash(5)
    print(d)
