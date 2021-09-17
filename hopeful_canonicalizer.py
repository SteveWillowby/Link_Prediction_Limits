import copy

# NOTE!!!
#
# In this file, if a graph is undirected, `out_neighbor_sets` is simply used for
#   all neighbor relations.
# The `out` part only becomes relevant when you also include `in_neighbor_sets`.

# ALSO NOTE!!!
#
# All functions in this file assume the graph nodes are integers starting with 0

# Further NOTE:
# If the graph is undirected and edge_types is not None, edge_types should
#   include an entry for both directions (a, b), (b, a)






# One hop just gives a spike with a node's neighbors (normal WL).
# Two hop gives all the connections among the neighbors and the edges out to
#   secondary neighbors.
# Three hop gives all the connections amound the 1- & 2-hop nodes and the
#   connections from the 2-hop nodes to the 3-hop nodes.
# Etc.
#
# Updates coloring_list in place.
def k_hop_WL(out_neighbor_sets, coloring_list, k, edge_types=None, \
             in_neighbor_sets=None, init_active_set=None, k_hop_graphs=None):

    directed = in_neighbor_sets is not None

    if len(coloring_list) <= 2:
        return

    num_nodes = len(coloring_list)
    nodes = [i for i in range(0, num_nodes)]

    if k > 1 and k_hop_graphs is None:
        k_hop_graphs = {}

    active = set()
    if init_active_set is not None:
        active = init_active_set
    else:
        # Add all nodes which are not already singletons in the color partition.
        color_cells = {}
        for n in range(0, num_nodes):
            color = coloring_list[n]
            if color not in color_cells:
                color_cells[color] = set()
            color_cells[color].add(n)
        for color, cell in color_cells.items():
            if len(cell) > 1:
                active |= cell

    if len(active) == 0:
        return

    color_check = sorted(list(set(coloring_list)))
    assert color_check[0] == 0
    assert color_check[-1] == len(color_check) - 1

    while len(active) > 0:

        # new_colors will contain tuples of the following:
        #   (canonical_representation_of_k_hop_neighborhood_of_node_A, A)
        new_colors = []

        if k == 1:
            for node in active:
                if directed:
                    if edge_types is None:
                        new_colors.append(((coloring_list[node], \
                                            tuple(sorted([coloring_list[n] for n in out_neighbor_sets[node]])), \
                                            tuple(sorted([coloring_list[n] for n in in_neighbor_sets[node]]))), \
                                           node))
                    else:
                        new_colors.append(((coloring_list[node], \
                                            tuple(sorted([(coloring_list[n], edge_types[(node, n)]) \
                                                                    for n in out_neighbor_sets[node]])), \
                                            tuple(sorted([(coloring_list[n], edge_types[(n, node)]) \
                                                                    for n in in_neighbor_sets[node]]))), \
                                           node))
                else:
                    if edge_types is None:
                        new_colors.append(((coloring_list[node], \
                                            tuple(sorted([coloring_list[n] for n in out_neighbor_sets[node]]))), \
                                           node))
                    else:
                        new_colors.append(((coloring_list[node], \
                                            tuple(sorted([(coloring_list[n], edge_types[(node, n)]) \
                                                                    for n in out_neighbor_sets[node]])), \
                                            tuple(sorted([(coloring_list[n], edge_types[(n, node)]) \
                                                                    for n in in_neighbor_sets[node]]))), \
                                           node))
        else:
            # TODO: Add check to speed things up by ensuring that nodes which
            #   point to exactly the same nodes get the same orbit. (I.e. Only
            #   canonicalize for ONE of the nodes' k-hop graphs.)
            for node in active:
                if node in k_hop_graphs:
                    # Load k-hop graph.
                    (out_s, in_s, e_types, node_remap) = \
                        k_hop_graphs[node]
                else:
                    # Create k-hop graph.
                    (out_s, in_s, e_types, node_remap) = \
                        k_hop_graph(k, node, out_neighbor_sets, \
                                        edge_types=edge_types, \
                                        in_neighbor_sets=in_neighbor_sets)
                    k_hop_graphs[node] = \
                        (out_s, in_s, e_types, node_remap)

                # Update with latest coloring. Make sure to highlight `node`.
                latest_ext_coloring = [c for (i, c) in \
                                        sorted([(i, coloring_list[n]) for n, i in node_remap.items()])]
                node_already_singleton = True
                node_color = coloring_list[node]
                count = 0
                for c in latest_ext_coloring:
                    if c == node_color:
                        count += 1
                        if count > 1:
                            node_already_singleton = False
                            break

                latest_colors = sorted(list(set(latest_ext_coloring)))
                latest_colors = {latest_colors[i]: i for i in range(0, len(latest_colors))}
                latest_int_coloring = [latest_colors[latest_ext_coloring[n]] \
                                            for n in range(0, len(latest_ext_coloring))]
                if not node_already_singleton:
                    latest_int_coloring[node_remap[node]] = len(latest_colors)

                canonical_node_order = \
                    hopeful_canonicalizer(out_s, latest_int_coloring, edge_types=e_types, \
                                          in_neighbor_sets=in_s, \
                                          return_canon_order=True, print_info=False)
                directed = in_s is not None
                representation = node_order_to_representation(canonical_node_order, \
                                                              out_s, directed, \
                                                              external_colors=latest_ext_coloring, \
                                                              edge_types=e_types)
                # The representation includes the external colors. Thus,
                #   node_to_color[node] is canonically included.
                new_colors.append((representation, node))

        new_colors.sort()
        prev_color = None
        next_i = -1
        new_color_dict = {}
        for (new_color, n) in new_colors:
            if prev_color is None or new_color != prev_color:
                next_i += 1

            new_color_dict[n] = next_i
            prev_color = new_color

        shattered_nodes = shatter_coloring(coloring_list, new_color_dict)

        # The new active set should be all nodes within k hops of a shattered node.
        active = set()
        current_perimeter = shattered_nodes
        for i in range(0, k):
            new_perimeter = set()
            for node in current_perimeter:
                new_perimeter |= out_neighbor_sets[node]
                if in_neighbor_sets is not None:
                    new_perimeter |= in_neighbor_sets[node]
            current_perimeter = new_perimeter
            active |= new_perimeter

# Returns a canonical node order.
#
# Also, updates coloring_list in place to contain canonical node orbit ids.
#
# This function works recursively. First, it finds provably correct orbits.
#   Second, it checks to see if the orbits are all singletons. If so, done.
#   Otherwise, highlight a node from the lowest orbit and then call recursively.
#
# To save computation, the k_hop_graph_collections are passed on in the
#   recursive subcalls. Because colors only get further refined, not less
#   refined, this is not a problem.
def hopeful_canonicalizer(out_neighbor_sets, coloring_list, edge_types=None, \
                          in_neighbor_sets=None, return_canon_order=True, \
                          print_info=False, k_hop_graph_collections=None):
    if k_hop_graph_collections is None:
        k_hop_graph_collections = []

    k = 1
    while True:
        if k > len(coloring_list):
            print("Alert!!! k = %d, len(coloring_list) == %d" % (k, len(coloring_list)))
            print(out_neighbor_sets)
            print(coloring_list)
        assert k <= len(coloring_list)

        # Update coloring.
        if len(k_hop_graph_collections) < k:
            k_hop_graph_collections.append({})
        k_hop_graphs = k_hop_graph_collections[k - 1]

        k_hop_WL(out_neighbor_sets, coloring_list, k, edge_types=edge_types, \
                 in_neighbor_sets=in_neighbor_sets, k_hop_graphs=k_hop_graphs)
        if print_info:
            print("Did the main k-hop WL - Obtained %d classes." % len(set(coloring_list)))

        # The updated coloring is a candidate for the automorphism partitioning.
        candidate_colors = sorted(list(set(coloring_list)))
        assert candidate_colors[0] == 0
        assert candidate_colors[-1] == len(candidate_colors) - 1
        candidate_cells = [set() for _ in candidate_colors]
        for n in range(0, len(coloring_list)):
            candidate_cells[coloring_list[n]].add(n)

        # If the partitioning is unique, no further refinement is needed.
        if len(candidate_cells) == len(coloring_list):
            canonical_ordering = [(coloring_list[n], n) \
                                  for n in range(0, len(coloring_list))]
            canonical_ordering.sort()
            return [n for (c, n) in canonical_ordering]

        # Attempt to prove that this partitioning is correct.
        found_inconsistency = False
        found_correctness = False
        node_to_proven_equality = {n: set([n]) for n in range(0, len(coloring_list))}
        while (not found_inconsistency) and (not found_correctness):
            # Attempt to find a non-trivial isomorphism proving nodes are part
            #   of the same cell. Repeat until all cells have been linked.
            coloring_for_iso_A = list(coloring_list)
            coloring_for_iso_B = list(coloring_list)
            k_hop_graphs_A = copy.deepcopy(k_hop_graphs)
            k_hop_graphs_B = copy.deepcopy(k_hop_graphs)
            found_iso = False

            singletons_A = set()
            non_singletons_A = set()
            for cell in candidate_cells:
                if len(cell) == 1:
                    singletons_A |= cell
                else:
                    non_singletons_A |= cell
            singletons_B = set(singletons_A)
            non_singletons_B = set(non_singletons_A)

            while (not found_iso) and (not found_inconsistency):
                next_color = len(set(coloring_for_iso_A))

                # Find a node pair that has not yet been matched, if possible.
                colors_observed_in_A = {}
                for n in non_singletons_A:
                    color_A = coloring_for_iso_A[n]
                    if color_A not in colors_observed_in_A:
                        colors_observed_in_A[color_A] = set()
                    colors_observed_in_A[color_A].add(n)
                new_match = None
                for n in non_singletons_B:
                    color_B = coloring_for_iso_B[n]
                    if color_B in colors_observed_in_A:
                        for node_A in colors_observed_in_A[color_B]:
                            if node_A not in node_to_proven_equality[n]:
                                new_match = (node_A, n)
                                break
                    if new_match is not None:
                        break

                if new_match is None:
                    # Find some color-allowed pair that has been matched before.
                    match = None
                    for n in non_singletons_B:
                        color_B = coloring_for_iso_B[n]
                        if color_B in colors_observed_in_A:
                            for node_A in colors_observed_in_A[color_B]:
                                match = (node_A, n)
                                break
                            break
                else:
                    match = new_match

                assert match is not None

                if match is not None:
                    # Highlight the nodes in their respective graphs and get an
                    #   updated k-hop coloring.
                    (node_A, node_B) = match
                    # Rather than giving the nodes a new color, we'll let the
                    #   active set take care of that.
                    active_A = set([node_A])
                    active_B = set([node_B])

                    # Coloring.
                    k_hop_WL(out_neighbor_sets, coloring_for_iso_A, k, edge_types=edge_types, \
                             in_neighbor_sets=in_neighbor_sets, init_active_set=active_A, \
                             k_hop_graphs=k_hop_graphs_A)
                    k_hop_WL(out_neighbor_sets, coloring_for_iso_B, k, edge_types=edge_types, \
                             in_neighbor_sets=in_neighbor_sets, init_active_set=active_B, \
                             k_hop_graphs=k_hop_graphs_B)

                    # Add more nodes to singletons.
                    new_partitions_A = {}
                    new_singletons_A = set()
                    for n in non_singletons_A:
                        new_color = coloring_for_iso_A[n]
                        if new_color not in new_partitions_A:
                            new_partitions_A[new_color] = set()
                        new_partitions_A[new_color].add(n)
                    for _, partition in new_partitions_A.items():
                        if len(partition) == 1:
                            new_singletons_A |= partition
                    new_partitions_B = {}
                    new_singletons_B = set()
                    for n in non_singletons_B:
                        new_color = coloring_for_iso_B[n]
                        if new_color not in new_partitions_B:
                            new_partitions_B[new_color] = set()
                        new_partitions_B[new_color].add(n)
                    for _, partition in new_partitions_B.items():
                        if len(partition) == 1:
                            new_singletons_B |= partition

                    singletons_A |= new_singletons_A
                    singletons_B |= new_singletons_B
                    non_singletons_A -= new_singletons_A
                    non_singletons_B -= new_singletons_B

                    # Check that the new singletons form isomorphic wiring.
                    #   If so, carry on, if not, move to a higher k.
                    singleton_node_order_A = [(n not in new_singletons_A, coloring_for_iso_A[n], n) for n in singletons_A]
                    singleton_node_order_A.sort()
                    singleton_node_order_A_dict = {singleton_node_order_A[i][2]: i \
                                                   for i in range(0, len(singleton_node_order_A))}
                    singleton_node_order_B = [(n not in new_singletons_B, coloring_for_iso_B[n], n) for n in singletons_B]
                    singleton_node_order_B.sort()
                    singleton_node_order_B_dict = {singleton_node_order_B[i][2]: i \
                                                   for i in range(0, len(singleton_node_order_B))}

                    for i in range(0, len(singleton_node_order_A)):
                        (tf_A, _, n_A) = singleton_node_order_A[i]
                        (tf_B, _, n_B) = singleton_node_order_B[i]
                        if tf_A != tf_B:
                            # if print_info:
                            #     print("Found inconsistency due to different numbers of new singletons.")
                            found_inconsistency = True
                            break
                        if tf_A:  # n_A not in new_singletons_A
                            # No more new singletons.
                            break

                        # Check edges.
                        sing_neighbors_of_n_A = out_neighbor_sets[n_A] & singletons_A
                        sing_neighbors_of_n_B = out_neighbor_sets[n_B] & singletons_B
                        if edge_types is None:
                            sing_neighbors_of_n_A = tuple(sorted([singleton_node_order_A_dict[n] \
                                                                    for n in sing_neighbors_of_n_A]))
                            sing_neighbors_of_n_B = tuple(sorted([singleton_node_order_B_dict[n] \
                                                                    for n in sing_neighbors_of_n_B]))
                        else:
                            sing_neighbors_of_n_A = tuple(sorted([(singleton_node_order_A_dict[n], edge_types[(n_A, n)]) \
                                                                    for n in sing_neighbors_of_n_A]))
                            sing_neighbors_of_n_B = tuple(sorted([(singleton_node_order_B_dict[n], edge_types[(n_B, n)]) \
                                                                    for n in sing_neighbors_of_n_B]))

                        if sing_neighbors_of_n_A != sing_neighbors_of_n_B:
                            # if print_info:
                            #     print("Found inconsistency due to different out edges.")
                            #     print(sing_neighbors_of_n_A)
                            #     print(sing_neighbors_of_n_B)
                            found_inconsistency = True
                            break

                        if in_neighbor_sets is not None:
                            sing_neighbors_of_n_A = in_neighbor_sets[n_A] & singletons_A
                            sing_neighbors_of_n_B = in_neighbor_sets[n_B] & singletons_B
                            if edge_types is None:
                                sing_neighbors_of_n_A = tuple(sorted([singleton_node_order_A_dict[n] \
                                                                        for n in sing_neighbors_of_n_A]))
                                sing_neighbors_of_n_B = tuple(sorted([singleton_node_order_B_dict[n] \
                                                                        for n in sing_neighbors_of_n_B]))
                            else:
                                sing_neighbors_of_n_A = tuple(sorted([(singleton_node_order_A_dict[n], edge_types[(n, n_A)]) \
                                                                        for n in sing_neighbors_of_n_A]))
                                sing_neighbors_of_n_B = tuple(sorted([(singleton_node_order_B_dict[n], edge_types[(n, n_B)]) \
                                                                        for n in sing_neighbors_of_n_B]))

                            if sing_neighbors_of_n_A != sing_neighbors_of_n_B:
                                # if print_info:
                                #     print("Found inconsistency due to different in edges.")
                                found_inconsistency = True
                                break

                    if (not found_inconsistency) and len(non_singletons_A) == 0:
                        found_iso = True

            if found_iso:
                # Update info with the paired nodes.
                node_order_A = [n for (_, n) in sorted([(coloring_for_iso_A[n], n) for n in range(0, len(coloring_list))])]
                node_order_B = [n for (_, n) in sorted([(coloring_for_iso_B[n], n) for n in range(0, len(coloring_list))])]
                for i in range(0, len(coloring_list)):
                    n_A = node_order_A[i]
                    n_B = node_order_B[i]
                    partial_orbit_A = node_to_proven_equality[n_A]
                    if n_B in partial_orbit_A:
                        continue
                    partial_orbit_B = node_to_proven_equality[n_B]
                    if len(partial_orbit_A) > len(partial_orbit_B):  # This `if` is just for time savings.
                        partial_orbit_A |= partial_orbit_B
                        for node in partial_orbit_B:
                            node_to_proven_equality[node] = partial_orbit_A
                    else:
                        partial_orbit_B |= partial_orbit_A
                        for node in partial_orbit_A:
                            node_to_proven_equality[node] = partial_orbit_B

                # Compute the number of partial orbits...
                partial_orbit_size_occurrences = {}
                for _, partial_orbit in node_to_proven_equality.items():
                    l = len(partial_orbit)
                    if l not in partial_orbit_size_occurrences:
                        partial_orbit_size_occurrences[l] = 0
                    partial_orbit_size_occurrences[l] += 1

                num_orbits = sum([count / l for l, count in partial_orbit_size_occurrences.items()])
                if print_info:
                    print("Found a new iso. Down to %d partial orbits." % num_orbits)
                if num_orbits == len(candidate_cells):
                    found_correctness = True

        # If correct, proceed to canonize. Otherwise, move to higher k.
        if found_correctness:
            if print_info:
                print("Found correct orbits with k = %d" % k)
            if not return_canon_order:
                return None

            if print_info:
                print("  Proceeding to further canonize...")

            # Highlight a single node in a multicell. Then recompute. 
            new_coloring_list = list(coloring_list)
            for cell in candidate_cells:
                if len(cell) > 1:
                    some_node = cell.pop()
                    new_coloring_list[some_node] = max(candidate_colors) + 1
                    return hopeful_canonicalizer(out_neighbor_sets, new_coloring_list, \
                                                 edge_types=edge_types, \
                                                 in_neighbor_sets=in_neighbor_sets, \
                                                 return_canon_order=True, \
                                                 print_info=print_info, \
                                                 k_hop_graph_collections=k_hop_graph_collections)

        if print_info:
            print("Did not finish with k = %d. Moving on to k = %d." % (k, k + 1))
        k += 1


# Returns a representation of the graph that can be hashed.
def node_order_to_representation(node_order, out_neighbor_sets, directed, \
                                 external_colors=None, \
                                 edge_types=None):
    # Get the Edges in a "Node: Neighbors-List" format.
    #   This format implicitly includes the number of nodes, because nodes
    #   can have empty neighbors lists.
    #
    # The only remaining info is the external colors, added at the end.
    node_to_idx = {node_order[i]: i for i in range(0, len(node_order))}
    edges = []
    if directed:
        if edge_types is None:
            for node in node_order:
                edges.append((node_to_idx[node], \
                              tuple(sorted([node_to_idx[n] for n in out_neighbor_sets[node]]))))
        else:
            for node in node_order:
                edges.append((node_to_idx[node], \
                              tuple(sorted([(node_to_idx[n], edge_types[(node, n)]) \
                                                for n in out_neighbor_sets[node]]))))
    else:
        for node in node_order:
            neighbors = []
            for n in out_neighbor_sets[node]:
                if node_to_idx[node] <= node_to_idx[n]:
                    neighbors.append(n)

            if edge_types is None:
                edges.append((node_to_idx[node], \
                              tuple(sorted([node_to_idx[n] for n in neighbors]))))
            else:
                edges.append((node_to_idx[node], \
                              tuple(sorted([(node_to_idx[n], edge_types[(node, n)]) \
                                                for n in neighbors]))))

    edges = tuple(edges)

    if external_colors is None:
        return edges
    else:
        return (tuple([external_colors[node_order[i]] for i in range(0, len(node_order))]), \
                edges)

# shatters coloring_list in place and returns a collection of the nodes in
#   shattered cells.
def shatter_coloring(coloring_list, partial_coloring_dict, color_to_nodes=None):
    new_col_types_by_old = {}
    compute_ctn = color_to_nodes is None
    if color_to_nodes is None:
        color_to_nodes = {}

    new_colors = []
    for n in range(0, len(coloring_list)):
        if n in partial_coloring_dict:
            new_colors.append(((coloring_list[n], partial_coloring_dict[n]), n))
        else:
            new_colors.append(((coloring_list[n], ), n))

        old_c = coloring_list[n]
        if old_c not in new_col_types_by_old:
            new_col_types_by_old[old_c] = set()
        new_col_types_by_old[old_c].add(new_colors[-1][0])

        if compute_ctn:
            if old_c not in color_to_nodes:
                color_to_nodes[old_c] = []
            color_to_nodes[old_c].append(n)

    new_colors.sort()

    prev_color = None
    next_i = -1
    for (new_color, n) in new_colors:
        if prev_color is None or new_color != prev_color:
            next_i += 1

        coloring_list[n] = next_i

        prev_color = new_color

    changed_nodes = []
    for old_color, nodes in color_to_nodes.items():
        if len(new_col_types_by_old[old_color]) > 1:
            changed_nodes += nodes

    return changed_nodes

# Include the node itself in the graph. This is crucial for directed graphs.
#
# Also, highlights the node.
def k_hop_graph(k, node, out_neighbor_sets, edge_types=None, \
                in_neighbor_sets=None):

    inner = set()
    new_inner = set()
    fringe = set([node])
    for i in range(0, k):
        new_inner = fringe
        inner |= new_inner
        fringe = set()

        for n in new_inner:
            fringe |= out_neighbor_sets[n]
            if in_neighbor_sets is not None:
                fringe |= in_neighbor_sets[n]
        fringe = fringe - inner

    all_nodes = inner | fringe

    # Zero-index the new graph.
    node_remap = list(all_nodes)  # The order does not matter.
    node_remap = {node_remap[i]: i for i in range(0, len(node_remap))}

    new_out_sets = {}
    for n in inner:
        new_out_sets[node_remap[n]] = \
            set([node_remap[p] for p in out_neighbor_sets[n] & all_nodes])
    for n in fringe:
        new_out_sets[node_remap[n]] = \
            set([node_remap[p] for p in out_neighbor_sets[n] & inner])

    if in_neighbor_sets is None:
        new_in_sets = None
    else:
        new_in_sets = {}
        for n in inner:
            new_in_sets[node_remap[n]] = \
                set([node_remap[p] for p in in_neighbor_sets[n] & all_nodes])
        for n in fringe:
            new_in_sets[node_remap[n]] = \
                set([node_remap[p] for p in in_neighbor_sets[n] & inner])

    if edge_types is None:
        new_edge_types = None
    else:
        new_edge_types = {}
        if in_neighbor_sets is None:
            for n in inner:
                for neighbor in out_neighbor_sets[n]:
                    new_edge_types[(node_remap[n], node_remap[neighbor])] = \
                        edge_types[(n, neighbor)]
                    new_edge_types[(node_remap[neighbor], node_remap[n])] = \
                        edge_types[(n, neighbor)]
        else:
            for n in inner:
                for neighbor in out_neighbor_sets[n]:
                    new_edge_types[(node_remap[n], node_remap[neighbor])] = \
                        edge_types[(n, neighbor)]
            for n in inner:
                for neighbor in in_neighbor_sets[n]:
                    new_edge_types[(node_remap[neighbor], node_remap[n])] = \
                        edge_types[(neighbor, n)]

    """
    external_coloring_list = [None for i in range(0, len(all_nodes))]
    for n in all_nodes:
        external_coloring_list[node_remap[n]] = coloring_list[n]

    internal_coloring_list = [0 for _ in range(0, len(all_nodes))]
    external_for_internal_dict = {i: external_coloring_list[i] for i in range(0, len(all_nodes))}
    node_highlight = max(external_coloring_list) + 1
    external_for_internal_dict[node_remap[node]] = node_highlight
    shatter_coloring(internal_coloring_list, external_for_internal_dict)
    """
            # internal_coloring_list, external_coloring_list, \

    return (new_out_sets, new_in_sets, \
            new_edge_types, node_remap)

if __name__ == "__main__":
    print("Canonical Form for a Simple Undirected Wedge.")
    o_s = {0: set([1, 2]), 1: set([0]), 2: set([0])}
    i_s = None
    coloring_list = [0, 0, 0]
    init_coloring_list = list(coloring_list)
    e_types = None
    directed = i_s is not None
    canon_order = hopeful_canonicalizer(o_s, coloring_list, edge_types=e_types, \
                          in_neighbor_sets=i_s, return_canon_order=True, \
                          print_info=True, k_hop_graph_collections=None)
    print(node_order_to_representation(canon_order, o_s, directed, \
                                       external_colors=init_coloring_list, \
                                       edge_types=e_types))
    print("Orbits of Original Node Labels: %s" % coloring_list)

    print("\n\nCanonical Form for an Undirected Square with Color.")
    o_s = {0: set([1, 3]), 1: set([0, 2]), 2: set([1, 3]), 3: set([2, 0])}
    i_s = None
    coloring_list = [0, 1, 0, 0]
    init_coloring_list = list(coloring_list)
    e_types = None
    directed = i_s is not None
    canon_order = hopeful_canonicalizer(o_s, coloring_list, edge_types=e_types, \
                          in_neighbor_sets=i_s, return_canon_order=True, \
                          print_info=True, k_hop_graph_collections=None)
    print(node_order_to_representation(canon_order, o_s, directed, \
                                       external_colors=init_coloring_list, \
                                       edge_types=e_types))
    print("Orbits of Original Node Labels: %s" % coloring_list)

    print("\n\nCanonical Form for a Sneaky Graph - should need k=2.")
    o_s = {0: set([1, 2, 4]), 1: set([0, 2, 3]), 2: set([0, 1, 3]), 3: set([1, 2, 7]), \
           4: set([0, 5, 6]), 5: set([4, 6, 7]), 6: set([4, 5, 7]), 7: set([5, 6, 3])}
    i_s = None
    coloring_list = [0 for _ in range(0, len(o_s))]
    init_coloring_list = list(coloring_list)
    e_types = None
    directed = i_s is not None
    canon_order = hopeful_canonicalizer(o_s, coloring_list, edge_types=e_types, \
                          in_neighbor_sets=i_s, return_canon_order=True, \
                          print_info=True, k_hop_graph_collections=None)
    print(node_order_to_representation(canon_order, o_s, directed, \
                                       external_colors=init_coloring_list, \
                                       edge_types=e_types))
    print("Orbits of Original Node Labels: %s" % coloring_list)
