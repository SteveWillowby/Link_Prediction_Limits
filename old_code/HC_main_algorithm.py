from HC_basic_container_types import default_set, default_dict
from HC_coloring import Coloring
from HC_print_debugging import turn_on_debug_printing, debug_print, turn_off_debug_printing, debug_printing_conditional_on, debug_printing_conditional_off
from HC_proven_equivalences import ProvenEquivalences
import random
from HC_relabeling import Relabeling
from HC_sampling import sample_proportionately
from HC_views import GraphView, KHopSubGraphView, obj_peak

# Returns a tuple (C, O) where C is a Coloring corresponding to the automorphism
#   orbits and O is a canonical ordering of the nodes. If
#   `return_canonical_order` is set to False, then the algorithm is faster and O
#   is None.
def hopeful_canonicalizer(graph_view, coloring, return_canonical_order=True, \
                          initial_seed=None):
    if (type(graph_view) is not GraphView) and \
            (type(graph_view) is not KHopSubGraphView):
        raise TypeError("Error! hopeful_canonicalizer() only works on " + \
                        "GraphView and KHopSubGraphView objects.")
    if type(coloring) is not Coloring:
        raise TypeError("Error! hopeful_canonicalizer() only takes " + \
                        "colorings of type Coloring.")

    coloring = Coloring(coloring)

    canon_order = __hopeful_canonicalizer__(graph_view, coloring, \
                                            return_canonical_order, \
                                            next_seed=initial_seed)
    return (coloring, canon_order)

def canonical_representation(graph_view, canonical_order, initial_coloring):
    return __canonical_representation__(graph_view, canonical_order, \
                                        initial_coloring)

################################################################################
########################      IMPLEMENTATION BELOW      ########################
################################################################################

def __hopeful_canonicalizer__(graph_view, coloring, return_canonical_order, \
                              k_hop_subgraphs=None, k_hop_subcanons=None, \
                              debug_indent=0, max_or_min_flag=True, \
                              next_seed=None):

    # if debug_indent == 0:
    #     debug_print("Canonicalizing the graph with neighbor dict:", debug_indent)
    #     debug_print([(n, sorted(list(graph_view.neighbors(n)))) for n in range(0, graph_view.num_nodes())], debug_indent)
    #     debug_print("  and coloring: %s" % str(coloring.__list__), debug_indent)

    if k_hop_subgraphs is None:
        k_hop_subgraphs = [None]
    if k_hop_subcanons is None:
        k_hop_subcanons = [None]

    num_nodes = graph_view.num_nodes()

    PE = ProvenEquivalences(num_nodes, debug_indent=debug_indent + 1)

    k = 0
    while True:
        if coloring.get_num_colors() == coloring.get_num_singletons():
            # We have a singleton coloring from the get-go.
            assert k == 0
            return coloring.get_singletons()

        if k == 0:
            k = 1
        else:
            k *= 2

        if k > 1 and debug_indent == 0:
            # debug_printing_conditional_on()
            debug_print("Attempting k = %d -- starting with color/node ratio of %f" % \
                            (k, float(coloring.get_num_colors()) / len(coloring)), debug_indent)
            # debug_printing_conditional_off()

        while len(k_hop_subgraphs) <= k:
            k_hop_subgraphs.append(default_dict())
        while len(k_hop_subcanons) <= k:
            k_hop_subcanons.append(default_dict())

        num_colors = coloring.get_num_cells()
        # TODO: Consider removing use of next_seed
        __k_hop_WL__(graph_view, coloring, k, \
                     k_hop_subgraphs[k], k_hop_subcanons[k], \
                     debug_indent=debug_indent+1, \
                     next_seed=next_seed)
        # if k > 1:
            # debug_printing_conditional_on()
            # debug_print("Got k-%d coloring." % k, debug_indent)
            # debug_printing_conditional_off()
            
        if coloring.get_num_cells() == num_colors and k > 1:
            # The check for k > 1 is essential to prevent a case where the
            #   correct orbits are passed in and thus the WL gets no updates.
            if k >= graph_view.num_nodes() * 4:
                print("BAD:")
                print("k = %d" % k)
                print([(n, sorted(list(graph_view.neighbors(n)))) for n in range(0, graph_view.num_nodes())])
                print("Coloring:")
                print({i: coloring[i] for i in range(0, len(coloring))})
                raise RuntimeError("Error! It appears that we already had the orbits but failed to prove them correct.")

        # To avoid using sub_k's, just set sub_k to k
        sub_k = k  # max(1, int(k / 2))
        while sub_k <= k:
            PE.set_proven_limits(coloring)
            if coloring.get_num_cells() == coloring.get_num_singletons():
                assert PE.done()
            while (not PE.done()) and not PE.failed():  # Go until equivalences are
                # proven or, until we fail.
                node_order = list(coloring.get_singletons())
                node_order_A = default_dict()
                for i in range(0, len(node_order)):
                    node_order_A[node_order[i]] = i
                node_order_B = default_dict(node_order_A)

                canon_nodes_A = default_set(node_order)
                canon_nodes_B = default_set(canon_nodes_A)

                (coloring_A, coloring_B) = PE.start_session()

                while PE.session_active():
                    (node_1, node_2) = PE.next_pairing()

                    sub_graphs_A = default_dict(k_hop_subgraphs[sub_k])
                    sub_graphs_B = default_dict(k_hop_subgraphs[sub_k])
                    sub_canons_A = default_dict()
                    sub_canons_B = default_dict()
                    for n, (c, no) in k_hop_subcanons[sub_k].items():
                        sub_canons_A[n] = (Coloring(c), no)
                        sub_canons_B[n] = (Coloring(c), no)

                    # Begin Certain Version (more likely to be correct)

                    # TODO: Consider removing use of next_seed
                    __k_hop_WL__(graph_view, coloring_A, sub_k, \
                                 sub_graphs_A, sub_canons_A, \
                                 init_active_set=default_set([node_1, node_2]), \
                                 init_active_set_order=default_dict({node_1: 1, node_2: 2}), \
                                 debug_indent=debug_indent + 1, \
                                 next_seed=next_seed)
                    __k_hop_WL__(graph_view, coloring_B, sub_k, \
                                 sub_graphs_B, sub_canons_B, \
                                 init_active_set=default_set([node_1, node_2]), \
                                 init_active_set_order=default_dict({node_1: 2, node_2: 1}), \
                                 debug_indent=debug_indent + 1, \
                                 next_seed=next_seed)
                    # End Certain Version

                    # Begin Uncertain Version (faster - less certain of correctness)
                    """
                    # TODO: Consider removing use of next_seed
                    __k_hop_WL__(graph_view, coloring_A, sub_k, \
                                 sub_graphs_A, sub_canons_A, \
                                 init_active_set=default_set([node_1]), \
                                 debug_indent=debug_indent + 1, \
                                 next_seed=next_seed)
                    __k_hop_WL__(graph_view, coloring_B, sub_k, \
                                 sub_graphs_B, sub_canons_B, \
                                 init_active_set=default_set([node_2]), \
                                 debug_indent=debug_indent + 1, \
                                 next_seed=next_seed)
                    # TODO: Implement mutual refinement so the copying is not needed
                    # TODO: Also, make the mutual refinement only use recent colors?
                    copy_of_A = Coloring(coloring_A)
                    coloring_A.refine_with(coloring_B)
                    coloring_B.refine_with(copy_of_A)
                    """
                    # End Uncertain Version

                    prev_num_canon_nodes = len(node_order_A)
                    if not PE.sub_colorings_are_consistent():
                        PE.end_session_in_failure()
                        debug_print("The coloring: failed -- diff num (or sizes) of cells in subcolorings.", debug_indent)
                        break

                    s_A = coloring_A.get_singletons()
                    s_B = coloring_B.get_singletons()
                    failed = False
                    for i in range(prev_num_canon_nodes, len(s_A)):
                        node_A = s_A[i]
                        node_B = s_B[i]
                        # Check that appending node_A to one "canonical" order and
                        #   node_B to another "canonical" order does not cause a
                        #   contradiction (i.e. all the edges line up.)
                        if graph_view.is_directed():
                            canon_out_A_neighbors = \
                                graph_view.out_neighbors(node_A) & canon_nodes_A
                            canon_out_B_neighbors = \
                                graph_view.out_neighbors(node_B) & canon_nodes_B
                            if graph_view.has_edge_types():
                                o_A = default_set([(node_order_A[n], \
                                            graph_view.edge_type(node_A, n)) \
                                                for n in canon_out_A_neighbors])
                                o_B = default_set([(node_order_B[n], \
                                            graph_view.edge_type(node_B, n)) \
                                                for n in canon_out_B_neighbors])
                            else:
                                o_A = default_set([node_order_A[n] \
                                                for n in canon_out_A_neighbors])
                                o_B = default_set([node_order_B[n] \
                                                for n in canon_out_B_neighbors])
                            if o_A != o_B:
                                debug_print("Failure due to edge lineup.", debug_indent)
                                failed = True
                                break

                            canon_in_A_neighbors = \
                                graph_view.in_neighbors(node_A) & canon_nodes_A
                            canon_in_B_neighbors = \
                                graph_view.in_neighbors(node_B) & canon_nodes_B
                            if graph_view.has_edge_types():
                                i_A = default_set([(node_order_A[n], \
                                            graph_view.edge_type(n, node_A)) \
                                                for n in canon_in_A_neighbors])
                                i_B = default_set([(node_order_B[n], \
                                            graph_view.edge_type(n, node_B)) \
                                                for n in canon_in_B_neighbors])
                            else:
                                i_A = default_set([node_order_A[n] \
                                                for n in canon_in_A_neighbors])
                                i_B = default_set([node_order_B[n] \
                                                for n in canon_in_B_neighbors])
                            if i_A != i_B:
                                debug_print("Failure due to edge lineup.", debug_indent)
                                failed = True
                                break
                        else:
                            canon_A_neighbors = \
                                graph_view.neighbors(node_A) & canon_nodes_A
                            canon_B_neighbors = \
                                graph_view.neighbors(node_B) & canon_nodes_B
                            if graph_view.has_edge_types():
                                n_A = default_set([(node_order_A[n], \
                                            graph_view.edge_type(n, node_A)) \
                                                for n in canon_A_neighbors])
                                n_B = default_set([(node_order_B[n], \
                                            graph_view.edge_type(n, node_B)) \
                                                for n in canon_B_neighbors])
                            else:
                                n_A = default_set([node_order_A[n] \
                                            for n in canon_A_neighbors])
                                n_B = default_set([node_order_B[n] \
                                            for n in canon_B_neighbors])
                            if n_A != n_B:
                                failed = True
                                debug_print("Failure due to edge lineup.", debug_indent)
                                # debug_print(coloring_A.get_singletons(), debug_indent)
                                # debug_print(coloring_B.get_singletons(), debug_indent)
                                # debug_print("%s vs %s for (%d, %d)" % (n_A, n_B, node_A, node_B), debug_indent)
                                break
                        canon_nodes_A.add(node_A)
                        canon_nodes_B.add(node_B)
                        node_order_A[node_A] = i
                        node_order_B[node_B] = i

                    if failed:
                        PE.end_session_in_failure()
                        # debug_print("The coloring: %s failed." % coloring.__list__, debug_indent)
                        break

                    # If we found an isomorphism, record all the pairings.
                    if len(canon_nodes_A) == num_nodes:
                        PE.end_session_in_success()

            if PE.done():
                break

            sub_k *= 2

        # if k > 1:
            # debug_printing_conditional_on()
            # if PE.done():
            #     debug_print("Limits proven.", debug_indent)
            # else:
            #     debug_print("Failed to prove.", debug_indent)
            # debug_printing_conditional_off()

        if PE.done():
            # We found the orbits.
            if not return_canonical_order:
                return None
            if coloring.get_num_singletons() == num_nodes:
                assert coloring.get_singletons() is not None
                return coloring.get_singletons()
            non_singleton_cells = coloring.get_non_singleton_colors()

            if next_seed is not None:
                # Deterministically choose a "random" cell.
                #   It is deterministic because the seed is fixed.
                
                # TODO: Make choice a bit more clever/efficient.
                #
                # Right now it chooses a cell with probability proportional to
                #   the cell's size.
                non_singleton_cells = sorted(list(non_singleton_cells))
                w = [len(coloring.get_cell(c)) for c in non_singleton_cells]
                random.seed(next_seed)
                chosen_cell = sample_proportionately(non_singleton_cells, w)
                next_seed = random.random()
            else:
                # Alternate between choosing the first largest cell and the first
                #   smallest cell.
                chosen_cell = None
                chosen_size = None
                for color in non_singleton_cells:
                    if chosen_cell is None:
                        chosen_cell = color
                        chosen_size = len(coloring.get_cell(chosen_cell))
                        continue
                    curr_size = len(coloring.get_cell(color))
                    if chosen_size == curr_size and color < chosen_cell:
                        chosen_cell = color
                        chosen_size = curr_size
                    elif (max_or_min_flag and chosen_size < curr_size) or \
                            ((not max_or_min_flag) and chosen_size > curr_size):
                        chosen_cell = color
                        chosen_size = curr_size

            node = obj_peak(coloring.get_cell(chosen_cell))
            new_coloring = Coloring(coloring)
            new_coloring.make_singleton(node)
            if new_coloring.get_num_singletons() == num_nodes:
                assert new_coloring.get_singletons() is not None
                return new_coloring.get_singletons()

            return __hopeful_canonicalizer__(graph_view, new_coloring, \
                                             return_canonical_order=return_canonical_order, \
                                             k_hop_subgraphs=k_hop_subgraphs, \
                                             k_hop_subcanons=k_hop_subcanons, \
                                             max_or_min_flag=(not max_or_min_flag), \
                                             debug_indent=debug_indent, \
                                             next_seed=next_seed)

# TODO: Speed up by working with the active set one cell at a time.
def __k_hop_WL__(graph_view, coloring, k, \
                 subgraph_dict, subgraph_canon_dict, init_active_set=None, \
                 init_active_set_order=None, debug_indent=0, \
                 next_seed=None):

    # If no active_set is given, choose the non-singleton nodes.
    if init_active_set is None:
        if init_active_set_order is not None:
            raise RuntimeError("Error! Can only pass init_active_set_order" + \
                    " when init_active_set is also passed.")
        active_set = default_set()
        for color in coloring.get_non_singleton_colors():
            for node in coloring.get_cell(color):
                active_set.add(node)
    else:
        active_set = init_active_set

    first_round = True
    while len(active_set) > 0:
        next_id = 0
        new_unordered_ids = []
        new_form_to_unordered_id = default_dict()

        if k > 1:
            singleton_fraction = \
                float(coloring.get_num_singletons()) / coloring.get_num_cells()
            SINGLETON_UPDATE_CONSTANT = 0.125

        for node in active_set:
            if k == 1:  # Basic WL
                if graph_view.is_directed():
                    out_neighbors = graph_view.out_neighbors(node)
                    in_neighbors = graph_view.in_neighbors(node)
                    if graph_view.has_edge_types():
                        form = tuple(sorted([(coloring[n], \
                                              graph_view.edge_type(node, n)) \
                                                    for n in out_neighbors]))
                    else:
                        form = tuple(sorted([coloring[n] \
                                               for n in out_neighbors]))
                else:
                    neighbors = graph_view.neighbors(node)
                    if graph_view.has_edge_types():
                        form = tuple(sorted([(coloring[n], \
                                              graph_view.edge_type(node, n)) \
                                                        for n in neighbors]))
                    else:
                        form = tuple(sorted([coloring[n] \
                                               for n in neighbors]))
            else:  # k > 1
                entirely_new = node not in subgraph_dict
                new = entirely_new
                if not entirely_new:
                    (subgraph, sing_frac) = subgraph_dict[node]

                    # Update periodically in case the new singletons decrease
                    #   the graph size.
                    if singleton_fraction >= \
                            sing_frac + SINGLETON_UPDATE_CONSTANT:
                        new = True

                if new:
                    new_subgraph = KHopSubGraphView(graph_view, [node], k, \
                                                    restricting_coloring=coloring)
                    if not entirely_new:
                        # If the new graph is the same, ignore the "change" and
                        #   KEEP the old coloring info.
                        if subgraph.num_nodes() == new_subgraph.num_nodes():
                            new_subgraph = subgraph
                            new = False
                    subgraph = new_subgraph
                    subgraph_dict[node] = (subgraph, singleton_fraction)

                new_coloring = new or (node not in subgraph_canon_dict)                        

                s_r = subgraph.get_node_relabeling()
                outer_coloring = Coloring([coloring[s_r.new_to_old(i)] \
                                            for i in range(0, len(s_r))], s_r)

                already_canonized = False
                if new_coloring:
                    inner_coloring = outer_coloring
                    inner_coloring.make_singleton(s_r.old_to_new(node))
                else:
                    (inner_coloring, inner_order) = subgraph_canon_dict[node]
                    old_len = inner_coloring.get_num_colors()
                    inner_coloring.refine_with(outer_coloring)
                    if inner_coloring.get_num_colors() == old_len:
                        already_canonized = True

                if not already_canonized:
                    canon_order = __hopeful_canonicalizer__(subgraph, inner_coloring, \
                                                            return_canonical_order=True, \
                                                            k_hop_subgraphs=None, \
                                                            debug_indent=debug_indent + 1, \
                                                            next_seed=next_seed)
                else:
                    canon_order = inner_order

                subgraph_canon_dict[node] = (inner_coloring, canon_order)

                form = __canonical_representation__(subgraph, canon_order, \
                                                    coloring, \
                                                    special_node=node, \
                                                    col_and_sn_from_higher_space=True)

            if first_round and init_active_set_order is not None:
                form = (init_active_set_order[node], form)

            if form not in new_form_to_unordered_id:
                new_form_to_unordered_id[form] = next_id
                next_id += 1
            new_unordered_ids.append((node, new_form_to_unordered_id[form]))

        new_forms = sorted([f for f, _ in new_form_to_unordered_id.items()])
        old_id_to_new_id = default_dict()
        for i in range(0, len(new_forms)):
            f = new_forms[i]
            old_id = new_form_to_unordered_id[f]
            new_id = i
            old_id_to_new_id[old_id] = new_id

        # Construct a coloring of the active nodes.
        the_relabeling = Relabeling(Relabeling.SUB_COLLECTION_TYPE, \
                                    list(active_set))
        sub_coloring_list = [None for _ in active_set]
        for (node, unordered_id) in new_unordered_ids:
            sub_coloring_list[the_relabeling.old_to_new(node)] = \
                old_id_to_new_id[unordered_id]

        sub_coloring = Coloring(sub_coloring_list)
        affected_colors = \
            coloring.refine_with(sub_coloring, alt_relabeling=the_relabeling)

        # New Active Set
        active_set = default_set()
        border = default_set()
        for color in affected_colors:
            if len(coloring.get_cell(color)) > 1:
                for n in coloring.get_cell(color):
                    active_set.add(n)
                    border.add(n)
            else:
                for n in coloring.get_cell(color):
                    border.add(n)
        # 1 --> 0
        # 2 --> 0
        # 3 --> 1
        # 4 --> 1
        # 5 --> 2
        # 6 --> 2
        # 7 --> 3
        # 8 --> 3
        # Etc.
        for i in range(0, int((k + 1) / 2) - 1):
            new_border = default_set()
            for n in border:
                for neighbor in graph_view.neighbors(n) - active_set:
                    # Only add non-singletons to active set.
                    if len(coloring.get_cell(coloring[neighbor])) > 1:
                        new_border.add(neighbor)
                        active_set.add(neighbor)
                    else:
                        new_border.add(neighbor)
            border = new_border

        first_round = False

# `initial_coloring` and `special_node` are assumed to be from the higher
#   space IFF there is a higher space.
def __canonical_representation__(graph_view, canon_order, \
                                 initial_coloring, special_node=None, \
                                 col_and_sn_from_higher_space=False):
    node_to_order = [None for i in range(0, len(canon_order))]
    for i in range(0, len(canon_order)):
        node_to_order[canon_order[i]] = i

    if col_and_sn_from_higher_space:
        relabeling = graph_view.get_node_relabeling()

    l = []
    if special_node is not None:
        if col_and_sn_from_higher_space:
            l.append(node_to_order[relabeling.old_to_new(special_node)])
        else:
            l.append(node_to_order[special_node])

    l.append((graph_view.num_nodes(), graph_view.num_edges()))

    # Node colors.
    if col_and_sn_from_higher_space:
        l.append(tuple([initial_coloring[\
                                relabeling.new_to_old(node)] \
                            for node in canon_order]))
    else:
        l.append(tuple([initial_coloring[node] for node in canon_order]))

    # Edges with types.
    if not graph_view.is_directed():
        seen = default_set()  # Used to avoid duplicating undirected edges.
    edges = []
    for node in canon_order:
        if graph_view.is_directed():
            out_neighbors = graph_view.out_neighbors(node)
            if graph_view.has_edge_types():
                edges.append(tuple(sorted([(node_to_order[n], \
                                            graph_view.edge_type(node, n)) \
                                                for n in out_neighbors])))
            else:
                edges.append(tuple(sorted([node_to_order[n] \
                                               for n in out_neighbors])))
        else:
            seen.add(node)  # Putting this here allows self-loops.
            neighbors = graph_view.neighbors(node) & seen
            if graph_view.has_edge_types():
                edges.append(tuple(sorted([(node_to_order[n], \
                                            graph_view.edge_type(node, n)) \
                                                for n in neighbors])))
            else:
                edges.append(tuple(sorted([node_to_order[n] \
                                                for n in neighbors])))
    edges = tuple(edges)
    l.append(edges)
    return tuple(l)

if __name__ == "__main__":
    turn_on_debug_printing()

    from basic_container_types import set_default_set_type, set_default_dict_type, Set, Dict
    from sampling import set_default_sample_set_type, SampleListSet, SampleSet
    from list_containers import ListSet, ListDict
    set_default_set_type(ListSet)
    set_default_dict_type(Dict)
    set_default_sample_set_type(SampleSet)

    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    G = GraphView(directed=True, nodes=nodes, edges=edges)
    coloring = Coloring([0, 0, 0])
    (coloring, canon_order) = hopeful_canonicalizer(G, coloring, \
                                                    return_canonical_order=True)
    print("Canon Order: %s" % str(canon_order))
    print("Coloring: %s" % str(coloring.__list__))

    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 3), (3, 5)]
    G = GraphView(directed=False, nodes=nodes, edges=edges)
    coloring = Coloring([0, 1, 0, 0, 0, 0])
    __k_hop_WL__(G, coloring, 1, default_dict(), default_dict())
    print(coloring)
    __k_hop_WL__(G, coloring, 1, default_dict(), default_dict(), init_active_set=default_set([3]))
    print(coloring)

    nodes = [0, 1, 2, 3, 4, 5, 6, 7]
    edges = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 3), (3, 7), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7)]
    G = GraphView(directed=False, nodes=nodes, edges=edges)
    coloring = Coloring([0, 0, 0, 0, 0, 0, 0, 0])
    (coloring, canon_order) = hopeful_canonicalizer(G, coloring, \
                                                    return_canonical_order=True)
    print("Canon Order: %s" % str(canon_order))
    print("Coloring: %s" % str(coloring.__list__))
