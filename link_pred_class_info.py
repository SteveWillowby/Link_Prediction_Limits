from ram_friendly_NT_session import RAMFriendlyNTSession

# TODO: Update to allow for undirected edges and edges without types.
def get_class_info_for_target_triples(neighbors_collections, node_colors, triples):
    # A triple (a, t, b)'s class is also a triple (i, j, t).
    #   `i` is the index of the automorphism orbit of a.
    #   `j` is the index of the automorphism orbit of b given a is highlighted.
    #   `t` is the same.
    triple_classes = {}
    class_sizes = {}
    positives_in_class = {}

    num_nodes = len(neighbors_collections)

    print("  Overall ISO...")
    coloring_list = list(node_colors)
    session = RAMFriendlyNTSession(directed=True, \
                                   has_edge_types=True, \
                                   neighbors_collections=neighbors_collections, \
                                   kill_py_graph=True, \
                                   only_one_call=False)
    session.set_colors_by_coloring(node_colors)
    base_orbits = session.get_automorphism_orbits()
    session.run()
    base_orbits = base_orbits.get()
    print("  Done with Overall ISO.")

    base_orbit_colors = [0 for _ in range(0, num_nodes)]
    for i in range(0, len(base_orbits)):
        for n in base_orbits[i]:
            base_orbit_colors[n] = i

    main_cell_sizes = [len(o) for o in base_orbits]

    next_color = len(base_orbits)
    print("  Found %d base orbits for %d nodes" % (next_color, num_nodes))

    triple_classes = set()
    class_sizes = {}
    positives_in_class = {}

    percent_done = 0
    for i in range(0, len(triples)):
        percent = int((100 * i) / len(triples))
        if percent > percent_done:
            percent_done = percent
            print("%d percent done" % percent_done)

        (a, t, b) = triples[i]
        main_a_color = base_orbit_colors[a]
        main_b_color = base_orbit_colors[b]
        a_cell_size = main_cell_sizes[main_a_color]
        b_cell_size = main_cell_sizes[main_b_color]
        if a_cell_size == 1 or b_cell_size == 1:
            orbit = (main_a_color, main_b_color, t, True)
            orbit_size = a_cell_size * b_cell_size
        else:
            base_orbits[main_a_color].remove(a)
            base_orbits.append([a])
            session.set_colors_by_partitions(base_orbits)
            sub_orbits = session.get_automorphism_orbits()
            session.run()
            sub_orbits = sub_orbits.get()
            base_orbits[main_a_color].append(a)
            base_orbits.pop()

            sub_cell_sizes = [len(o) for o in sub_orbits]

            for j in range(0, len(sub_orbits)):
                if b in sub_orbits[j]:
                    sub_b_color = j
                    break

            b_sub_cell_size = sub_cell_sizes[sub_b_color]
            orbit = (main_a_color, sub_b_color, t, False)
            orbit_size = a_cell_size * b_sub_cell_size

        if orbit not in triple_classes:
            triple_classes.add(orbit)
            class_sizes[orbit] = orbit_size
            positives_in_class[orbit] = 0
        positives_in_class[orbit] += 1

    session.end_session()

    return [(tc, class_sizes[tc], positives_in_class[tc]) for tc in triple_classes]


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
def get_k_hop_info_classes_for_link_pred(neighbors_collections, orig_colors, \
                                         directed, \
                                         has_edge_types, \
                                         true_edges, k):
    assert type(orig_colors[0]) is int or type(orig_colors[0]) is list
    if type(orig_colors[0]) is list:
        orig_partitions = orig_colors
        next_color = len(orig_partitions)

        orig_colors = [0 for _ in range(0, num_nodes)]
        for i in range(0, orig_partitions):
            for n in orig_partitions[i]:
                orig_colors[n] = i
        del orig_partitions

    # While the _graph_ is flattened, `true_edges` might not be. Flatten it.
    self_loops_in_true_edges = False
    if has_edge_types:
        if directed:
            te = {}
            for (source, edge_type, target) in true_edges:
                edge = (source, target)
                self_loops_in_true_edges |= source == target
                if edge not in te:
                    te[edge] = {}
                if edge_type not in te[edge]:
                    te[edge][edge_type] = 0
                te[edge][edge_type] += 1
            true_edges = te
        else:
            te = {}
            for (source, edge_type, target) in true_edges:
                edge = (min(source, target), max(source, target))
                self_loops_in_true_edges |= source == target
                if edge not in te:
                    te[edge] = {}
                if edge_type not in te[edge]:
                    te[edge][edge_type] = 0
                te[edge][edge_type] += 1
            true_edges = te
    else:
        if directed:
            te = {}
            for (source, target) in true_edges:
                edge = (source, target)
                self_loops_in_true_edges |= source == target
                if edge not in te:
                    te[edge] = 0
                te[edge] += 1
            true_edges = te
        else:
            te = {}
            for (source, target) in true_edges:
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

    edge_classes = {}
    positives_in_edge_class = {}
    next_orig_color = max(orig_colors) + 1

    total_iterations = int((num_nodes * (num_nodes - 1)) / 2) + \
                       int(self_loops_in_true_edges) * num_nodes
    if directed:
        total_edges = num_nodes * (num_nodes - 1) + \
                      int(self_loops_in_true_edges) * num_nodes
    else:
        total_edges = total_iterations

    iteration = 0
    percent_done = 0
    for a in range(0, num_nodes):
        for b in range(a + int(not self_loops_in_true_edges), num_nodes):
            if int((iteration * 100) / total_iterations) > percent_done:
                percent_done = int((iteration * 100) / total_iterations)
                if percent_done < 5 or \
                        (percent_done <= 30 and percent_done % 5 == 0) or \
                        (percent_done <= 100 and percent_done % 10 == 0):
                    print("    %d percent done." % percent_done)
            iteration += 1

            k_hop_nodes = __k_hop_nodes__(neighbors, k, [a, b])
            (new_node_to_old, new_neighbors_collections, \
                observed_edge_types) = \
                    __induced_subgraph__(neighbors_collections, \
                                         k_hop_nodes, has_edge_types)

            old_a_color = orig_colors[a]
            old_b_color = orig_colors[b]
            if directed and a != b:
                ab_pairs = [(a, b), (b, a)]
            else:
                ab_pairs = [(a, b)]
            for (c, d) in ab_pairs:
                # The canonicalizing code does not require that all colors in
                #   orig_colors be in the range 0 - max_C
                if directed:
                    orig_colors[c] = next_orig_color
                    orig_colors[d] = next_orig_color + 1
                else:
                    orig_colors[c] = next_orig_color
                    orig_colors[d] = next_orig_color

                new_colors = __new_color_partitioning__(new_node_to_old, orig_colors)

                EC = __canon_rep__(new_node_to_old, new_neighbors_collections, \
                                   new_colors, orig_colors, \
                                   observed_edge_types, \
                                   directed, has_edge_types)

                orig_colors[a] = old_a_color
                orig_colors[b] = old_b_color

                if EC not in edge_classes:
                    edge_classes[EC] = 0
                edge_classes[EC] += 1

    print("#")
    print("#  %d Edge Classes for %d Total Edges" % \
                    (len(edge_classes), total_edges))
    print("#")
    # for (ec, count) in edge_classes.items():
    #     print("%s -- %d" % (ec, count))

    # TODO: Update (or more likely, remove) rest of stuff.
    return

    if type(orig_colors[0]) is list:
        orig_partitions = orig_colors
        next_color = len(orig_partitions)

        orig_colors = [0 for _ in range(0, num_nodes)]
        for i in range(0, orig_partitions):
            for n in orig_partitions[i]:
                orig_colors[n] = i
    else:
        next_color = max(orig_colors) + 1
        orig_partitions = [[] for _ in range(0, next_color)]
        for n in range(0, num_nodes):
            orig_partitions[orig_colors[n]].append(n)

    # Note: Make sure to use the _external_, _base_ colors when making a
    #   canonical representation of a subgraph.
    main_session = \
        RAMFriendlyNTSession(directed=directed, \
                             has_edge_types=has_edge_types, \
                             neighbors_collections=neighbors_collections, \
                             only_one_call=False, \
                             kill_py_graph=False, \
                             mode="Traces")
    main_session.set_colors_by_partitions(orig_partitions)

    base_orbits = main_session.get_automorphism_orbits()
    no = main_session.get_num_automorphisms()
    main_session.run()
    base_orbits = base_orbits.get()
    no = no.get()

    base_orbit_colors = [0 for _ in range(0, num_nodes)]
    for i in range(0, len(base_orbits)):
        for n in base_orbits[i]:
            base_orbit_colors[n] = i

    main_cell_sizes = [len(o) for o in base_orbits]

    print("  Found %d base orbits for %d nodes" % \
            (len(base_orbits), num_nodes))
    print("  There were a total of %s automorphisms." % no)

    # TODO: Since it seems most graphs are almost rigid, simplify the code
    #   below by looking at every single non-edge.
    avg_nontrivial_orbit_size = 0
    num_nontrivial_orbits = 0
    avg_degree_of_node_in_nt_orbit = 0
    num_nodes_in_nt_orbits = 0
    for o in base_orbits:
        if len(o) > 1:
            num_nodes_in_nt_orbits += len(o)
            num_nontrivial_orbits += 1
            avg_nontrivial_orbit_size += len(o)
            avg_degree_of_node_in_nt_orbit += len(o) * len(neighbors_collections[o[0]])
    
    if num_nodes_in_nt_orbits > 0:
        avg_degree_of_node_in_nt_orbit /= float(num_nodes_in_nt_orbits)
        avg_nontrivial_orbit_size /= float(num_nontrivial_orbits)
        print("Num non-trivial orbits: %d (average size of %f)" % \
                (num_nontrivial_orbits, avg_nontrivial_orbit_size))
        print("Num nodes in nt orbits: %d (average degree of %f)" % \
                (num_nodes_in_nt_orbits, avg_degree_of_node_in_nt_orbit))

    positives_in_edge_class = {}
    print("Proceeding to get edge classes for the true edges.")
    identifier = 0
    percent_done = 0
    for i in range(0, len(true_edges)):
        percent = int((100 * i) / len(true_edges))
        if percent > percent_done:
            percent_done = percent
            print("%d percent done" % percent_done)

        if has_edge_types:
            (a, identifier, b) = true_edges[i]
        else:
            (a, b) = true_edges[i]

        main_a_color = base_orbit_colors[a]
        main_b_color = base_orbit_colors[b]
        a_cell_size = main_cell_sizes[main_a_color]
        b_cell_size = main_cell_sizes[main_b_color]
        if a_cell_size == 1 or b_cell_size == 1:
            if directed:
                edge_class = (main_a_color, main_b_color, False)
            else:
                edge_class = (min(main_a_color, main_b_color), \
                              max(main_a_color, main_b_color), False)
        else:
            if directed:
                singleton_node = a
                other_node = b
                old_cell = main_a_color
            else:
                if main_a_color <= main_b_color:
                    singleton_node = a
                    other_node = b
                    old_cell = main_a_color
                else:
                    singleton_node = b
                    other_node = a
                    old_cell = main_b_color

            base_orbits[old_cell].remove(singleton_node)
            base_orbits.append([singleton_node])

            main_session.set_colors_by_partitions(base_orbits)
            sub_orbits = main_session.get_automorphism_orbits()
            main_session.run()
            sub_orbits = sub_orbits.get()
            base_orbits[old_cell].append(singleton_node)
            base_orbits.pop()

            for j in range(0, len(sub_orbits)):
                if other_node in sub_orbits[j]:
                    sub_node_color = j
                    break

            edge_class = (old_cell, sub_node_color, True)

        if edge_class not in positives_in_edge_class:
            positives_in_edge_class[edge_class] = {}
        if identifier not in positives_in_edge_class[edge_class]:
            positives_in_edge_class[edge_class][identifier] = 0
        positives_in_edge_class[edge_class][identifier] += 1


    print("Proceeding to get k_hop_classes for _all_ edge classes.")

    k_hop_edge_classes = set()
    class_sizes = {}
    positives_in_kh_class = {}

    # IMPORTANT NOTE: The subsequent node needs to assign edge classes in
    #   exactly the same way even though the above code loops through edges
    #   differently.
    #
    # The code above looks at all (x, y) pairs given in true edges.
    # The code below looks at a single representative x from EACH base orbit.
    #   It then looks at a representative y from each sub-orbit obtained after
    #   making x a singleton.

    t = None
    percent_done = 0
    for i in range(0, len(base_orbits)):
        percent = int((100 * i) / len(base_orbits))
        if percent > percent_done:
            percent_done = percent
            print("%d percent done with ALL edge types" % percent_done)

        a = min(base_orbits[i])
        main_a_color = i
        a_cell_size = main_cell_sizes[main_a_color]
        b_candidates = []
        if a_cell_size == 1:
            if directed:
                start = 0
            else:
                start = i + 1
            for j in range(start, len(base_orbits)):
                if j == i:
                    continue
                b = base_orbits[j][0]
                main_b_color = j
                b_cell_size = main_cell_sizes[main_b_color]

                # a's color denotes whether or not b's color is from a
                #   subcoloring
                edge_class = (main_a_color, main_b_color, False)

                b_candidates.append((b, b_cell_size, edge_class))
        else:
            base_orbits.append([a])
            base_orbits[main_a_color].remove(a)
            main_session.set_colors_by_partitions(base_orbits)
            base_orbits[main_a_color].append(a)
            base_orbits.pop()
            sub_orbits = main_session.get_automorphism_orbits()
            main_session.run()
            sub_orbits = sub_orbits.get()
            for j in range(0, len(sub_orbits)):
                b = min(sub_orbits[j])
                if b == a:
                    continue
                if directed or a < b:
                    if len(base_orbits[base_orbit_colors[b]]) == 1:
                        if directed:
                            edge_class = (main_a_color, base_orbit_colors[b], False)
                        else:
                            min_ = min(base_orbit_colors[b], main_a_color)
                            max_ = max(base_orbit_colors[b], main_a_color)
                            edge_class = (min_, max_, False)
                    else:
                        edge_class = (main_a_color, j, True)
                    b_candidates.append((b, a_cell_size * len(sub_orbits[j]), \
                                         edge_class))

        for (b, main_size, edge_class) in b_candidates:

            orig_a_color = orig_colors[a]
            orig_b_color = orig_colors[b]
            # orig_a_partition_size = len(orig_partitions[a])
            # orig_b_partition_size = len(orig_partitions[b])

            # The canonicalizing code does not require that all colors in
            #   orig_colors be in the range 0 - max_C
            if directed:
                orig_colors[a] = next_color
                orig_colors[b] = next_color + 1
            else:
                orig_colors[a] = next_color
                orig_colors[b] = next_color

            # Set k_hop_class via a canonicalization around (a, b).
            relevant_nodes = __k_hop_nodes__(neighbors, k, [a, b])

            (sorted_sub_nodes_list, sub_neighbors_collections, \
                sub_colors, observed_edge_types) = \
                    __induced_subgraph__(neighbors_collections, \
                                         orig_colors, \
                                         relevant_nodes, \
                                         has_edge_types)
            
            k_hop_class = __canon_rep__(sorted_sub_nodes_list, \
                                        sub_neighbors_collections, \
                                        sub_colors, orig_colors, \
                                        observed_edge_types, \
                                        directed, has_edge_types)

            orig_colors[a] = orig_a_color
            orig_colors[b] = orig_b_color

            if k_hop_class not in k_hop_edge_classes:
                k_hop_edge_classes.add(k_hop_class)
                class_sizes[k_hop_class] = 0

                # TODO: Figure out how to get positives in class via the code above.
                positives_in_kh_class[k_hop_class] = {}
            class_sizes[k_hop_class] += main_size

            if edge_class in positives_in_edge_class:
                for identifier, positives in \
                        positives_in_edge_class[edge_class].items():
                    if identifier not in positives_in_kh_class[k_hop_class]:
                        positives_in_kh_class[k_hop_class][identifier] = 0
                    positives_in_kh_class[k_hop_class][identifier] += positives

    main_session.end_session()

    class_info = []
    for khc in k_hop_edge_classes:
        for identifier, positives in positives_in_kh_class[khc].items():
            class_info.append(((khc, identifier), class_sizes[khc], positives))
    print("There were a total of %d edge and non-edge classes." % len(k_hop_edge_classes))
    return class_info
def OLD_get_k_hop_info_classes_for_link_pred(neighbors_collections, orig_colors, \
                                         directed, \
                                         has_edge_types, \
                                         true_edges, k):
    assert type(orig_colors[0]) is int or type(orig_colors[0]) is list

    if type(true_edges) is not list:
        true_edges = list(true_edges)

    num_nodes = len(neighbors_collections)

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

    if type(orig_colors[0]) is list:
        orig_partitions = orig_colors
        next_color = len(orig_partitions)

        orig_colors = [0 for _ in range(0, num_nodes)]
        for i in range(0, orig_partitions):
            for n in orig_partitions[i]:
                orig_colors[n] = i
    else:
        next_color = max(orig_colors) + 1
        orig_partitions = [[] for _ in range(0, next_color)]
        for n in range(0, num_nodes):
            orig_partitions[orig_colors[n]].append(n)

    # Note: Make sure to use the _external_, _base_ colors when making a
    #   canonical representation of a subgraph.
    main_session = \
        RAMFriendlyNTSession(directed=directed, \
                             has_edge_types=has_edge_types, \
                             neighbors_collections=neighbors_collections, \
                             only_one_call=False, \
                             kill_py_graph=False, \
                             mode="Traces")
    main_session.set_colors_by_partitions(orig_partitions)

    base_orbits = main_session.get_automorphism_orbits()
    no = main_session.get_num_automorphisms()
    main_session.run()
    base_orbits = base_orbits.get()
    no = no.get()

    base_orbit_colors = [0 for _ in range(0, num_nodes)]
    for i in range(0, len(base_orbits)):
        for n in base_orbits[i]:
            base_orbit_colors[n] = i

    main_cell_sizes = [len(o) for o in base_orbits]

    print("  Found %d base orbits for %d nodes" % \
            (len(base_orbits), num_nodes))
    print("  There were a total of %s automorphisms." % no)

    # TODO: Since it seems most graphs are almost rigid, simplify the code
    #   below by looking at every single non-edge.
    avg_nontrivial_orbit_size = 0
    num_nontrivial_orbits = 0
    avg_degree_of_node_in_nt_orbit = 0
    num_nodes_in_nt_orbits = 0
    for o in base_orbits:
        if len(o) > 1:
            num_nodes_in_nt_orbits += len(o)
            num_nontrivial_orbits += 1
            avg_nontrivial_orbit_size += len(o)
            avg_degree_of_node_in_nt_orbit += len(o) * len(neighbors_collections[o[0]])
    
    if num_nodes_in_nt_orbits > 0:
        avg_degree_of_node_in_nt_orbit /= float(num_nodes_in_nt_orbits)
        avg_nontrivial_orbit_size /= float(num_nontrivial_orbits)
        print("Num non-trivial orbits: %d (average size of %f)" % \
                (num_nontrivial_orbits, avg_nontrivial_orbit_size))
        print("Num nodes in nt orbits: %d (average degree of %f)" % \
                (num_nodes_in_nt_orbits, avg_degree_of_node_in_nt_orbit))

    positives_in_edge_class = {}
    print("Proceeding to get edge classes for the true edges.")
    identifier = 0
    percent_done = 0
    for i in range(0, len(true_edges)):
        percent = int((100 * i) / len(true_edges))
        if percent > percent_done:
            percent_done = percent
            print("%d percent done" % percent_done)

        if has_edge_types:
            (a, identifier, b) = true_edges[i]
        else:
            (a, b) = true_edges[i]

        main_a_color = base_orbit_colors[a]
        main_b_color = base_orbit_colors[b]
        a_cell_size = main_cell_sizes[main_a_color]
        b_cell_size = main_cell_sizes[main_b_color]
        if a_cell_size == 1 or b_cell_size == 1:
            if directed:
                edge_class = (main_a_color, main_b_color, False)
            else:
                edge_class = (min(main_a_color, main_b_color), \
                              max(main_a_color, main_b_color), False)
        else:
            if directed:
                singleton_node = a
                other_node = b
                old_cell = main_a_color
            else:
                if main_a_color <= main_b_color:
                    singleton_node = a
                    other_node = b
                    old_cell = main_a_color
                else:
                    singleton_node = b
                    other_node = a
                    old_cell = main_b_color

            base_orbits[old_cell].remove(singleton_node)
            base_orbits.append([singleton_node])

            main_session.set_colors_by_partitions(base_orbits)
            sub_orbits = main_session.get_automorphism_orbits()
            main_session.run()
            sub_orbits = sub_orbits.get()
            base_orbits[old_cell].append(singleton_node)
            base_orbits.pop()

            for j in range(0, len(sub_orbits)):
                if other_node in sub_orbits[j]:
                    sub_node_color = j
                    break

            edge_class = (old_cell, sub_node_color, True)

        if edge_class not in positives_in_edge_class:
            positives_in_edge_class[edge_class] = {}
        if identifier not in positives_in_edge_class[edge_class]:
            positives_in_edge_class[edge_class][identifier] = 0
        positives_in_edge_class[edge_class][identifier] += 1


    print("Proceeding to get k_hop_classes for _all_ edge classes.")

    k_hop_edge_classes = set()
    class_sizes = {}
    positives_in_kh_class = {}

    # IMPORTANT NOTE: The subsequent node needs to assign edge classes in
    #   exactly the same way even though the above code loops through edges
    #   differently.
    #
    # The code above looks at all (x, y) pairs given in true edges.
    # The code below looks at a single representative x from EACH base orbit.
    #   It then looks at a representative y from each sub-orbit obtained after
    #   making x a singleton.

    t = None
    percent_done = 0
    for i in range(0, len(base_orbits)):
        percent = int((100 * i) / len(base_orbits))
        if percent > percent_done:
            percent_done = percent
            print("%d percent done with ALL edge types" % percent_done)

        a = min(base_orbits[i])
        main_a_color = i
        a_cell_size = main_cell_sizes[main_a_color]
        b_candidates = []
        if a_cell_size == 1:
            if directed:
                start = 0
            else:
                start = i + 1
            for j in range(start, len(base_orbits)):
                if j == i:
                    continue
                b = base_orbits[j][0]
                main_b_color = j
                b_cell_size = main_cell_sizes[main_b_color]

                # a's color denotes whether or not b's color is from a
                #   subcoloring
                edge_class = (main_a_color, main_b_color, False)

                b_candidates.append((b, b_cell_size, edge_class))
        else:
            base_orbits.append([a])
            base_orbits[main_a_color].remove(a)
            main_session.set_colors_by_partitions(base_orbits)
            base_orbits[main_a_color].append(a)
            base_orbits.pop()
            sub_orbits = main_session.get_automorphism_orbits()
            main_session.run()
            sub_orbits = sub_orbits.get()
            for j in range(0, len(sub_orbits)):
                b = min(sub_orbits[j])
                if b == a:
                    continue
                if directed or a < b:
                    if len(base_orbits[base_orbit_colors[b]]) == 1:
                        if directed:
                            edge_class = (main_a_color, base_orbit_colors[b], False)
                        else:
                            min_ = min(base_orbit_colors[b], main_a_color)
                            max_ = max(base_orbit_colors[b], main_a_color)
                            edge_class = (min_, max_, False)
                    else:
                        edge_class = (main_a_color, j, True)
                    b_candidates.append((b, a_cell_size * len(sub_orbits[j]), \
                                         edge_class))

        for (b, main_size, edge_class) in b_candidates:

            orig_a_color = orig_colors[a]
            orig_b_color = orig_colors[b]
            # orig_a_partition_size = len(orig_partitions[a])
            # orig_b_partition_size = len(orig_partitions[b])

            # The canonicalizing code does not require that all colors in
            #   orig_colors be in the range 0 - max_C
            if directed:
                orig_colors[a] = next_color
                orig_colors[b] = next_color + 1
            else:
                orig_colors[a] = next_color
                orig_colors[b] = next_color

            # Set k_hop_class via a canonicalization around (a, b).
            relevant_nodes = __k_hop_nodes__(neighbors, k, [a, b])

            (sorted_sub_nodes_list, sub_neighbors_collections, \
                sub_colors, observed_edge_types) = \
                    __induced_subgraph__(neighbors_collections, \
                                         orig_colors, \
                                         relevant_nodes, \
                                         has_edge_types)
            
            k_hop_class = __canon_rep__(sorted_sub_nodes_list, \
                                        sub_neighbors_collections, \
                                        sub_colors, orig_colors, \
                                        observed_edge_types, \
                                        directed, has_edge_types)

            orig_colors[a] = orig_a_color
            orig_colors[b] = orig_b_color

            if k_hop_class not in k_hop_edge_classes:
                k_hop_edge_classes.add(k_hop_class)
                class_sizes[k_hop_class] = 0

                # TODO: Figure out how to get positives in class via the code above.
                positives_in_kh_class[k_hop_class] = {}
            class_sizes[k_hop_class] += main_size

            if edge_class in positives_in_edge_class:
                for identifier, positives in \
                        positives_in_edge_class[edge_class].items():
                    if identifier not in positives_in_kh_class[k_hop_class]:
                        positives_in_kh_class[k_hop_class][identifier] = 0
                    positives_in_kh_class[k_hop_class][identifier] += positives

    main_session.end_session()

    class_info = []
    for khc in k_hop_edge_classes:
        for identifier, positives in positives_in_kh_class[khc].items():
            class_info.append(((khc, identifier), class_sizes[khc], positives))
    print("There were a total of %d edge and non-edge classes." % len(k_hop_edge_classes))
    return class_info

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

    return (num_nodes, observed_edge_types, edge_list, old_colors_in_order)
