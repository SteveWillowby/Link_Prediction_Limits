import networkx as nx

# Assumes the input graph is zero-indexed.
# Also assumes the coloring is sorted by node index.

# The edge_types is a dict mapping (source, target) to type label
# Note that the types can be different when pointing in the opposite direction.

# Updates coloring_list in place.

# BEWARE that if init_active_set is specified, the changes that would be implied by the
# regular WL algorithm might not spread through the entire graph (i.e. if the boundary
# at some iteration does not change).
# For example, if node x has a unique color, and init_active_set is set([x]), nothing
# will change.
# I recommend just making a color unique OR being sure you already ran WL without an init_active_set.
def WL(G, coloring_list, edge_types=None, init_active_set=None, return_comparable_output=False):

    nodes = [i for i in range(0, len(G.nodes()))]
    active = set(nodes)
    active_hashes = {}
    if init_active_set is not None:
        active = init_active_set
        active_hashes = {tuple(sorted(list(active))): 1}

    # non_active = set(nodes) - active
    neighbor_lists = [list(G.neighbors(n)) for n in nodes]

    node_to_color = coloring_list
    color_to_nodes = [set([]) for i in range(0, len(nodes))] # There will at most be |V| colors.
    next_color = 1
    for node in range(0, len(node_to_color)):
        color_to_nodes[node_to_color[node]].add(node)
        if node_to_color[node] >= next_color:
            next_color = node_to_color[node] + 1

    for i in range(0, next_color):
        if len(color_to_nodes[i]) == 0:
            print("ERRONEOUS INPUT! MISSING COLOR %d in coloring_list passed to WL()" % i)
    

    color_counts = [0 for i in range(0, len(nodes))]
    used_full_color = [False for i in range(0, len(nodes))]

    comparable_output = []
    the_round = 0

    while len(active) > 0:
        if return_comparable_output:
            the_round += 1

        previous_partition_sizes = {node: len(color_to_nodes[node_to_color[node]]) for node in active}

        new_colors = []
        for node in active:
            if edge_types is not None:
                new_colors.append(((node_to_color[node], sorted([(node_to_color[n], edge_types[(n, node)]) for n in neighbor_lists[node]])), node))
            else:
                new_colors.append(((node_to_color[node], sorted([node_to_color[n] for n in neighbor_lists[node]])), node))
            color_counts[node_to_color[node]] += 1
            if color_counts[node_to_color[node]] == len(color_to_nodes[node_to_color[node]]):
                used_full_color[node_to_color[node]] = True

        new_colors.sort()

        had_a_change = False
        last_was_a_pass = True

        old_color = new_colors[0][0][0]
        on_first_in_partition = True
        full_color = used_full_color[old_color]

        if not used_full_color[old_color]:
            node_to_color[new_colors[0][1]] = next_color
            color_to_nodes[old_color].remove(new_colors[0][1])
            color_to_nodes[next_color].add(new_colors[0][1])
            had_a_change = True
            last_was_a_pass = False
            if return_comparable_output:
                comparable_output.append([the_round, next_color, 1, new_colors[0][0][1]])

        used_full_color[old_color] = False
        color_counts[old_color] = 0

        for i in range(1, len(new_colors)):
            new_color_or_partition = False
            prev_old_color = old_color
            old_color = new_colors[i][0][0]
            node = new_colors[i][1]
            if prev_old_color != old_color:
                if not on_first_in_partition: # If the previous partition actually used up a next_color
                    next_color += 1
                new_color_or_partition = True
                full_color = used_full_color[old_color]
                used_full_color[old_color] = False
                color_counts[old_color] = 0
                on_first_in_partition = True
            elif new_colors[i][0][1] != new_colors[i - 1][0][1]:
                if not on_first_in_partition or not full_color: # If we've used up a next_color in this partition.
                    next_color += 1
                new_color_or_partition = True
                on_first_in_partition = False

            if new_color_or_partition and return_comparable_output:
                if on_first_in_partition and full_color:
                    comparable_output.append([the_round, old_color, 1, new_colors[i][0][1]])
                else:
                    comparable_output.append([the_round, next_color, 1, new_colors[i][0][1]])
            elif return_comparable_output and len(comparable_output) > 0:
                comparable_output[-1][2] += 1

            if on_first_in_partition and full_color:
                last_was_a_pass = True
            else:
                node_to_color[node] = next_color 
                color_to_nodes[old_color].remove(node)
                color_to_nodes[next_color].add(node)
                had_a_change = True
                last_was_a_pass = False

        if not had_a_change: # No changes! We're done!
            break

        if not last_was_a_pass:
            next_color += 1

        if next_color == len(color_to_nodes):
            break

        new_active = set([])
        for node in active:
            if len(color_to_nodes[node_to_color[node]]) != previous_partition_sizes[node]:
                for neighbor in neighbor_lists[node]:
                    if len(color_to_nodes[node_to_color[neighbor]]) > 1: # Don't add singletons.
                        new_active.add(neighbor)
        active = new_active

    if not return_comparable_output:
        return None

    comparable_output = []
    for node_set in color_to_nodes:
        if len(node_set) == 0:
            break
        a_node = node_set.pop()
        comparable_output.append((len(node_set) + 1, tuple(sorted([node_to_color[n] for n in neighbor_lists[a_node]]))))
    return tuple(comparable_output)

def __tuple_substitute(t, idx, element):
    l = list(t)
    l[idx] = element
    return tuple(l)

def zero_indexed_and_map(G):
    m = {}
    next_idx = 0
    if G.is_directed():
        G_new = nx.DiGraph()
    else:
        G_new = nx.Graph()
    for node in G.nodes():
        m[node] = next_idx
        G_new.add_node(next_idx)
        next_idx += 1
    for (a, b) in G.edges():
        G_new.add_edge(m[a], m[b])

    return (G_new, m)
