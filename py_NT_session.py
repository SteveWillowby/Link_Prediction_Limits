import pynauty
import time

# This file contains the class PyNTSession
#
# NOTE: Requires that the nodes are numbered 0 through |V| - 1.
# NOTE: Requires that nodes are of type int, and no other type, not even
#   int-like types.
#
# The init function for PyNTSession takes the following arguments:
#   `directed` -- True or False
#
#   `has_edge_types` -- True or False
#
#   `neighbors_collections` -- should be a list of |V| collections (e.g. lists
#       or sets). Collection i should contain the neighbors of node i. If the
#       graph is directed, these should be out-neighbors only (i.e. successors).
#
#       If the graph has edge types, rather than being collections of integers
#       they should be one of the following:
#           A) collections of pairs where pair[0] is the neighbor and pair[1] is
#               the edge type. The collection should support the `in` operator
#               and iteration over the pairs.
#           B) dictionary-like collections where the keys are the neighbors and
#               the values are the edge types. The collection should support a
#               call to .items(), [] access, the `in` operator for keys, and
#               iteration over the keys (i.e. `for x in collection` should
#               iterate over keys).
#
#       Neighbors SHOULD NOT be repeated in a collection.
#
#   `kill_py_graph` -- True or False -- if True, destroys neighbors_collections
#       before running Nauty or Traces so that RAM usage is reduced.
#       defaults to False. In order for this to work, `neighbors_collections`
#       must support the pop() operation.
#
# INTERFACE:
#
# COLORING FUNCTIONS:
#
# O(color-map log color-map) = O(nodes log nodes)
#   color_map must support a .items() call.
# set_colors_by_coloring(color_map)
#
# O(1)
# blank_coloring()
#
# Will override coloring. Requires a list of lists.
# O(the input)
# set_colors_by_partitions(lists_of_nodes)
#
#
#
#
# RESULT FUNCTIONS: IMPORTANT: To actually get the results, you must first
#   call ALL the get_ functions you want and then call run().
#   After run() is called, nauty/traces will actually run. Once this
#   finishes, you can access the result values by calling .get()
#
#   After calling session_complete(), the session is done.
#
# Example:
#
#   session = PyNTSession(some_graph)
#   first_num_automorphisms = session.get_num_automorphisms()
#   first_runtime = session.get_runtime()
#   session.run()  # THIS LINE IS ESSENTIAL
#   session.set_colors_by_coloring(color_map)
#   second_num_automorphisms = session.get_num_automorphisms()
#   second_runtime = session.get_runtime()
#   session.run()  # THIS LINE IS ESSENTIAL
#   session.end_session()  # Acts a bit like a deconstructor
#
#   print(first_num_automorphisms.get())
#   print(first_runtime.get())
#   print(second_num_automorphisms.get())
#   print(second_runtime.get())
#
# NOTE: As of this comment the author of this code does not know if Nauty/Traces
#   always returns the orbits in a canonical order.
# get_automorphism_orbits()
#
# get_runtime()
#
# get_canonical_order()
#
# get_automorphisms() -- currently not implemented
#
# NOTE: Presently returns a _pair_ of numbers (a, b)
#           where the real number of automorphisms is approximately a * 10^b.
# get_num_automorphisms()
#
# run()
#
# end_session()

class PyNTSession:

    def __init__(self, directed, has_edge_types, \
                    neighbors_collections, \
                    kill_py_graph=False, \
                    dir_augment=True):

        self.__n__ = len(neighbors_collections)

        # Number of extra nodes used to express things like edge types and
        #   edge direction. Updated in __write_graph__()
        self.__extra_n__ = 0

        if self.__n__ == 0:
            raise ValueError("Error! The graph must have at least one node.")
        if self.__n__ > 2000000000:
            raise ValueError("Error! Nauty/Traces cannot handle graphs " + \
                             "with over 2000000000 nodes. This graph has " + \
                             "%d nodes." % self.__n__)

        self.__directed__ = directed
        self.__dir_augment__ = directed and dir_augment
        self.__neighbors_collections__ = neighbors_collections
        self.__kill_py_graph__ = kill_py_graph

        # Augment to add edge types.
        self.__et_augment__ = has_edge_types
        if self.__et_augment__:
            self.__dictlike_collections__ = False
            for n in range(0, self.__n__):
                neighbors = self.__neighbors_collections__[n]
                if len(neighbors) > 0:
                    for n2 in neighbors:
                        self.__dictlike_collections__ = type(n2) is int
                        break
                    break
        else:
            for n in range(0, self.__n__):
                neighbors = self.__neighbors_collections__[n]
                if len(neighbors) > 0:
                    for n2 in neighbors:
                        if type(n2) is not int:
                            raise ValueError("Error! Neighbors must be of " + \
                                             "type int, not %s." % type(n2))
                        break
                    break

        self.__additional_color_partitions__ = []
        self.__write_graph__()

        # The intro file is now written. It contains the graph.
        # All subsequent writes will be for colorings and output commands.

        # For a given graph state, store the result collection in this sub-list
        #   until the full result collection is present, so that results can be
        #   ordered in the way the program spits them out.
        # The format for each element is:
        # [result_type, nauty-traces-result object]
        self.__result_collection__ = []

        self.__collecting_results__ = False
        self.__session_ended__ = False
        self.__coloring_set__ = False


    # Result types are below:
    AUTOMORPHISMS = 0  # This one currently not implemented.
    NUM_AUTOMORPHISM_ORBITS = 1  # This one currently not implemented.
    NUM_AUTOMORPHISMS = 2
    RUNTIME = 3
    CANONICAL_ORDER = 4
    AUTOMORPHISM_ORBITS = 5

    # O(coloring * C) where C is the lookup time for an element in color_map
    #
    # Thus coloring can be a list or a dict, or some other such class.
    def set_colors_by_coloring(self, coloring):
        self.__doing_coloring__()

        if len(coloring) != self.__n__:
            raise ValueError(\
                   "Error! Different number of nodes in coloring and in graph.")

        distinct_colors = set()
        for n in range(0, len(coloring)):
            distinct_colors.add(coloring[n])
        distinct_colors = list(distinct_colors)
        distinct_colors.sort()

        partitions = [set() for _ in distinct_colors]
        if distinct_colors[0] != 0 or \
                distinct_colors[-1] != len(distinct_colors) - 1:
            color_remap = \
                {distinct_colors[i]: i for i in range(0, len(distinct_colors))}
            for n in range(0, len(coloring)):
                partitions[color_remap[coloring[n]]].add(n)
        else:
            for n in range(0, len(coloring)):
                partitions[coloring[n]].add(n)

        self.__write_partition_for_sets_of_nodes__(partitions)

    # O(1)
    def blank_coloring(self):
        self.set_colors_by_partitions([set(range(0, self.__n__))])

    # Will override coloring. Requires a list of containers -- ideally sets.
    # O(input)
    def set_colors_by_partitions(self, sets_of_nodes):

        self.__doing_coloring__()
        self.__write_partition_for_sets_of_nodes__(sets_of_nodes)

    def get_automorphism_orbits(self):
        return self.__get_result__(PyNTSession.AUTOMORPHISM_ORBITS)

    def get_runtime(self):
        return self.__get_result__(PyNTSession.RUNTIME)

    def get_canonical_order(self):
        return self.__get_result__(PyNTSession.CANONICAL_ORDER)

    def get_automorphisms(self):
        raise ValueError("Error! get_automorphisms() not yet implemented.\n" + \
               "  Did you want get_automorphism_orbits() by any chance?\n" + \
               "  Or maybe get_num_automorphisms() ?")
        return self.__get_result__(PyNTSession.AUTOMORPHISMS)

    def get_num_automorphisms(self):
        return self.__get_result__(PyNTSession.NUM_AUTOMORPHISMS)

    def run(self):

        if not self.__coloring_set__:
            self.blank_coloring()

        get_canon = self.__finish_result_collection__()

        # run Nauty/Traces
        print("Starting")
        start_t = time.time()
        orbit_info = pynauty.autgrp(self.__graph__)
        # Throw out the automorphisms group lists.
        v = orbit_info[0]
        orbit_info = (orbit_info[1],orbit_info[2],orbit_info[3],orbit_info[4])
        del v
        if get_canon:
            canon_order = pynauty.canon_label(self.__graph__)
        else:
            canon_order = None
        total_t = time.time() - start_t
        print("Finished")

        self.__populate_results__(total_t, orbit_info, canon_order)

    def end_session(self):
        self.__session_ended__ = True
        del self.__graph__

    ########################## Result Functions ##########################

    def __get_result__(self, result_type):
        if self.__session_ended__:
            raise RuntimeError("Error! Cannot get a new result " + \
                               "after session is ended.")

        if not self.__collecting_results__:
            self.__start_result_collection__()

        result = __PyNautyTracesResult__()
        self.__result_collection__.append([result_type, result])
        return result
        

    def __start_result_collection__(self):
        self.__collecting_results__ = True

        self.__result_collection__ = []

    def __finish_result_collection__(self):
        self.__collecting_results__ = False
        self.__coloring_set__ = False

        for l in self.__result_collection__:
            if l[0] == PyNTSession.CANONICAL_ORDER:
                return True
        return False

    def __doing_coloring__(self):
        if self.__session_ended__:
            raise RuntimeError("Error! Cannot perform coloring on the graph " +\
                               "after session is ended.")
        if self.__coloring_set__:
            raise RuntimeError("Error! Must wait until run() is called to " + \
                               "perform another coloring. Note that " + \
                               "blank_coloring() counts.")

        if not self.__collecting_results__:
            self.__start_result_collection__()

        self.__coloring_set__ = True

    def __write_partition_for_sets_of_nodes__(self, sets):
        if type(sets[0]) is not set:
            sets = [set(s) for s in sets]

        if len(self.__additional_color_partitions__) > 0:
            self.__graph__.set_vertex_coloring(\
                            sets + self.__additional_color_partitions__)
        else:
            self.__graph__.set_vertex_coloring(sets)

    def __populate_results__(self, total_t, orbit_info, canon_order):
        total_num_results = len(self.__result_collection__)

        for current_result in range(0, total_num_results):
            current_result_type = self.__result_collection__[current_result][0]
            current_result_value = self.__result_collection__[current_result][1]

            if current_result_type == PyNTSession.AUTOMORPHISMS:
                raise ValueError("Error! get_automorphisms() not implemented.")
                    # ...also, remember that this data is ABOVE the 'header.'

            elif current_result_type == PyNTSession.NUM_AUTOMORPHISM_ORBITS:
                current_result_value.__set_result__(orbit_info[3])

            elif current_result_type == PyNTSession.NUM_AUTOMORPHISMS:
                current_result_value.__set_result__((orbit_info[0], \
                                                     orbit_info[1]))

            elif current_result_type == PyNTSession.RUNTIME:
                our_hours = int(int(total_t) / 3600)
                our_remainder = int(total_t) - our_hours * 3600
                our_minutes = int(our_remainder / 60)
                our_seconds = our_remainder - our_minutes * 60

                res_str = "%d:%d:%d" % (our_hours, our_minutes, our_seconds)

                current_result_value.__set_result__(res_str)

            elif current_result_type == PyNTSession.CANONICAL_ORDER:
                res = []
                for node in canon_order:
                    if node < self.__n__:
                        res.append(node)
                current_result_value.__set_result__(res)

            elif current_result_type == PyNTSession.AUTOMORPHISM_ORBITS:
                orbits = orbit_info[2][:self.__n__]  # TODO: double-check
                orbits_relabel = sorted(list(set(orbits)))
                orbits_relabel = \
                    {orbits_relabel[i]: i for i in range(0, len(orbits_relabel))}
                orbits_result = [[] for _ in range(0, len(orbits_relabel))]
                for i in range(0, len(orbits)):
                    orbits_result[orbits_relabel[orbits[i]]].append(i)
                current_result_value.__set_result__(orbits_result)

            else:
                raise ValueError("Error! Unknown result type %d." % \
                        current_result_type)

    def __has_edge__(self, a, b, t=None):
        if t is None:
            if self.__et_augment__:
                if self.__dictlike_collections__:
                    return b in self.__neighbors_collections__[a]

                for (c, _) in self.__neighbors_collections__[a]:
                    if c == b:
                        return True
                return False
            return b in self.__neighbors_collections__[a]

        if self.__dictlike_collections__:
            return b in self.__neighbors_collections__[a] and \
                self.__neighbors_collections__[a][b] == t

        return (b, t) in self.__neighbors_collections__[a]

    def __write_graph__(self):
        # This big comment has the syntax of the `dreadnaut` binary rather than
        #   the pynauty interface.
        #
        # Example input for an undirected triangle with a dangler from node 1:
        #   -d
        #   n 4
        #   g ; 0 : 1 2 ; 1 : 2 3 .
        #   x
        #   b

        # Determine number of edge types and if they need relabeling.
        if self.__et_augment__:
            edge_types = set()
            if self.__dictlike_collections__:
                for n in range(0, self.__n__):
                    for (n2, t) in self.__neighbors_collections__[n].items():
                        edge_types.add(t)
            else:
                for n in range(0, self.__n__):
                    for (n2, t) in self.__neighbors_collections__[n]:
                        edge_types.add(t)

            edge_types = list(edge_types)
            edge_types.sort()
            if edge_types[0] == 0 and edge_types[-1] == len(edge_types) - 1:
                edge_type_relabeling = None
            else:
                edge_type_relabeling = \
                    {edge_types[i]: i for i in range(0, len(edge_types))}
            num_edge_types = len(edge_types)
            del edge_types

        # Add extra color partition lists for the extra node types.
        if self.__dir_augment__ and self.__et_augment__:
            for _ in range(0, num_edge_types + 1):
                self.__additional_color_partitions__.append(set())
        elif self.__dir_augment__:
            self.__additional_color_partitions__.append(set())
            self.__additional_color_partitions__.append(set())
        elif self.__et_augment__:
            for _ in range(0, num_edge_types):
                self.__additional_color_partitions__.append(set())

        # The input graph needs to be designed such that there are no extra
        #   symmetries due to augmented edges. Otherwise, these extra symmetries
        #   will inflate the number of automorphisms Nauty/Traces reports.
        #
        # The variable `__dir_augment__` is set to True when the input graph is
        #   directed and the mode is Traces.
        #
        # The variable `__et_augment__` is set to True if there are edge types.
        #
        # Cases:
        #
        # 1. Input = a -- b
        #   Output = a -- b
        #
        # 2. Input = a --> b and mode = Nauty
        #   Output = a --> b
        #
        # 3. Input = a --> b and mode = Traces
        #   Output = a     b
        #            |     |
        #            g1 -- g2
        #
        # 4. Input = a <--> b and mode = Nauty
        #   Output = a <--> b
        #
        # 5. Input = a <--> b and mode = Traces
        #   Output = a -- b
        #
        # 6. Input = a -t- b
        #   Output = a -- t -- b
        #
        # 7. Input = a -t-> b and mode = Nauty
        #   Output = a --> t --> b
        #
        # 8. Input = a -t-> b and mode = Traces
        #   Output = a    b
        #            |    |   where g is a special color
        #            g -- t
        #
        # 9. Input = a <-t-> b and mode = Nauty
        #   Output = a <-- t --> b
        #
        # 10. Input = a <-t-> b and mode = Traces
        #   Output  = a -- t -- b
        #
        # 11. Input = a ---t1 and mode = Nauty
        #             ^    |
        #             |    V
        #             t2-- b
        #   Output =  a ---> t1
        #             ^      |
        #             |      V
        #             t2 <-- b
        #
        # 12. Input = a ---t1 and mode = Traces
        #             ^    |
        #             |    V
        #             t2-- b
        #   Output =  g --- t1
        #             |     |
        #             a     b   where g is a special color
        #             |     |
        #             t2 -- g


        if self.__dir_augment__ and self.__et_augment__:
            halves = 0
            for n in range(0, self.__n__):
                if self.__dictlike_collections__:
                    neighbors = self.__neighbors_collections__[n].items()
                else:
                    neighbors = self.__neighbors_collections__[n]
                for (neighbor, t) in neighbors:
                    if self.__has_edge__(neighbor, n, t):
                        # Case 10
                        halves += 1
                    else:
                        # Cases 8 and 12
                        self.__extra_n__ += 2
            assert halves % 2 == 0
            self.__extra_n__ += int(halves / 2)

        elif self.__dir_augment__:
            for n in range(0, self.__n__):
                for neighbor in self.__neighbors_collections__[n]:
                    if not self.__has_edge__(neighbor, n):
                        # Case 3
                        self.__extra_n__ += 2
                    # else: Case 5 -- no extra nodes

        elif self.__et_augment__:
            if self.__directed__:
                halves = 0
                for n in range(0, self.__n__):
                    if self.__dictlike_collections__:
                        neighbors = self.__neighbors_collections__[n].items()
                    else:
                        neighbors = self.__neighbors_collections__[n]
                    for (neighbor, t) in neighbors:
                        if self.__has_edge__(neighbor, n, t):
                            # Case 9
                            halves += 1
                        else:
                            # Cases 7 and 11
                            self.__extra_n__ += 1
                assert halves % 2 == 0
                self.__extra_n__ += int(halves / 2)
            else:
                # Case 6
                for n in range(0, self.__n__):
                    self.__extra_n__ += len(self.__neighbors_collections__[n])
                assert self.__extra_n__ % 2 == 0
                self.__extra_n__ = int(self.__extra_n__ / 2)
        else:
            # Cases 1, 2, and 4 -- no extra nodes
            pass

        total_n = self.__n__ + self.__extra_n__
        if total_n > 2000000000:
            self.__input_file__.close()
            if self.__only_one_call__:
                os.remove(self.__input_filename__)
            else:
                os.remove(self.__intro_filename__)

            if self.__dir_augment__ or self.__et_augment__:
                self.__augment_file__.close()
                os.remove(self.__augment_filename__)

            raise ValueError(("Error! Nauty/Traces cannot handle graphs " + \
                             "with over 2000000000 nodes. \nThis graph has " + \
                "%d nodes plue %d nodes needed for augmenting: %d total." % \
                             (self.__n__, self.__extra_n__, total_n)) + \
                " Note that Traces often needs more augment nodes than Nauty.")

        self.__graph__ = pynauty.Graph(\
            total_n, directed=(self.__directed__ and not self.__dir_augment__))

        # TODO: Update because pynauty cannot take connections piecemeal.

        # Now do the same checks all over again but this time write the graph.
        next_node = self.__n__
        if self.__dir_augment__ and self.__et_augment__:
            for n in range(0, self.__n__):

                if self.__dictlike_collections__:
                    neighbors = self.__neighbors_collections__[n].items()
                else:
                    neighbors = self.__neighbors_collections__[n]

                to_connect = []

                for (neighbor, t) in neighbors:
                    if self.__has_edge__(neighbor, n, t):
                        # Case 10
                        if n >= neighbor:
                            continue

                        # n -- next_node
                        to_connect.append(next_node)
                        # next_node -- neighbor
                        self.__graph__.connect_vertex(next_node, [neighbor])

                        if edge_type_relabeling is None:
                            self.__additional_color_partitions__[t].add(next_node)
                        else:
                            self.__additional_color_partitions__[\
                                edge_type_relabeling[t]].add(next_node)
                        next_node += 1
                    else:
                        # Cases 8 and 12

                        # n -- next_node
                        to_connect.append(next_node)
                        # next_node -- next_node + 1
                        self.__graph__.connect_vertex(next_node, [next_node+1])
                        # next_node + 1 -- neighbor
                        self.__graph__.connect_vertex(next_node + 1, [neighbor])

                        if edge_type_relabeling is None:
                            self.__additional_color_partitions__[-1].add(next_node)
                            self.__additional_color_partitions__[t].add(next_node + 1)
                        else:
                            self.__additional_color_partitions__[\
                                edge_type_relabeling[-1]].add(next_node)
                            self.__additional_color_partitions__[\
                                edge_type_relabeling[t]].add(next_node + 1)
                        next_node += 2

                self.__graph__.connect_vertex(n, to_connect)

        elif self.__dir_augment__:
            for n in range(0, self.__n__):

                to_connect = []

                for neighbor in self.__neighbors_collections__[n]:
                    if not self.__has_edge__(neighbor, n):
                        # Case 3

                        # n -- next_node
                        to_connect.append(next_node)
                        # next_node -- next_node + 1
                        self.__graph__.connect_vertex(next_node, [next_node+1])
                        # next_node + 1 -- neighbor
                        self.__graph__.connect_vertex(next_node + 1, [neighbor])

                        self.__additional_color_partitions__[0].add(next_node)
                        self.__additional_color_partitions__[1].add(next_node + 1)
                        next_node += 2
                    else:
                        # Case 5 -- no extra nodes

                        # n -- neighbor
                        to_connect.append(neighbor)

                self.__graph__.connect_vertex(n, to_connect)

        elif self.__et_augment__:
            if self.__directed__:
                for n in range(0, self.__n__):

                    to_connect = []

                    if self.__dictlike_collections__:
                        neighbors = self.__neighbors_collections__[n].items()
                    else:
                        neighbors = self.__neighbors_collections__[n]
                    for (neighbor, t) in neighbors:
                        if self.__has_edge__(neighbor, n, t):
                            # Case 9
                            if n >= neighbor:
                                continue

                            # next_node --> n
                            # next_node --> neighbor
                            self.__graph__.connect_vertex(next_node, \
                                                            [n, neighbor])

                            if edge_type_relabeling is None:
                                self.__additional_color_partitions__[\
                                    t].add(next_node)
                            else:
                                self.__additional_color_partitions__[\
                                    edge_type_relabeling[t]].add(next_node)
                            next_node += 1
                        else:
                            # Cases 7 and 11

                            # n --> next_node
                            to_connect.append(next_node)
                            # next_node --> neighbor
                            self.__graph__.connect_vertex(next_node, [neighbor])

                            if edge_type_relabeling is None:
                                self.__additional_color_partitions__[\
                                    t].add(next_node)
                            else:
                                self.__additional_color_partitions__[\
                                    edge_type_relabeling[t]].add(next_node)
                            next_node += 1

                    self.__graph__.connect_vertex(n, to_connect)
            else:
                # Case 6
                for n in range(0, self.__n__):

                    to_connect = []

                    if self.__dictlike_collections__:
                        neighbors = self.__neighbors_collections__[n].items()
                    else:
                        neighbors = self.__neighbors_collections__[n]
                    for (neighbor, t) in neighbors:
                        if n >= neighbor:
                            continue

                        # n -- next_node
                        to_connect.append(next_node)
                        # next_node -- neighbor
                        self.__graph__.connect_vertex(next_node, [neighbor])

                        if edge_type_relabeling is None:
                            self.__additional_color_partitions__[\
                                t].add(next_node)
                        else:
                            self.__additional_color_partitions__[\
                                edge_type_relabeling[t]].add(next_node)

                        next_node += 1

                    self.__graph__.connect_vertex(n, to_connect)

        else:
            # Cases 1, 2, and 4 -- no extra nodes
            for n in range(0, self.__n__):

                to_connect = []

                for neighbor in self.__neighbors_collections__[n]:
                    if (not self.__directed__) and n >= neighbor:
                        continue
                    to_connect.append(neighbor)

                self.__graph__.connect_vertex(n, to_connect)


        if self.__kill_py_graph__:
            for _ in range(0, self.__n__):
                v = self.__neighbors_collections__.pop()
                del v
            # A simple del is not enough due to external references.
            del self.__neighbors_collections__

class __PyNautyTracesResult__:
    def __init__(self):
        self.__result__ = None

    def __set_result__(self, r):
        self.__result__ = r

    def get(self):
        if self.__result__ is None:
            raise ValueError("Error! Cannot access Nauty-Traces result " + \
                   "until Nauty/Traces is run. Call <your session>.run().")
        return self.__result__

if __name__ == "__main__":
    print("Testing pynauty Nauty-Traces Session")
    directed = True
    has_edge_types = False
    if directed:
        neighbors_collections = [set([1, 2]), set([3]), set([3]), set(), set()]
    else:
        neighbors_collections = [set([1, 2]), set([0, 3]), set([0, 3]), set([1, 2]), set()]

    session = PyNTSession(directed, has_edge_types, neighbors_collections, \
                                 kill_py_graph=True)

    assert len(neighbors_collections) == 0

    # rt_1 = session.get_runtime()
    # session.set_colors_by_partitions([[1], [2, 0, 3, 4]])
    num_aut_1 = session.get_num_automorphisms()
    orbits_1 = session.get_automorphism_orbits()
    # node_order_1 = session.get_canonical_order()
    session.run()
    # print("Runtime 1:    %s" % rt_1.get())
    num_aut_1 = num_aut_1.get()
    print("Num Aut 1:    %d" % (num_aut_1[0] * 10**(num_aut_1[1])))
    print("Orbits 1:     %s" % orbits_1.get())
    # print("Node Order 1: %s" % node_order_1.get())
    # session.blank_coloring()
    # session.set_colors_by_partitions([[1], [2, 0, 3, 4]])
    session.set_colors_by_coloring([0, 1, 0, 0, 0])
    num_aut_2 = session.get_num_automorphisms()
    orbits_2 = session.get_automorphism_orbits()
    node_order_2 = session.get_canonical_order()
    rt_2 = session.get_runtime()
    session.run()

    session.end_session()
    print("Runtime 2:    %s" % rt_2.get())
    num_aut_2 = num_aut_2.get()
    print("Num Aut 2:    %d" % (num_aut_2[0] * 10**(num_aut_2[1])))
    print("Orbits 2:     %s" % orbits_2.get())
    print("Node Order 2: %s" % node_order_2.get())

    has_edge_types = True
    neighbors_collections = [set([(1, 0), (2, 1)]), set([(3, 0), (0, 0)]), \
                             set([(3, 0), (0, 0)]), set([(1, 0), (2, 0)]), \
                             set([(5, 0)]), set([(4, 0)])]
    # neighbors_collections = [{a: b for (a, b) in s} for s in neighbors_collections]

    for directed in [True, False]:

        session = PyNTSession(directed, has_edge_types, \
                                       neighbors_collections, \
                                       kill_py_graph=False)

        rt_3 = session.get_runtime()
        # session.set_colors_by_partitions([[1], [2, 0, 3]])
        num_aut_3 = session.get_num_automorphisms()
        orbits_3 = session.get_automorphism_orbits()
        node_order_3 = session.get_canonical_order()
        session.run()
        session.end_session()
        # print("Runtime 3:    %s" % rt_3.get())
        num_aut_3 = num_aut_3.get()
        print("Num Aut 3:    %d" % (num_aut_3[0] * 10**(num_aut_3[1])))
        print("Orbits 3:     %s" % orbits_3.get())
        print("Node Order 3: %s" % node_order_3.get())
