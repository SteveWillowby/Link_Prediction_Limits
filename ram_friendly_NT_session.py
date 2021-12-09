import os
import time

# TODO: Add security checks to ensure that the given filepath really is a
#   filepath.

# This file contains the class RAMFriendlyNTSession
#
# NOTE: Requires that the nodes are numbered 0 through |V| - 1.
# NOTE: Requires that nodes are of type int, and no other type, not even
#   int-like types.
#
# The init function for MemEfficientNTSession takes the following arguments:
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
#   `mode` -- "Nauty" or "Traces" -- default of "Traces"
#
#   `kill_py_graph` -- True or False -- if True, destroys neighbors_collections
#       before running Nauty or Traces so that RAM usage is reduced.
#       defaults to False. In order for this to work, `neighbors_collections`
#       must support the pop() operation.
#
#   `only_one_call` -- True or False -- if True, the user can only run on the
#       graph once before needing a new session to run again. However, setting
#       to True reduces Disk usage: -- default of False
#
#   `sparse` -- True or False, only relevant if mode = "Nauty"
#
#   `tmp_path_base` -- part of a path for temporary files
#       default is "/tmp"
#
#   `dreadnaut_call` -- path to (and including) the dreadnaut binary
#       default is "Nauty_n_Traces/nauty26r12/dreadnaut"
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
#   session = RAMFriendlyNTSession(some_graph)
#   first_num_automorphisms = session.get_num_automorphisms()
#   first_runtime = session.get_runtime()
#   session.run()  # THIS LINE IS ESSENTIAL
#   session.set_colors_by_coloring(color_map)
#   second_num_automorphisms = session.get_num_automorphisms()
#   second_runtime = session.get_runtime()
#   session.run()  # THIS LINE IS ESSENTIAL
#   session.end_session()  # Performs cleanup of temp files.
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
# NOTE: Presently get_num_automorphisms returns a STRING to avoid overflow.
# get_num_automorphisms()
#
# run()
#
# end_session()

class RAMFriendlyNTSession:

    def __init__(self, directed, has_edge_types, \
                    neighbors_collections, \
                    mode="Traces", sparse=True, \
                    kill_py_graph=False, \
                    only_one_call=False, \
                    dreadnaut_call="Nauty_n_Traces/nauty26r12/dreadnaut", \
                    tmp_path_base="/tmp", \
                    flush_threshold=None, \
                    announce_launch=False):

        if mode != "Traces" and mode != "Nauty":
            raise ValueError("Error! `mode` must be 'Traces' or 'Nauty'.")
        if mode == "Traces" and not sparse:
            raise ValueError("Error! Cannot run with mode 'Traces' on a " + \
                             "dense graph. Set `sparse` to True.")

        if "/" != tmp_path_base[-1]:
            tmp_path_base = tmp_path_base + "/"
        tmp_path_base = tmp_path_base + ("dreadnaut_%d" % os.getpid())

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
        self.__mode__ = mode
        self.__neighbors_collections__ = neighbors_collections
        self.__dreadnaut_call__ = dreadnaut_call
        self.__kill_py_graph__ = kill_py_graph
        self.__only_one_call__ = only_one_call
        self.__sub_session_num__ = 0
        self.__chars_flushed__ = 0
        self.__scrap_chars_flushed__ = 0
        self.__FLUSH_THRESHOLD__ = flush_threshold
        self.__announce_launch__ = announce_launch

        # Augment to add direction to edges.
        self.__dir_augment__ = self.__mode__ == "Traces" and self.__directed__
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

        if self.__dir_augment__:
            print("NOTE: Running Traces with a directed graph --" + \
                  " this system will augment the graph accordingly.")
        if self.__et_augment__:
            print("NOTE: Running Nauty/Traces with edge types --" + \
                  " this system will augment the graph accordingly.")

        if not self.__only_one_call__:
            self.__intro_filename__ = tmp_path_base + "_intro.txt"
        self.__input_filename__ = tmp_path_base + "_input.txt"
        self.__output_filename__ = tmp_path_base + "_output.txt"
        self.__scrap_filename__ = tmp_path_base + "_scrap.txt"

        if not self.__only_one_call__:
            self.__input_file__ = open(self.__intro_filename__, "w")
        else:
            self.__input_file__ = open(self.__input_filename__, "w")

        if mode == "Traces":
            self.__write__("At\n")
        else:  # mode == "Nauty"
            if sparse:
                self.__write__("As\n")
            else:
                self.__write__("An\n")

        if self.__directed__ and not self.__dir_augment__:
            self.__write__("+d\n")
        else:
            self.__write__("-d\n")

        # Do not print out automorphisms
        self.__write__("-a\n")

        # Do not print out level markers
        self.__write__("-m\n")

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

        partitions = [[] for _ in distinct_colors]
        if distinct_colors[0] != 0 or \
                distinct_colors[-1] != len(distinct_colors) - 1:
            color_remap = \
                {distinct_colors[i]: i for i in range(0, len(distinct_colors))}
            for n in range(0, len(coloring)):
                partitions[color_remap[coloring[n]]].append(n)
        else:
            for n in range(0, len(coloring)):
                partitions[coloring[n]].append(n)

        self.__write_partition_for_lists_of_nodes__(partitions)

    # O(1)
    def blank_coloring(self):
        self.set_colors_by_partitions([[i for i in range(0, self.__n__)]])

    # Will override coloring. Requires a list of lists.
    # O(input)
    def set_colors_by_partitions(self, lists_of_nodes):
        self.__doing_coloring__()
        self.__write_partition_for_lists_of_nodes__(lists_of_nodes)

    def get_automorphism_orbits(self):
        return self.__get_result__(RAMFriendlyNTSession.AUTOMORPHISM_ORBITS)

    def get_runtime(self):
        return self.__get_result__(RAMFriendlyNTSession.RUNTIME)

    def get_canonical_order(self):
        return self.__get_result__(RAMFriendlyNTSession.CANONICAL_ORDER)

    def get_automorphisms(self):
        raise ValueError("Error! get_automorphisms() not yet implemented.\n" + \
               "  Did you want get_automorphism_orbits() by any chance?\n" + \
               "  Or maybe get_num_automorphisms() ?")
        return self.__get_result__(RAMFriendlyNTSession.AUTOMORPHISMS)

    def get_num_automorphisms(self):
        return self.__get_result__(RAMFriendlyNTSession.NUM_AUTOMORPHISMS)

    def run(self):
        if not self.__coloring_set__:
            self.blank_coloring()

        self.__finish_result_collection__()

        # call dreadnaut
        if self.__announce_launch__:
            print("Nauty/Traces input has been written. Calling dreadnaut...")
        start_t = time.time()
        # os.system("cat %s" % self.__input_filename__)
        os.system(self.__dreadnaut_call__ + " < " + self.__input_filename__ + \
                  " > " + self.__output_filename__)
        # os.system("cat %s" % self.__output_filename__)
        total_t = time.time() - start_t
        if self.__announce_launch__:
            print("    ...dreadnaut finished. Now to parse the results.")

        self.__populate_results__(total_t)

    def end_session(self):
        self.__session_ended__ = True
        self.__input_file__.close()

        # Clean up temporary files
        if not self.__only_one_call__:
            os.remove(self.__intro_filename__)
        os.remove(self.__input_filename__)
        if os.path.exists(self.__output_filename__):
            os.remove(self.__output_filename__)


    ########################## Result Functions ##########################

    def __get_result__(self, result_type):
        if self.__session_ended__:
            raise RuntimeError("Error! Cannot get a new result " + \
                               "after session is ended.")

        if not self.__collecting_results__:
            self.__start_result_collection__()

        result = __RAMFriendlyNautyTracesResult__()
        self.__result_collection__.append([result_type, result])
        return result
        

    def __start_result_collection__(self):
        self.__collecting_results__ = True

        self.__input_file__.flush()
        self.__input_file__.close()

        self.__result_collection__ = []

        self.__sub_session_num__ += 1
        if self.__only_one_call__:
            # Continue with the same file.
            if self.__sub_session_num__ > 1:
                raise RuntimeError("Error! Cannot get two of the same kind " + \
                                   "of result (or re-assign colors) when " + \
                                   "`only_one_call` is set to True.")
        else:
            # Copy the intro file to the input file, then add to the input file.
            os.system("cp " + self.__intro_filename__ + " " + \
                      self.__input_filename__)

        self.__input_file__ = open(self.__input_filename__, "a")

    def __finish_result_collection__(self):
        self.__collecting_results__ = False
        self.__coloring_set__ = False
        self.__result_collection__.sort()
        has_canonization = False
        for result in self.__result_collection__:
            if result[0] == RAMFriendlyNTSession.CANONICAL_ORDER:
                has_canonization = True
                break
        if has_canonization:
            self.__write__("+c\n")  # Canonize when running
        else:
            self.__write__("-c\n")  # Don't canonize when running
        self.__write__("x\n")  # Run

        output_canonization = False
        output_orbits = False
        for result in self.__result_collection__:
            if result[0] == RAMFriendlyNTSession.CANONICAL_ORDER:
                if not output_canonization:
                    self.__write__("b\n")
                output_canonization = True
            elif result[0] == RAMFriendlyNTSession.AUTOMORPHISM_ORBITS:
                if not output_orbits:
                    self.__write__("o\n")
                output_orbits = True

        self.__input_file__.flush()

    def __write__(self, s):
        self.__input_file__.write(s)
        if self.__FLUSH_THRESHOLD__ is None:
            return

        curr_size = self.__input_file__.tell()
        if curr_size >= self.__chars_flushed__ + self.__FLUSH_THRESHOLD__:
            self.__input_file__.flush()
            self.__chars_flushed__ = curr_size

    def __scrap_write__(self, s):
        self.__scrap_file__.write(s)
        if self.__FLUSH_THRESHOLD__ is None:
            return

        curr_size = self.__scrap_file__.tell()
        if curr_size >= self.__scrap_chars_flushed__ + self.__FLUSH_THRESHOLD__:
            self.__scrap_file__.flush()
            self.__scrap_chars_flushed__ = curr_size

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

    def __write_partition_for_lists_of_nodes__(self, lists):
        edge_colors = len(self.__additional_color_partitions__) > 0

        self.__write__("f=[")
        for i in range(0, len(lists)):
            for j in range(0, len(lists[i])):
                self.__write__("%d" % lists[i][j])
                if j < len(lists[i]) - 1:
                    self.__write__(", ")
            if i < len(lists) - 1 or edge_colors:
                self.__write__("|")

        if edge_colors:
            added_lists = self.__additional_color_partitions__
            for i in range(0, len(added_lists)):
                for j in range(0, len(added_lists[i])):
                    self.__write__("%d" % added_lists[i][j])
                    if j < len(added_lists[i]) - 1:
                        self.__write__(", ")
                if i < len(added_lists) - 1:
                    self.__write__("|")

        self.__write__("]\n")


    def __readline__(self):
        line = self.__output_file__.readline()
        # print(line)
        done = len(line) == 0
        return (line.strip().split(" "), done)

    def __populate_results__(self, total_t):
        current_result = 0
        total_num_results = len(self.__result_collection__)
        # last_header_line_idx = -1
        # line_idx = -1
        got_canonical_order = False

        self.__output_file__ = open(self.__output_filename__, "r")

        found_header = False
        while not found_header:
            (line, _) = self.__readline__()
            if len(line) >= 3 and \
                    line[2][:7] == "grpsize":
                found_header = True
                num_orbits = int(line[0])

                num_automorphisms = line[2][8:-1]

        got_canonical_order = False
        got_runtime = False

        while current_result < total_num_results:
            current_result_type = self.__result_collection__[current_result][0]
            current_result_value = self.__result_collection__[current_result][1]

            if current_result_type == RAMFriendlyNTSession.AUTOMORPHISMS:
                raise ValueError("Error! get_automorphisms() not implemented.")
                    # ...also, remember that this data is ABOVE the 'header.'

            elif current_result_type == RAMFriendlyNTSession.NUM_AUTOMORPHISM_ORBITS:
                current_result_value.__set_result__(num_orbits)

            elif current_result_type == RAMFriendlyNTSession.NUM_AUTOMORPHISMS:
                current_result_value.__set_result__(num_automorphisms)

            elif current_result_type == RAMFriendlyNTSession.RUNTIME:
                # Line was at header. Go down one.
                (line, _) = self.__readline__()

                time_idx = 0
                while line[time_idx] != "time":
                    time_idx += 1
                time_idx += 2  # skip the "time" and the "=" sign

                res_str = "Nauty/Traces: ' cpu time = "
                while time_idx < len(line):
                    res_str += line[time_idx] + " "
                    time_idx += 1

                # That was Nauty/Traces estimate. Our timing of running
                #   the subprocess is a bit longer:
                # (i.e.) subprocess-run("./dreadnaut < prepared_input.txt")
                our_hours = int(int(total_t) / 3600)
                our_remainder = int(total_t) - our_hours * 3600
                our_minutes = int(our_remainder / 60)
                our_seconds = our_remainder - our_minutes * 60

                res_str += "' | Our Subproces Call: %d:%d:%d" % \
                            (our_hours, our_minutes, our_seconds)

                got_runtime = True
                current_result_value.__set_result__(res_str)

            elif current_result_type == RAMFriendlyNTSession.CANONICAL_ORDER:
                if not got_runtime:
                    _ = self.__readline__()
                (line, _) = self.__readline__()
                node_order = []
                while ':' not in line:
                    for s in line:
                        i = int(s)
                        if i < self.__n__:
                            node_order.append(i)

                    (line, _) = self.__readline__()

                # print("Got Canonical Order of %s" % node_order)
                # print("Line currently is: '%s'" % line)
                current_result_value.__set_result__(node_order)

                got_canonical_order = True

            elif current_result_type == RAMFriendlyNTSession.AUTOMORPHISM_ORBITS:
                if not got_canonical_order:
                    if not got_runtime:
                        (line, _) = self.__readline__()
                    (line, _) = self.__readline__()
                if got_canonical_order:
                    # Skip over the canonical adjacency list.
                    assert ':' in line
                    for _ in range(0, self.__n__ + self.__extra_n__):
                        while ';' not in line[-1]:
                            (line, _) = self.__readline__()
                        (line, _) = self.__readline__()

                orbits = []
                orbit = []
                last_value = False
                while (line[0][0] in \
                            ['0','1','2','3','4','5','6','7','8','9'] or \
                       line[0][-2:] == ");") and \
                            'orbits;' not in line:
                    for sub_str in line:
                        if sub_str[0] == "(":
                            last_value = True
                        else:
                            if sub_str[-1] == ";":
                                value = sub_str[:-1]
                                last_value = True
                            else:
                                value = sub_str

                            if ":" in value:
                                s = value.split(":")
                                orbit += [i for i in range(int(s[0]), int(s[1])+1)]
                            else:
                                orbit.append(int(value))

                        if last_value:
                            if orbit[0] < self.__n__:
                                orbits.append(orbit)
                            last_value = False
                            orbit = []

                    (line, done) = self.__readline__()
                    if done:
                        # End of file reached. Stop.
                        break

                current_result_value.__set_result__(orbits)

            else:
                self.__output_file__.close()
                os.remove(self.__output_filename__)
                raise ValueError("Error! Unknown result type %d." % \
                        current_result_type)

            current_result += 1

        self.__output_file__.close()
        os.remove(self.__output_filename__)

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
        # Add graph to session
        #
        # Example input for an undirected triangle with a dangler from node 1:
        #   -d
        #   n 4
        #   g ; 0 : 1 2 ; 1 : 2 3 .
        #   x
        #   b
        #
        # NOTE: Nauty/Traces can receive nodes' neighbors in multiple spurts.
        #   For example, the below does the same as the above.
        #   -d
        #   n 4
        #   g ; 0 : 1 ; 1 : 2 3 ; 0 : 2 .
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
                self.__additional_color_partitions__.append([])
        elif self.__dir_augment__:
            self.__additional_color_partitions__.append([])
            self.__additional_color_partitions__.append([])
        elif self.__et_augment__:
            for _ in range(0, num_edge_types):
                self.__additional_color_partitions__.append([])

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
        #   Output = a --> t <-- b
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


        self.__write__("n %d\n" % (self.__n__ + self.__extra_n__))
        self.__write__("g ")

        # Now do the same checks all over again but this time write the graph.
        next_node = self.__n__
        if self.__dir_augment__ and self.__et_augment__:
            for n in range(0, self.__n__):
                if self.__dictlike_collections__:
                    neighbors = self.__neighbors_collections__[n].items()
                else:
                    neighbors = self.__neighbors_collections__[n]

                for (neighbor, t) in neighbors:
                    if self.__has_edge__(neighbor, n, t):
                        # Case 10
                        if n >= neighbor:
                            continue

                        self.__write__("; %d : %d " % (n, next_node))
                        self.__write__("; %d : %d " % (neighbor, next_node))
                        if edge_type_relabeling is None:
                            self.__additional_color_partitions__[t].append(next_node)
                        else:
                            self.__additional_color_partitions__[\
                                edge_type_relabeling[t]].append(next_node)
                        next_node += 1
                    else:
                        # Cases 8 and 12
                        self.__write__("; %d : %d " % (n, next_node))
                        self.__write__("; %d : %d " % \
                                        (neighbor, next_node + 1))
                        self.__write__("; %d : %d " % \
                                        (next_node, next_node + 1))
                        if edge_type_relabeling is None:
                            self.__additional_color_partitions__[-1].append(next_node)
                            self.__additional_color_partitions__[t].append(next_node + 1)
                        else:
                            self.__additional_color_partitions__[\
                                edge_type_relabeling[-1]].append(next_node)
                            self.__additional_color_partitions__[\
                                edge_type_relabeling[t]].append(next_node + 1)
                        next_node += 2

        elif self.__dir_augment__:
            for n in range(0, self.__n__):
                for neighbor in self.__neighbors_collections__[n]:
                    if not self.__has_edge__(neighbor, n):
                        # Case 3
                        self.__write__("; %d : %d " % (n, next_node))
                        self.__write__("; %d : %d " % (neighbor, next_node + 1))
                        self.__write__("; %d : %d " % (next_node, next_node + 1))
                        self.__additional_color_partitions__[0].append(next_node)
                        self.__additional_color_partitions__[1].append(next_node + 1)
                        next_node += 2
                    else:
                        # Case 5 -- no extra nodes
                        self.__write__("; %d : %d " % (n, neighbor))

        elif self.__et_augment__:
            if self.__directed__:
                for n in range(0, self.__n__):
                    if self.__dictlike_collections__:
                        neighbors = self.__neighbors_collections__[n].items()
                    else:
                        neighbors = self.__neighbors_collections__[n]
                    for (neighbor, t) in neighbors:
                        if self.__has_edge__(neighbor, n, t):
                            # Case 9
                            if n >= neighbor:
                                continue
                            self.__write__("; %d : %d " % (n, next_node))
                            self.__write__("; %d : %d " % (neighbor, next_node))
                            if edge_type_relabeling is None:
                                self.__additional_color_partitions__[\
                                    t].append(next_node)
                            else:
                                self.__additional_color_partitions__[\
                                    edge_type_relabeling[t]].append(next_node)
                            next_node += 1
                        else:
                            # Cases 7 and 11
                            self.__write__("; %d : %d " % (n, next_node))
                            self.__write__("; %d : %d " % (next_node, neighbor))
                            if edge_type_relabeling is None:
                                self.__additional_color_partitions__[\
                                    t].append(next_node)
                            else:
                                self.__additional_color_partitions__[\
                                    edge_type_relabeling[t]].append(next_node)
                            next_node += 1
            else:
                # Case 6
                for n in range(0, self.__n__):
                    if self.__dictlike_collections__:
                        neighbors = self.__neighbors_collections__[n].items()
                    else:
                        neighbors = self.__neighbors_collections__[n]
                    for (neighbor, t) in neighbors:
                        if n >= neighbor:
                            continue
                        self.__write__("; %d : %d " % (n, next_node))
                        self.__write__("; %d : %d " % (neighbor, next_node))

                        if edge_type_relabeling is None:
                            self.__additional_color_partitions__[\
                                t].append(next_node)
                        else:
                            self.__additional_color_partitions__[\
                                edge_type_relabeling[t]].append(next_node)

                        next_node += 1

        else:
            # Cases 1, 2, and 4 -- no extra nodes
            for n in range(0, self.__n__):
                self.__write__("; %d : " % n)
                for neighbor in self.__neighbors_collections__[n]:
                    if (not self.__directed__) and n >= neighbor:
                        continue
                    self.__write__("%d " % neighbor)

        self.__write__(".\n")

        if self.__kill_py_graph__:
            for _ in range(0, self.__n__):
                self.__neighbors_collections__.pop()
            # A simple del is not enough due to external references.
            del self.__neighbors_collections__

class __RAMFriendlyNautyTracesResult__:
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
    print("Testing RAM-Friendly Nauty-Traces Session")
    directed = True
    has_edge_types = False
    if directed:
        neighbors_collections = [set([1, 2]), set([3]), set([3]), set(), set()]
    else:
        neighbors_collections = [set([1, 2]), set([0, 3]), set([0, 3]), set([1, 2]), set()]

    session = RAMFriendlyNTSession(directed, has_edge_types, neighbors_collections, \
                                 mode="Traces", \
                                 only_one_call=False, \
                                 kill_py_graph=True, \
                                 sparse=True)

    assert len(neighbors_collections) == 0

    # TODO: Debug -- the Traces direction augmentation is failing.

    # rt_1 = session.get_runtime()
    # session.set_colors_by_partitions([[1], [2, 0, 3, 4]])
    num_aut_1 = session.get_num_automorphisms()
    orbits_1 = session.get_automorphism_orbits()
    # node_order_1 = session.get_canonical_order()
    session.run()
    # print("Runtime 1:    %s" % rt_1.get())
    print("Num Aut 1:    %s" % num_aut_1.get())
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
    print("Num Aut 2:    %s" % num_aut_2.get())
    print("Orbits 2:     %s" % orbits_2.get())
    print("Node Order 2: %s" % node_order_2.get())

    has_edge_types = True
    neighbors_collections = [set([(1, 0), (2, 1)]), set([(3, 0), (0, 0)]), \
                             set([(3, 0), (0, 0)]), set([(1, 0), (2, 0)]), \
                             set([(5, 0)]), set([(4, 0)])]
    # neighbors_collections = [{a: b for (a, b) in s} for s in neighbors_collections]

    for directed in [True, False]:

        session = RAMFriendlyNTSession(directed, has_edge_types, \
                                       neighbors_collections, \
                                       mode="Traces", \
                                       only_one_call=False, \
                                       kill_py_graph=False, \
                                       sparse=True)

        rt_3 = session.get_runtime()
        # session.set_colors_by_partitions([[1], [2, 0, 3]])
        num_aut_3 = session.get_num_automorphisms()
        orbits_3 = session.get_automorphism_orbits()
        node_order_3 = session.get_canonical_order()
        session.run()
        # print("Runtime 3:    %s" % rt_3.get())
        print("Num Aut 3:    %s" % num_aut_3.get())
        print("Orbits 3:     %s" % orbits_3.get())
        print("Node Order 3: %s" % node_order_3.get())
