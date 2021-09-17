import bigfloat  # TODO: Move away from bigfloat.
import subprocess
from os import getpid, remove

# This file contains the class NautyTracesSession, a class designed
#   to allow runs of nauty/traces that do not require continually
#   re-loading the entire graph after small changes are made to the graph.
#
# The init function for NautyTracesSession takes the following arguments:
#   `start_graph`
#   `mode` -- "Nauty" or "Traces" -- default of "Traces"
#   `sparse` -- True or False, only relevant if mode = "Nauty"
#   `tmp_path` -- path for a temporary file used when calling nauty
#       default is "/tmp/<pid>_dreadnaut.txt"
#   `dreadnaut_call` -- path to (and including) the dreadnaut binary
#       default is "Nauty_n_Traces/nauty26r12/dreadnaut"
#
# Assumes `start_graph` is a networkx graph, or a graph class which supports
#   the following methods:
#       nodes()
#       is_directed()
#       neighbors(node) -- required if is_directed() returns false
#       successors(node) -- required if is_directed() returns true
#       predecessors(node) -- required if is_directed() returns true
#
# `start_graph` is not modified during the run of the session; rather, the
#   class COPIES `start_graph`
#
# INTERFACE:
#
# GRAPH EDIT FUNCTIONS:
#
# IMPORTANT: graph edit functions can only be used with "Nauty" and sparse=False
#
# O(1)
# add_node(node)
#
# O(num edges for node)
# delete_node(node)
#
# O(1)
# add_edge(s, t)
#
# O(1)
# delete_edge(s, t)
#
#
#
# COLORING FUNCTIONS:
#
# O(color-map log color-map) = O(nodes log nodes)
#   color_map must support a .items() call.
# set_colors_by_map(color_map)
#
# O(1)
# reset_colors()
#
# Will override coloring. Requires a list of lists.
# O(the input)
#   All the nodes not included are given a distinct color.
# set_colors_by_highlights(lists_of_nodes)
#
#
#
#
# RESULT FUNCTIONS: IMPORTANT: To actually get the results, you must first
#   call ALL the get_ functions you want and then call complete().
#   After complete() is called, nauty/traces will actually run. Once this
#   finishes, you can access the result values by calling .get()
#
#   After calling complete(), the session is done.
#
# Example:
#
#   session = NautyTracesSession(some_graph)
#   first_num_automorphisms = session.get_num_automorphisms()
#   first_runtime = session.get_runtime()
#   session.add_node("Ben")
#   session.add_node("Jerry")
#   session.add_edge("Ben", "Jerry")
#   second_num_automorphisms = session.get_num_automorphisms()
#   second_runtime = session.get_runtime()
#   session.complete()  # THIS LINE IS ESSENTIAL
#
#   print(first_num_automorphisms.get())
#   print(first_runtime.get())
#   print(second_num_automorphisms.get())
#   print(second_runtime.get())
#   
#
# get_automorphism_orbits()
#
# get_runtime()
#
# get_canonical_order()
#
# get_automorphisms() -- currently not implemented
#
# get_num_automorphisms()
#
# get_num_automorphism_orbits()
#
# complete()

class NautyTracesSession:

    def __init__(self, start_graph, mode="Nauty", sparse=True, \
                    allow_edits=False, \
                    tmp_path="/tmp/%d_dreadnaut.txt" % getpid(), \
                    dreadnaut_call="Nauty_n_Traces/nauty26r12/dreadnaut"):

        if allow_edits and sparse:
            raise ValueError("Error! Can only edit the Nauty/Traces" + \
                             " graph mid-session in dense mode.")
        elif allow_edits and mode != "Nauty":
            raise ValueError("Error! Can only edit the Nauty/Traces" + \
                             " graph with mode='Nauty'.")
        self.allow_edits = allow_edits

        # start_nodes_list gets used in the original graph write to make sure
        #   that the node order there matches the node order used throughout the
        #   overal run
        self.start_nodes_list = list(start_graph.nodes())

        self.nodes = set(self.start_nodes_list)
        self.num_start_nodes = len(self.nodes)
        if self.num_start_nodes == 0:
            raise ValueError("Error! start_graph must have at least one node.")
        elif self.num_start_nodes > 2000000000:
            raise ValueError("Error! Nauty/Traces cannot handle graphs " + \
                             "with over 2000000000 nodes. This graph has " + \
                             "%d nodes." % self.num_start_nodes)

        self.max_num_nodes = self.num_start_nodes
        self.num_nodes = self.num_start_nodes
        self.unused_indices = []

        self.node_to_idx_map = {}
        self.idx_to_node_map = []
        i = 0
        for node in self.start_nodes_list:
            self.node_to_idx_map[node] = i
            self.idx_to_node_map.append(node)
            i += 1

        self.start_node_to_idx_map = dict(self.node_to_idx_map)

        self.directed = start_graph.is_directed()
        self.start_graph = start_graph

        if allow_edits:
            if self.directed:
                self.out_neighbors = \
                    {n : set(start_graph.successors(n)) for n in self.nodes}
                self.in_neighbors = \
                    {n : set(start_graph.predecessors(n)) for n in self.nodes}
            else:
                self.neighbors = \
                    {n : set(start_graph.neighbors(n)) for n in self.nodes}

        self.singleton_nodes = set()
        for n in self.nodes:
            if self.directed:
                # Coded this way to avoid copying the edges
                singleton = True
                for node in start_graph.predecessors(n):
                    singleton = False
                    break
                for node in start_graph.successors(n):
                    singleton = False
                    break
                if singleton:
                    self.singleton_nodes.add(n)
            else:
                # Coded this way to avoid copying the edges
                singleton = True
                for node in start_graph.neighbors(n):
                    singleton = False
                    break
                if singleton:
                    self.singleton_nodes.add(n)

        self.colored_nodes = set()

        self.tmp_path = tmp_path
        self.dreadnaut_call = dreadnaut_call

        # session_init_lines will be used to feed into 
        self.session_init_lines = []

        if mode == "Traces":
            if self.directed:
                print("Warning! Running Traces with a directed graph --" + \
                        " sometimes Traces segfaults on directed graphs.")
            self.session_init_lines.append("At")
        elif mode == "Nauty":
            if sparse:
                self.session_init_lines.append("As")
            else:
                self.session_init_lines.append("An")
        else:
            raise ValueError(("Error! Unknown mode '%s' - " % mode) + \
                              "Input 'Nauty' or 'Traces'")

        if self.directed:
            self.session_init_lines.append("+d")
        else:
            self.session_init_lines.append("-d")

        # Do not print out automorphisms
        self.session_init_lines.append("-a")

        # Do not print out level markers
        self.session_init_lines.append("-m")

        # Commands executed after the init.
        self.session_lines = []

        # A list of NautyTracesResult objects to populate after the execution.
        # The format of each element is:
        # [result_type, nauty-traces-result object, run-idx]
        self.result_list = []

        # Result types are below:
        self.AUTOMORPHISMS = 0  # This one currently not implemented.
        self.NUM_AUTOMORPHISM_ORBITS = 1
        self.NUM_AUTOMORPHISMS = 2
        self.RUNTIME = 3
        self.CANONICAL_ORDER = 4
        self.AUTOMORPHISM_ORBITS = 5

        # For a given graph state, store the result collection in this sub-list
        #   until the full result collection is present, so that results can be
        #   ordered in the way the program spits them out.
        # The format for each element is:
        # [result_type, nauty-traces-result object]
        self.current_result_collection = []

        # Have data for each execution.
        self.node_to_idx_maps = []
        self.idx_to_node_maps = []
        self.nums_of_uncolored_singleton_nodes = []

        self.num_completed_executions = 0
        self.collecting_results = False
        self.coloring = False

        self.everything_complete = False


    # Allow distinct nodes types with min and max functions:
    def __node_comp__(self, a, b):
        if (type(a) is tuple and type(b) is tuple) or \
                (type(a) is list and type(b) is list):
            if len(a) < len(b):
                return -1
            elif len(a) > len(b):
                return 1
            else:
                for i in range(0, len(a)):
                    r = self.__node_comp__(a[i], b[i])
                    if r != 0:
                        return r
                return 0
        elif type(a) is type(b):
            if a < b:
                return -1
            elif a > b:
                return 1
            else:
                0

        # Otherwise, just compare typenames.
        return str(type(a)) < str(type(b))

    def __node_min__(self, a, b):
        if self.__node_comp__(a, b) <= 0:
            return a
        return b

    def __node_max__(self, a, b):
        if self.__node_comp__(a, b) >= 0:
            return b
        return a

    # Graph Editing

    def __doing_edit__(self):
        if not self.allow_edits:
            raise RuntimeError("Error! Can only edit the graph mid-session " + \
                    "when running with mode='Nauty', sparse=False," + \
                    " and allow_edits=True.")
        if self.everything_complete:
            raise RuntimeError("Error! Cannot perform an edit to the graph " + \
                    "after session is complete.")

        if self.coloring:
            raise RuntimeError("Error! Cannot perform an edit to the graph " + \
                    "after coloring nodes and before getting results.")

        if self.collecting_results:
            self.__finish_result_collection__()
        self.collecting_results = False

    # O(1)
    def add_node(self, node):
        self.__doing_edit__()

        if node in self.nodes:
            raise ValueError("Error! Already have node %s." % str(node))
        if self.num_nodes + 1 > self.max_num_nodes:
            if self.num_nodes + 1 > 2000000000:
                raise ValueError("Error! Cannot have more than %d nodes." \
                                    % 2000000000)

            self.max_num_nodes = self.num_nodes + 1

            # Never had this many nodes before, so offer a new index.
            self.unused_indices.append(self.num_nodes)
            # Also, expand the idx-to-node map
            self.idx_to_node_map.append(None)

        self.num_nodes += 1
        self.singleton_nodes.add(node)
        self.nodes.add(node)
        node_idx = self.unused_indices.pop()

        self.node_to_idx_map[node] = node_idx
        self.idx_to_node_map[node_idx] = node
        if self.directed:
            self.out_neighbors[node] = set()
            self.in_neighbors[node] = set()
        else:
            self.neighbors[node] = set()

    # O(num edges for node)
    def delete_node(self, node):
        self.__doing_edit__()

        if node not in self.nodes:
            raise ValueError("Error! No node %s to delete." % str(node))

        self.num_nodes -= 1
        self.nodes.remove(node)
        node_idx = self.node_to_idx_map[node]
        del self.node_to_idx_map[node]
        self.idx_to_node_map[node_idx] = None
        self.unused_indices.append(node_idx)

        if self.directed:
            out_neighbors = list(self.out_neighbors[node])
            in_neighbors = list(self.in_neighbors[node])

            for t in out_neighbors:
                self.delete_edge(node, t)
            for s in in_neighbors:
                self.delete_edge(s, node)

            del self.out_neighbors[node]
            del self.in_neighbors[node]
        else:
            neighbors = list(self.neighbors[node])
            for n in neighbors:
                self.delete_edge(node, n)
            del self.neighbors[node]

        # The (deleted) node is now a singleton node disappearing.
        self.singleton_nodes.remove(node)

    # O(1)
    def add_edge(self, s, t):
        self.__doing_edit__()

        if s not in self.nodes:
            raise ValueError(("Error! Cannot add edge (%s, %s) " % (str(s), str(t))) + \
                                (" because node %s does not exist." % (str(s))))
        if t not in self.nodes:
            raise ValueError(("Error! Cannot add edge (%s, %s) " % (str(s), str(t))) + \
                                (" because node %s does not exist." % (str(t))))
        if self.directed:
            if t in self.out_neighbors[s]:
                raise ValueError("Error! Edge (%s, %s) already exists." % \
                                    (str(s), str(t)))
        else:
            if t in self.neighbors[s]:
                raise ValueError("Error! Edge (%s, %s) already exists." % \
                                    (str(s), str(t)))

        if self.directed:
            if len(self.in_neighbors[t]) + len(self.out_neighbors[t]) == 0:
                self.singleton_nodes.remove(t)
            if len(self.in_neighbors[s]) + len(self.out_neighbors[s]) == 0:
                self.singleton_nodes.remove(s)

            self.in_neighbors[t].add(s)
            self.out_neighbors[s].add(t)
        else:
            if len(self.neighbors[t]) == 0:
                self.singleton_nodes.remove(t)
            if len(self.neighbors[s]) == 0:
                self.singleton_nodes.remove(s)

            self.neighbors[s].add(t)
            self.neighbors[t].add(s)

        self.session_lines.append("e %d : %d ." % \
                (self.node_to_idx_map[s], self.node_to_idx_map[t]))

    # O(1)
    def delete_edge(self, s, t):
        self.__doing_edit__()

        if s not in self.nodes:
            raise ValueError(("Error! Cannot delete edge (%s, %s) " % (str(s), str(t))) + \
                                (" because node %s does not exist." % (str(s))))
        if t not in self.nodes:
            raise ValueError(("Error! Cannot delete edge (%s, %s) " % (str(s), str(t))) + \
                                (" because node %s does not exist." % (str(t))))
        if self.directed:
            if t not in self.out_neighbors[s]:
                raise ValueError("Error! Edge (%s, %s) does not exist." % \
                                    (str(s), str(t)))
        else:
            if t not in self.neighbors[s]:
                raise ValueError("Error! Edge (%s, %s) does not exist." % \
                                    (str(s), str(t)))

        if self.directed:
            self.in_neighbors[t].remove(s)
            self.out_neighbors[s].remove(t)

            if len(self.in_neighbors[t]) + len(self.out_neighbors[t]) == 0:
                self.singleton_nodes.add(t)
            if len(self.in_neighbors[s]) + len(self.out_neighbors[s]) == 0:
                self.singleton_nodes.add(s)
        else:
            self.neighbors[s].remove(t)
            self.neighbors[t].remove(s)

            if len(self.neighbors[t]) == 0:
                self.singleton_nodes.add(t)
            if len(self.neighbors[s]) == 0:
                self.singleton_nodes.add(s)

        self.session_lines.append("e %d : -%d ." % \
                (self.node_to_idx_map[s], self.node_to_idx_map[t]))

    def __doing_coloring__(self):
        if self.everything_complete:
            raise RuntimeError("Error! Cannot perform coloring on the graph " +\
                    "after session is complete.")

        if self.collecting_results:
            self.__finish_result_collection__()
        self.collecting_results = False

        self.coloring = True

    def __partition_string_for_lists_of_indices__(self, lists):
        partition_strings = [str(p)[1:-1] for p in lists]
        partition_string = "f=["
        for i in range(0, len(partition_strings)):
            partition_string += partition_strings[i]
            if i < len(partition_strings) - 1:
                partition_string += "|"
        partition_string += "]"
        return partition_string

    # O(color-map log color-map) = O(nodes log nodes)
    #
    # color_map must support a .items() call.
    def set_colors_by_map(self, color_map):
        self.__doing_coloring__()

        nodes_in_map = set([n for n, c in color_map.items()])
        if nodes_in_map != self.nodes:
            raise ValueError("Error! nodes in color_map are not the same " + \
                             "set as the nodes in the graph.")
        partition = [(c, self.node_to_idx_map[n]) for n, c in color_map.items()]
        partition.sort()
        partitions = []
        last_color = None
        for (c, i) in partition:
            if last_color is None:
                p = []
            elif c != last_color:
                partitions.append(p)
                p = []
            p.append(i)
        partitions.append(p)

        partition_string = \
            self.__partition_string_for_lists_of_indices__(partitions)
        self.session_lines.append(partition_string)

        self.colored_nodes = set(self.nodes)

    # O(1)
    def reset_colors(self):
        self.__doing_coloring__()

        self.session_lines.append("f=[]")

        self.colored_nodes = set()

    # Will override coloring. Requires a list of lists.
    # O(input)
    #
    # All the nodes not included are given a distinct color.
    def set_colors_by_highlights(self, lists_of_nodes):
        self.__doing_coloring__()

        nodes_in_list = list(lists_of_nodes[0])
        for i in range(1, len(lists_of_nodes)):
            nodes_in_list += lists_of_nodes[i]
        relevant_nodes = set(nodes_in_list)

        if len(nodes_in_list) != len(relevant_nodes):
            raise ValueError("Error! Cannot give a node two different colors!")

        index_lists = \
            [[self.node_to_idx_map[n] for n in l] for l in lists_of_nodes]
        partition_string = \
            self.__partition_string_for_lists_of_indices__(index_lists)
        self.session_lines.append(partition_string)

        self.colored_nodes = relevant_nodes

    # Result Functions

    def __get_result__(self, result_type):
        if self.everything_complete:
            raise RuntimeError("Error! Cannot get a new result " + \
                    "after session is complete.")

        self.coloring = False

        if not self.collecting_results:
            self.__start_result_collection__()

        result = NautyTracesResult()
        self.current_result_collection.append(\
                [result_type, result, self.num_completed_executions])
        return result
        

    def __start_result_collection__(self):
        self.collecting_results = True

        self.node_to_idx_maps.append(dict(self.node_to_idx_map))
        self.idx_to_node_maps.append(list(self.idx_to_node_map))
        self.nums_of_uncolored_singleton_nodes.append(\
            len(self.singleton_nodes - self.colored_nodes))

        self.current_result_collection = []

    def __finish_result_collection__(self):
        self.collecting_results = True
        self.current_result_collection.sort()
        has_canonization = False
        for result in self.current_result_collection:
            if result[0] == self.CANONICAL_ORDER:
                has_canonization = True
                break
        if has_canonization:
            self.session_lines.append("+c")  # Canonize when running
        else:
            self.session_lines.append("-c")  # Don't canonize when running
        self.session_lines.append("x")  # Run

        output_canonization = False
        output_orbits = False
        for result in self.current_result_collection:
            if result[0] == self.CANONICAL_ORDER:
                if not output_canonization:
                    self.session_lines.append("b")
                output_canonization = True
            elif result[0] == self.AUTOMORPHISM_ORBITS:
                if not output_orbits:
                    self.session_lines.append("o")
                output_orbits = True

        self.num_completed_executions += 1
        self.result_list += list(self.current_result_collection)

    def get_automorphism_orbits(self):
        return self.__get_result__(self.AUTOMORPHISM_ORBITS)

    def get_runtime(self):
        return self.__get_result__(self.RUNTIME)

    def get_canonical_order(self):
        return self.__get_result__(self.CANONICAL_ORDER)

    def get_automorphisms(self):
        raise ValueError("Error! get_automorphisms() not yet implemented.\n" + \
               "  Did you want get_automorphism_orbits() by any chance?\n" + \
               "  Or maybe get_num_automorphisms() ?")
        return self.__get_result__(self.AUTOMORPHISMS)

    def get_num_automorphisms(self):
        return self.__get_result__(self.NUM_AUTOMORPHISMS)

    def get_num_automorphism_orbits(self):
        return self.__get_result__(self.NUM_AUTOMORPHISM_ORBITS)

    def complete(self):
        self.__finish_result_collection__()
        self.__input_start_graph_with_enough_extra_blank_nodes__()

        all_lines = self.session_init_lines + self.session_lines

        with open(self.tmp_path, 'w') as f:
            print("\n".join(all_lines), file=f)

        # call dreadnaut
        proc = subprocess.run([self.dreadnaut_call],
                              input=b"< " + self.tmp_path.encode(),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL)
        res = proc.stdout.decode()
        lines = res.strip().split("\n")
        for line_idx in range(0, len(lines)):
            l = lines[line_idx].strip().split(" ")
            new_line = []
            for v in l:
                if v != '':
                    new_line.append(v)
            lines[line_idx] = l

        print(lines)
        self.__populate_results_from_lines__(lines)

        self.everything_complete = True


        # Clean up temporary file
        remove(self.tmp_path)

    def __populate_results_from_lines__(self, lines):
        current_result = 0
        total_num_results = len(self.result_list)
        header_idx = -1
        last_header_line_idx = -1
        line_idx = -1
        got_canonical_order = False
        while current_result < total_num_results:
            target_run = self.result_list[current_result][2]

            if header_idx < target_run:
                line_idx = last_header_line_idx
            while header_idx < target_run:
                line_idx += 1
                if len(lines[line_idx]) >= 3 and \
                        lines[line_idx][2][:7] == "grpsize":
                    header_idx += 1
                    last_header_line_idx = line_idx
                    num_orbits = int(lines[line_idx][0])
                    # TODO: Move away from bigfloat.
                    num_automorphisms = bigfloat.BigFloat(lines[line_idx][2][8:-1])

                    node_indices_at_time = set([i for n, i in \
                            self.node_to_idx_maps[target_run].items()])

                    num_nodes_at_time = len(node_indices_at_time)
                    num_uncolored_singleton_nodes_at_time = \
                        self.nums_of_uncolored_singleton_nodes[target_run]

                    num_extra_nodes = self.max_num_nodes - num_nodes_at_time
                    if num_extra_nodes > 0:
                        if num_uncolored_singleton_nodes_at_time == 0:
                            num_orbits -= 1
                        total_uncolored_singletons = num_extra_nodes + \
                           num_uncolored_singleton_nodes_at_time
                        d_factor = total_uncolored_singletons
                        divisor = 1.0
                        for _ in range(0, num_extra_nodes):
                            divisor *= d_factor
                            d_factor -= 1
                        num_automorphisms = num_automorphisms / divisor

                    got_canonical_order = False

            current_result_type = self.result_list[current_result][0]
            current_result_value = self.result_list[current_result][1]

            if current_result_type == self.AUTOMORPHISMS:
                raise ValueError("Error! get_automorphisms() not implemented.")
                    # ...also, remember that this data is ABOVE the 'header.'

            elif current_result_type == self.NUM_AUTOMORPHISM_ORBITS:
                current_result_value.__set_result__(num_orbits)

            elif current_result_type == self.NUM_AUTOMORPHISMS:
                current_result_value.__set_result__(num_automorphisms)

            elif current_result_type == self.RUNTIME:
                # Line was at header. Go down one.
                line_idx = last_header_line_idx + 1
                time_idx = 0
                while lines[line_idx][time_idx] != "time":
                    time_idx += 1
                time_idx += 1  # skip the = sign

                res_str = ""
                while time_idx < len(lines[line_idx]):
                    res_str += lines[line_idx][time_idx] + " "
                    time_idx += 1
                current_result_value.__set_result__(res_str)

            elif current_result_type == self.CANONICAL_ORDER:
                line_idx = last_header_line_idx + 2
                idxs = []
                while ':' not in lines[line_idx]:
                    idxs += [int(s) for s in lines[line_idx]]
                    line_idx += 1

                filtered_idxs = []
                for idx in idxs:
                    if idx in node_indices_at_time:
                        filtered_idxs.append(idx)
                node_order = [self.idx_to_node_maps[target_run][idx] for \
                    idx in filtered_idxs]
                current_result_value.__set_result__(node_order)

                got_canonical_order = True

            elif current_result_type == self.AUTOMORPHISM_ORBITS:
                line_start_idx = last_header_line_idx + 2
                if got_canonical_order:
                    while ':' not in lines[line_start_idx]:
                        line_start_idx += 1
                    line_start_idx += self.max_num_nodes

                line_end_idx = line_start_idx + 1
                while line_end_idx < len(lines) and \
                        ((lines[line_end_idx][0][0] in \
                            ['0','1','2','3','4','5','6','7,','8','9']) or (\
                          lines[line_end_idx][0][-2:] == ");")) and \
                        'orbits;' not in lines[line_end_idx]:
                    line_end_idx += 1

                relevant_lines = lines[line_start_idx : line_end_idx]

                one_big_line = []
                for line in relevant_lines:
                    one_big_line += line

                orbits = []
                big_line_idx = 0
                while big_line_idx < len(one_big_line):
                    orbit = []
                    last_value = False
                    while big_line_idx < len(one_big_line) and not last_value:
                        if one_big_line[big_line_idx][0] == "(":
                            big_line_idx += 1
                            break
                        elif one_big_line[big_line_idx][-1] == ";":
                            value = one_big_line[big_line_idx][:-1]
                            last_value = True
                        else:
                            value = one_big_line[big_line_idx]

                        if ":" in value:
                            s = value.split(":")
                            orbit += [i for i in range(int(s[0]), int(s[1])+1)]
                        else:
                            orbit.append(int(value))
                        big_line_idx += 1

                    filtered_orbit = []
                    for i in orbit:
                        if i in node_indices_at_time:
                            filtered_orbit.append(\
                                self.idx_to_node_maps[target_run][i])
                    orbits.append(filtered_orbit)

                current_result_value.__set_result__(orbits)

            else:
                raise ValueError("Error! Unknown result type %d." % \
                        current_result_type)

            current_result += 1

    def __input_start_graph_with_enough_extra_blank_nodes__(self):
        # Add start graph to session
        #
        # Example input for triangle with dangler from node 1
        #   n=4
        #   g ; 0 : 1 2 ; 1 : 2 3 .
        self.session_init_lines.append("n=%d" % self.max_num_nodes)

        graph_line = "g "
        for node_idx in range(0, self.num_start_nodes):
            node = self.start_nodes_list[node_idx]

            if self.directed:
                # successors means neighbors this node points to
                neighbors = list(self.start_graph.successors(node))
            else:
                full_neighbors = list(self.start_graph.neighbors(node))
                neighbors = []
                for n in full_neighbors:
                    if self.start_node_to_idx_map[n] > node_idx:
                        neighbors.append(n)

            if len(neighbors) > 0:
                graph_line += "; %d : " % node_idx
                for n in neighbors:
                    graph_line += "%d " % self.start_node_to_idx_map[n]

        graph_line += "."
        self.session_init_lines.append(graph_line)

class NautyTracesResult:
    def __init__(self):
        self.result = None

    def __set_result__(self, r):
        self.result = r

    def get(self):
        if self.result is None:
            raise ValueError("Error! Cannot access Nauty-Traces result " + \
                   "until session is complete. Call <your session>.complete().")
        return self.result

if __name__ == "__main__":
    import networkx as nx
    print("Testing Nauty-Traces Session")
    start_graph = nx.Graph()
    start_graph.add_node("Larry")
    start_graph.add_node("Beth")
    start_graph.add_node("Sue")
    start_graph.add_edge("Beth", "Sue")
    session = NautyTracesSession(start_graph, mode="Nauty", sparse=False)
    num_aut_1 = session.get_num_automorphisms()
    orbits_1 = session.get_automorphism_orbits()
    node_order_1 = session.get_canonical_order()
    session.delete_edge("Sue", "Beth")
    session.add_node("Ben")
    session.add_edge("Ben", "Sue")
    num_aut_2 = session.get_num_automorphisms()
    orbits_2 = session.get_automorphism_orbits()
    node_order_2 = session.get_canonical_order()
    session.complete()
    print(num_aut_1.get())
    print(orbits_1.get())
    print(node_order_1.get())
    print(num_aut_2.get())
    print(orbits_2.get())
    print(node_order_2.get())
