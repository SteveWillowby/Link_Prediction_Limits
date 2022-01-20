from HC_basic_container_types import default_set, default_dict
from HC_coloring import Coloring
from HC_print_debugging import debug_print
from HC_sampling import default_sample_set
from HC_views import ListView, obj_peak

# Requires that the elements being equated are the numbers 0 through N-1.
class ProvenEquivalences:

    def __init__(self, num_elements, debug_indent=0):
        self.__size__ = num_elements
        self.__debug_indent__ = debug_indent

        # So far, nothing is proven.
        self.__num_partitions__ = None
        self.__proven_limits__ = None
        self.__session_begun__ = False

        self.__main_partition_id_to_color__ = None
        self.__main_color_to_partition_ids__ = None
        self.__main_element_to_partition_id__ = None
        self.__main_partition_id_to_elements__ = None

        self.__session_partition_id_to_B_colors__ = None
        self.__session_B_color_to_partition_ids__ = None
        self.__session_partition_id_to_A_colors__ = None
        self.__session_A_color_to_partition_ids__ = None
        self.__session_element_to_partition_id__ = None
        self.__session_partition_id_to_elements__ = None

        self.__session_colors_with_pairings__ = None

        self.__MAIN_MODE__ = 0
        self.__SESSION_MODE__ = 1

    def done(self):
        return self.__num_partitions__ == len(self.__proven_limits__.get_non_singleton_colors())

    def failed(self):
        return self.__failed__

    # Only call when proven_limits has been updated (i.e. refined).
    #
    # proven_limits is a partitioning that the real partitioning cannot be
    #   broader than. The real partitioning must be at least as refined as
    #   proven_limits.
    #
    # First run:  O(elts in non-singleton cells in proven_limits)
    # Other runs: O(elts in non-singleton cells in PREVIOUS proven_limits)
    def set_proven_limits(self, proven_limits):
        self.__failed__ = False

        first_time = self.__proven_limits__ is None
        self.__proven_limits__ = proven_limits

        # self.__main_partition_id_to_color__
        # self.__main_color_to_partition_ids__
        # self.__main_element_to_partition_id__
        # self.__main_partition_id_to_elements__

        if first_time:
            # First session ever.
            self.__main_partition_id_to_color__ = default_dict()
            self.__main_color_to_partition_ids__ = default_dict()
            self.__main_element_to_partition_id__ = default_dict()
            self.__main_partition_id_to_elements__ = default_dict()

            for color in self.__proven_limits__.get_non_singleton_colors():
                cell = self.__proven_limits__.get_cell(color)
                for element in cell:
                    self.__main_partition_id_to_color__[element] = color
                    if color not in self.__main_color_to_partition_ids__:
                        self.__main_color_to_partition_ids__[color] = default_set()
                    self.__main_color_to_partition_ids__[color].add(element)
                    self.__main_element_to_partition_id__[element] = element
                    self.__main_partition_id_to_elements__[element] = \
                                                                  default_set([element])
        else:
            # Already had session(s) with a previous proven_limits.
            #
            # Element partitioning will not change, but the color-partition
            #   mappings might change.
            #
            # Also, we might need to remove partitions entirely because the
            #   elements are now part of singleton cells.

            all_prev_elements = \
                [e for e, pid in self.__main_element_to_partition_id__.items()]
            self.__main_partition_id_to_color__ = default_dict()  # Assign this afresh.
            self.__main_color_to_partition_ids__ = default_dict()  # Assign this afresh.
            self.__main_partition_id_to_elements__ = default_dict()  # Use old values.
            for elt in all_prev_elements:
                color = self.__proven_limits__[elt]
                pid = self.__main_element_to_partition_id__[elt]
                if len(self.__proven_limits__.get_cell(color)) == 1:
                    # Element is now a singleton. Remove it.
                    del self.__main_element_to_partition_id__[elt]
                    continue

                if color not in self.__main_color_to_partition_ids__:
                    self.__main_color_to_partition_ids__[color] = default_set()
                self.__main_color_to_partition_ids__[color].add(pid)

                if pid not in self.__main_partition_id_to_elements__:
                    self.__main_partition_id_to_elements__[pid] = default_set()
                self.__main_partition_id_to_elements__[pid].add(elt)

                self.__main_partition_id_to_color__[pid] = color

        self.__num_partitions__ = len(self.__main_partition_id_to_elements__)

    # O(proven_limits)
    def start_session(self):

        self.__session_step__ = 1

        if self.__proven_limits__ is None:
            raise RuntimeError("Error! Cannot call start_session() before " + \
                               "calling set_proven_limits()")
        self.__session_begun__ = True
        self.__coloring_A__ = Coloring(self.__proven_limits__, \
                                       use_sample_sets=True)
        self.__coloring_B__ = Coloring(self.__proven_limits__, \
                                       use_sample_sets=True)
        self.__coloring_A__.set_record_point()  # The record point is used both
            # for updates and for checking consistency between A and B.

        # Copy the info in __main_... into __session...
        self.__session_partition_id_to_A_colors__ = default_dict()
        self.__session_partition_id_to_B_colors__ = default_dict()
        for pid, color in self.__main_partition_id_to_color__.items():
            self.__session_partition_id_to_A_colors__[pid] = default_set([color])
            self.__session_partition_id_to_B_colors__[pid] = default_set([color])

        self.__session_colors_with_pairings__ = default_sample_set()
        self.__session_A_color_to_partition_ids__ = default_dict()
        self.__session_B_color_to_partition_ids__ = default_dict()
        for color, pids in self.__main_color_to_partition_ids__.items():
            self.__session_A_color_to_partition_ids__[color] = default_set(pids)
            self.__session_B_color_to_partition_ids__[color] = default_set(pids)

            # The A-cell and the B-cell are the same at the start of the session
            if len(pids) > 1:
                self.__session_colors_with_pairings__.add(color)

        self.__session_element_to_partition_id__ = \
            default_dict(self.__main_element_to_partition_id__)
        self.__session_partition_id_to_elements__ = default_dict()
        for pid, elts in self.__main_partition_id_to_elements__.items():
            self.__session_partition_id_to_elements__[pid] = default_set(elts)

        # Only store pairings which will give some partition merger information.
        self.__session_pairings__ = []

        return (self.__coloring_A__, self.__coloring_B__)


    def sub_colorings_are_consistent(self):
        if not self.__session_begun__:
            raise ValueError("Error! Cannot call next_pairing() until" +\
                             " start_session() is called.")
        if self.__coloring_A__.get_num_colors() != \
                self.__coloring_B__.get_num_colors():
            debug_print("A", self.__debug_indent__ - 1)
            debug_print(len(self.__coloring_A__), self.__debug_indent__ - 1)
            debug_print(self.__session_pairings__, self.__debug_indent__ - 1)
            return False
        if self.__coloring_A__.get_num_singletons() != \
                self.__coloring_B__.get_num_singletons():
            debug_print("B", self.__debug_indent__ - 1)
            debug_print(len(self.__coloring_A__), self.__debug_indent__ - 1)
            debug_print(self.__session_pairings__, self.__debug_indent__ - 1)
            return False

        shattered = self.__coloring_A__.post_record_point_shattered_colors()
        new = self.__coloring_A__.post_record_point_colors()
        for color in shattered:
            if len(self.__coloring_A__.get_cell(color)) != \
                    len(self.__coloring_B__.get_cell(color)):
                debug_print("C", self.__debug_indent__ - 1)
                debug_print(len(self.__coloring_A__), self.__debug_indent__ - 1)
                debug_print(self.__session_pairings__, self.__debug_indent__ - 1)
                return False
        for color in new:
            if len(self.__coloring_A__.get_cell(color)) != \
                    len(self.__coloring_B__.get_cell(color)):
                debug_print("D", self.__debug_indent__ - 1)
                debug_print(len(self.__coloring_A__), self.__debug_indent__ - 1)
                debug_print(self.__session_pairings__, self.__debug_indent__ - 1)
                return False
        return True
        

    # Given two colorings, returns a pair of elements (a, b) where the following
    #   three criterion are met:
    #       1. a has the same color in coloring_A that b has in coloring_B.
    #       2. a and b have not yet been proven to be equivalent.
    #       3. The size of a and b's cells are both larger than 1.
    #
    # If no such pair is available, then a pair meeting criterion 1 and 3 is
    #   returned.
    #
    # If no such pair is available, then an error has occured. This function
    #   should only be called if coloring_A and coloring_B are not all
    #   singletons.
    def next_pairing(self):

        self.__session_step__ += 1
        if not self.__session_begun__:
            raise ValueError("Error! Cannot call next_pairing() until" +\
                             " start_session() is called.")

        # First, update candidate pairings.
        if len(self.__session_colors_with_pairings__) > 0:
            self.__session_internal_update__()

        # Re-check, then sample.
        if len(self.__session_colors_with_pairings__) > 0:

            color = self.__session_colors_with_pairings__.sample()
            A_pids = self.__session_A_color_to_partition_ids__[color]
            B_pids = self.__session_B_color_to_partition_ids__[color]
            if len(B_pids) > 1:
                elt1 = self.__coloring_A__.get_cell(color).sample()
                sampler = self.__coloring_B__
            else:
                elt1 = self.__coloring_B__.get_cell(color).sample()
                sampler = self.__coloring_A__

            elt1_pid = self.__session_element_to_partition_id__[elt1]
            elt1_partition = self.__session_partition_id_to_elements__[elt1_pid]
            elt2 = (sampler.get_cell(color) - elt1_partition).sample()

            pair = (elt1, elt2)
            self.__session_pairings__.append(pair)
        else:
            # No more pairings to be gained. Try to finish out an isomorphism.
            colors = self.__coloring_B__.get_non_singleton_colors()
            color = colors.sample()
            A_cell = self.__coloring_A__.get_cell(color)
            B_cell = self.__coloring_B__.get_cell(color)

            A_elt = A_cell.sample()
            if A_elt in B_cell:
                B_elt = B_cell.sample_excluding(A_elt)
            else:
                B_elt = B_cell.sample()
            pair = (A_elt, B_elt)

        return pair

    def session_active(self):
        return self.__session_begun__

    def end_session_in_failure(self):
        if not self.__session_begun__:
            raise ValueError("Error! Cannot call end_session_in_failure() " + \
                             "until start_session() is called.")
        self.__session_begun__ = False
        self.__failed__ = True

    # Update to add more equivalences.
    #
    # If the latest success proves that the equivalences are as broad as the
    #   proven_limits, then return True. Otherwise return False.
    def end_session_in_success(self):
        if not self.__session_begun__:
            raise ValueError("Error! Cannot call end_session_in_success() " + \
                             "until start_session() is called.")
        self.__session_begun__ = False
        self.__merge_elt_pairs__(self.__session_pairings__, \
                                 self.__MAIN_MODE__)

        return self.done()

    # O(# elements in affected cells)
    def __session_internal_update__(self):
        # Update partition info to account for partition mergers.
        recent_pairings = self.__find_recent_pairings__()
        self.__merge_elt_pairs__(recent_pairings, self.__SESSION_MODE__)

        # Update partition info to account for color splits.
        #
        # Colors may lose or gain partitions and vice versa.
        #
        # The set of elements with possible pairings may shrink.
        shattered = list(self.__coloring_A__.post_record_point_shattered_colors())
        new = self.__coloring_A__.post_record_point_colors()
        combined = shattered + new

        A_color_to_pids = self.__session_A_color_to_partition_ids__
        B_color_to_pids = self.__session_B_color_to_partition_ids__
        pid_to_A_colors = self.__session_partition_id_to_A_colors__
        pid_to_B_colors = self.__session_partition_id_to_B_colors__

        for color in new:
            A_color_to_pids[color] = default_set()
            B_color_to_pids[color] = default_set()

            # The color might be removed later; we will add it now.
            if len(self.__coloring_A__.get_cell(color)) > 1:
                self.__session_colors_with_pairings__.add(color)

        for color in combined:
            A_cell = self.__coloring_A__.get_cell(color)
            B_cell = self.__coloring_B__.get_cell(color)

            A_cell_pids = \
                default_set([self.__session_element_to_partition_id__[elt] for elt in A_cell])
            B_cell_pids = \
                default_set([self.__session_element_to_partition_id__[elt] for elt in B_cell])

            for (ctp, ptc, pids) in [(A_color_to_pids, pid_to_A_colors, A_cell_pids), \
                                     (B_color_to_pids, pid_to_B_colors, B_cell_pids)]:
                removed_pids = ctp[color] - pids
                new_pids = pids - ctp[color]
                for pid in removed_pids:
                    ctp[color].remove(pid)
                    ptc[pid].remove(color)
                for pid in new_pids:
                    ctp[color].add(pid)
                    ptc[pid].add(color)

                assert pids == ctp[color]

            if len(B_cell_pids) == 1 and len(A_cell_pids) == 1 and \
                    obj_peak(A_cell_pids) == obj_peak(B_cell_pids) and \
                    color in self.__session_colors_with_pairings__:
                self.__session_colors_with_pairings__.remove(color)

        self.__coloring_A__.clear_record_point()
        self.__coloring_A__.set_record_point()

    def __find_recent_pairings__(self):
        start_idx = len(self.__session_pairings__) - 1
        # Find what colors changed.
        ns = self.__coloring_A__.post_record_point_singletons()
        for elt1 in ns:
            color = self.__coloring_A__[elt1]
            B_cell = self.__coloring_B__.get_cell(color)
            if len(B_cell) > 1:
                raise RuntimeError("Error! Should not continue a session that"+\
                                   "has already failed.")
            elt2 = obj_peak(B_cell)
            self.__session_pairings__.append((elt1, elt2))

        return self.__session_pairings__[start_idx:]
    
    # Call BEFORE updating due to split color info.
    #
    # If main mode:
    # O(pairs + size of resulting partitions)
    #
    # If session mode:
    # Slightly more when B-colors get single partitions.
    def __merge_elt_pairs__(self, pairs, mode):
        assert mode == self.__MAIN_MODE__ or mode == self.__SESSION_MODE__
        if mode == self.__MAIN_MODE__:
            e_to_pi = self.__main_element_to_partition_id__
            pi_to_e = self.__main_partition_id_to_elements__
        else:
            e_to_pi = self.__session_element_to_partition_id__
            pi_to_e = self.__session_partition_id_to_elements__

        # Create merge-graph
        partition_neighbors = default_dict()
        for (e1, e2) in pairs:
            p1 = e_to_pi[e1]
            p2 = e_to_pi[e2]
            if p1 == p2:
                continue  # Don't need to merge already merged partitions.
            if p1 not in partition_neighbors:
                partition_neighbors[p1] = default_set()
            if p2 not in partition_neighbors:
                partition_neighbors[p2] = default_set()
            partition_neighbors[p1].add(p2)
            partition_neighbors[p2].add(p1)

        # Run DFS on all the subgraphs.
        unseen = default_set([p for p, _ in partition_neighbors.items()])
        total_nodes = len(unseen)
        seen = default_set()
        resulting_partitions = []
        while len(seen) < total_nodes:
            start = obj_peak(unseen)
            curr_result_partition = start
            stack = [n for n in partition_neighbors[start]]
            seen |= default_set(stack + [start])
            unseen -= default_set(stack + [start])
            while len(stack) > 0:
                n = stack.pop()
                for neighbor in partition_neighbors[n]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        unseen.remove(neighbor)
                        stack.append(neighbor)
                curr_result_partition = self.__merge_partitions__(\
                    curr_result_partition, n, mode=mode)
            resulting_partitions.append(curr_result_partition)

    # For debugging purposes only.
    def __check_consistency__(self, mode="full", main=False):
        assert self.__coloring_B__.get_num_colors() == self.__coloring_A__.get_num_colors()
        for i in range(0, self.__coloring_B__.get_num_colors()):
            assert len(self.__coloring_B__.get_cell(i)) == len(self.__coloring_A__.get_cell(i))

        if mode == "full":
            if not main:
                e_to_pi = self.__session_element_to_partition_id__
                pi_to_e = self.__session_partition_id_to_elements__
                pi_to_c = self.__session_partition_id_to_B_colors__
                c_to_pi = self.__session_B_color_to_partition_ids__
            else:
                e_to_pi = self.__main_element_to_partition_id__
                pi_to_e = self.__main_partition_id_to_elements__
                pi_to_c = self.__main_partition_id_to_color__
                c_to_pi = self.__main_color_to_partition_ids__

            observed_ptc = default_dict()
            observed_ctp = default_dict()
            observed_pte = default_dict()
            for elt, pid in e_to_pi.items():
                if main:
                    color = self.__proven_limits__[elt]
                else:
                    color = self.__coloring_B__[elt]

                if pid not in observed_ptc:
                    observed_ptc[pid] = default_set()
                if color not in observed_ctp:
                    observed_ctp[color] = default_set()
                if pid not in observed_pte:
                    observed_pte[pid] = default_set()
                observed_ptc[pid].add(color)
                observed_ctp[color].add(pid)
                observed_pte[pid].add(elt)
            assert len(observed_ptc) == len(pi_to_c)
            assert len(observed_ctp) == len(c_to_pi)
            assert len(observed_pte) == len(pi_to_e)
            for pid, colors in observed_ptc.items():
                if main:
                    assert colors == default_set([pi_to_c[pid]])
                else:
                    assert colors == pi_to_c[pid]
            for color, pids in observed_ctp.items():
                assert pids == c_to_pi[color]
            for pid, elts in observed_pte.items():
                if elts != pi_to_e[pid]:
                    debug_print(":(", self.__debug_indent__)
                    debug_print(pid, self.__debug_indent__)
                    debug_print(elts, self.__debug_indent__)
                    debug_print(pi_to_e[pid], self.__debug_indent__)
                assert elts == pi_to_e[pid]
            if not main:
                for color in self.__session_colors_with_pairings__:
                    B_pids = c_to_pi[color]
                    A_pids = self.__session_A_color_to_partition_ids__[color]
                    assert len(B_pids) > 1 or len(B_pids) > 1 or B_pids != A_pids

        else:
            for pid, B_colors in self.__session_partition_id_to_B_colors__.items():
                for B_color in B_colors:
                    assert pid in self.__session_B_color_to_partition_ids__[B_color]

            for B_color, pids in self.__session_B_color_to_partition_ids__.items():
                for pid in pids:
                    assert B_color in self.__session_partition_id_to_B_colors__[pid]

    # Returns the id of the result.
    #
    # If main_mode:
    # O(min(partition 1, partition 2))
    #
    # If session mode:
    # O(min(partition 1 + partition 2, elts_available_to_match))
    def __merge_partitions__(self, p1, p2, mode):

        assert mode == self.__MAIN_MODE__ or mode == self.__SESSION_MODE__
        if mode == self.__MAIN_MODE__:
            pi_to_elts = self.__main_partition_id_to_elements__
            e_to_pi = self.__main_element_to_partition_id__
            color_to_pids = self.__main_color_to_partition_ids__
            # singular
            pid_to_color = self.__main_partition_id_to_color__
        else:
            pi_to_elts = self.__session_partition_id_to_elements__
            e_to_pi = self.__session_element_to_partition_id__
            A_color_to_pids = self.__session_A_color_to_partition_ids__
            B_color_to_pids = self.__session_B_color_to_partition_ids__
            # plural
            pid_to_A_colors = self.__session_partition_id_to_A_colors__
            pid_to_B_colors = self.__session_partition_id_to_B_colors__

        assert p1 != p2
        if mode == self.__MAIN_MODE__:
            self.__num_partitions__ -= 1
        p1_set = pi_to_elts[p1]
        p2_set = pi_to_elts[p2]
        if len(p1_set) > len(p2_set):
            result_id = p1
            subsumed_id = p2
        else:
            result_id = p2
            subsumed_id = p1

        result_set = pi_to_elts[result_id]
        subsumed_set = pi_to_elts[subsumed_id]

        result_set |= subsumed_set  # pi_to_elts gets more elements.

        if mode == self.__MAIN_MODE__:
            assert self.__main_partition_id_to_elements__[result_id] == result_set

        for elt in subsumed_set:
            e_to_pi[elt] = result_id
        del pi_to_elts[subsumed_id]

        if mode == self.__MAIN_MODE__:
            # In the main coloring, two merging partitions already have the same
            #   color.
            main_color = pid_to_color[subsumed_id]
            color_to_pids[main_color].remove(subsumed_id)
            assert result_id in color_to_pids[main_color]
            del pid_to_color[subsumed_id]
        else:
            # In the A and B colorings, two merging partitions may have multiple
            #   different colors.
            for (ptc, ctp) in [(pid_to_A_colors, A_color_to_pids), \
                               (pid_to_B_colors, B_color_to_pids)]:
                for color in ptc[subsumed_id]:
                    ctp[color].remove(subsumed_id)
                    ctp[color].add(result_id)
                    ptc[result_id].add(color)
                    if len(A_color_to_pids[color]) == 1 and \
                            len(B_color_to_pids[color]) == 1 and \
                            obj_peak(A_color_to_pids[color]) == \
                                obj_peak(B_color_to_pids[color]) and \
                            color in self.__session_colors_with_pairings__:
                        self.__session_colors_with_pairings__.remove(color)
                del ptc[subsumed_id]

        return result_id
