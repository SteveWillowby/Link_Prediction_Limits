from HC_basic_container_types import default_set, default_dict
from HC_relabeling import Relabeling
from HC_sampling import default_sample_set
from HC_views import ListView, SetView, SubListView, SampleSetView, obj_peak

# Colorings can only get more refined; once two nodes have a different color,
#   they cannot have the same color again.
#
# A Coloring is only for n objects labeled 0 through n-1.
# A Coloring with k colors has the colors 0 through k-1.
class Coloring(object):

    # O(n) if c is a Coloring or Subcoloring
    # O(n) if colors already range from 0 to k-1.
    # O(n + k log k) otherwise  (where k is the number of colors in c)
    def __init__(self, c, use_sample_sets=False):
        if use_sample_sets:
            self.__set_type__ = default_sample_set
            self.__set_view_type__ = SampleSetView
        else:
            self.__set_type__ = default_set
            self.__set_view_type__ = SetView

        if type(c) is Coloring:
            self.__cells__ = [self.__set_type__(c.get_cell(i)) \
                                for i in range(0, c.get_num_cells())]
            self.__colors__ = [i for i in range(0, len(self.__cells__))]
            self.__list__ = [self.__colors__[c[i]] for i in range(0, len(c))]
        else:
            # Assume that c has len(), and [] access for 0 to len() - 1.
            #   E.g. a list
            self.__list__ = [c[i] for i in range(0, len(c))]

            colors = default_set(self.__list__)
            colors_already_indexed = True
            for color in colors:
                if color < 0 or color >= len(colors):
                    colors_already_indexed = False
                    break

            if not colors_already_indexed:
                colors = sorted(list(colors))
                old_to_new = default_dict()
                for i in range(0, len(colors)):
                    old_to_new[colors[i]] = i
                for i in range(0, len(self.__list__)):
                    old_color = self.__list__[i]
                    new_color = old_to_new[old_color]
                    self.__list__[i] = new_color

            self.__cells__ = [self.__set_type__() for _ in range(0, len(colors))]
            for i in range(0, len(self.__list__)):
                color = self.__list__[i]
                self.__cells__[color].add(i)

        self.__size__ = len(self.__list__)
        self.__num_colors__ = len(self.__cells__)

        self.__singletons__ = []  # In a canonical order.
        self.__non_singleton_colors__ = self.__set_type__()  # Not necessarily sorted.
        for i in range(0, len(self.__cells__)):
            if len(self.__cells__[i]) > 1:
                self.__non_singleton_colors__.add(i)
            else:
                for node in self.__cells__[i]:  # Only one element.
                    self.__singletons__.append(node)

        self.__record_point_set__ = False

    # No setting of colors with the [] operator.
    def __setitem__(self, node, color):
        raise ValueError("Error! Cannot edit a Coloring with the [] operator." + \
                         " Use make_singleton() or refine_with() instead.")

    # Used for the [] operator access.
    #
    # O(1)
    def __getitem__(self, node):
        return self.__list__[node]

    # O(1)
    def get_num_singletons(self):
        return len(self.__singletons__)

    # Return a non-editable view of the list of singletons. They are returned in
    #   a canonical order.
    #
    # O(1)
    def get_singletons(self):
        return ListView(self.__singletons__)

    # Return a non-editable view of the set of the colors (cells) with more
    #   than one element.
    #
    # O(1)
    def get_non_singleton_colors(self):
        return self.__set_view_type__(self.__non_singleton_colors__)

    # O(1)
    def get_num_cells(self):
        # The number of cells is just the number of colors.
        return self.__num_colors__

    # O(1)
    def get_num_colors(self):
        return self.__num_colors__

    # Returns a non-editable view of the i'th cell.
    def get_cell(self, i):
        return self.__set_view_type__(self.__cells__[i])

    # If node is already a singleton, node's color does not change.
    # O(1)
    def make_singleton(self, node):
        if self.__size__ > 1:
            r = Relabeling(Relabeling.SUB_COLLECTION_TYPE, [node])
            sc = Coloring([0])
            self.refine_with(sc, r)

    # Important: If `alt_coloring` has different element labels from this
    #   object, or if `alt_coloring` has a different (smaller) size, then this
    #   method requires that `alt_relabeling` is not None.
    #
    # `alt_relabeling` maps this object's elements ("old") to `alt_coloring`'s
    #   elements ("new")
    #
    # O(alt_coloring + k_s log k_s)  where k_s is the number of shattered cells.
    #
    # Note that in the worst case this comes out to be
    #   O(alt_coloring log alt_coloring)
    #
    # Outputs the cell numbers which experienced shattering and the new
    #   cells.
    def refine_with(self, alt_coloring, alt_relabeling=None):
        if type(alt_coloring) is not Coloring:
            raise ValueError("Error! alt_coloring must be of type Coloring.")
        if len(alt_coloring) > self.__size__:
            raise ValueError("Error! Cannot refine a Coloring with a larger" + \
                             " Coloring.")

        if alt_relabeling is None:
            if len(alt_coloring) < self.__size__:
                raise ValueError("Error! If `alt_coloring` is smaller, then" + \
                                 " you must pass an `alt_relabeling` as well.")

            # Use the Identity Relabeling
            alt_relabeling = Relabeling(Relabeling.IDENTITY_TYPE, self.__size__)
        elif len(alt_relabeling) != len(alt_coloring):
            raise ValueError("Error! If `alt_relabeling` is passed, it must" + \
                             " have the same size as `alt_coloring`.")

        # print("Refining %s with %s" % (str(self.__list__), str({alt_relabeling.new_to_old(n): alt_coloring[n] for n in range(0, len(alt_coloring))})))

        # Partition alt elements by main cell.
        #   O(alt_coloring)
        alt_elements_by_main_color = default_dict()
        # alt_colors_by_main_color_s = default_dict()
        # alt_colors_by_main_color_l = default_dict()
        num_elements_by_main_color = default_dict()
        for alt_color in range(0, alt_coloring.get_num_cells()):
            alt_cell = alt_coloring.get_cell(alt_color)
            for alt_element in alt_cell:
                main_elt = alt_relabeling.new_to_old(alt_element)
                main_color = self.__list__[main_elt]

                if main_color not in alt_elements_by_main_color:
                    alt_elements_by_main_color[main_color] = []
                alt_elements_by_main_color[main_color].append(alt_element)

        # O(k_s) -- often faster
        affected_main_colors = []  # shattered cells
        for main_color, alt_elements in alt_elements_by_main_color.items():
            num_items = len(alt_elements)
            more_than_one_color = \
                alt_coloring[alt_elements[0]] != alt_coloring[alt_elements[-1]]
            if more_than_one_color or \
                    num_items < len(self.__cells__[main_color]):

                # Record the shattered color and whether all elements were
                #   listed.
                affected_main_colors.append(\
                    (main_color, num_items == len(self.__cells__[main_color])))
                if self.__record_point_set__ and \
                        main_color <= self.__marked_max_color__:
                    self.__shattered_colors__.add(main_color)

        # O(k_s log k_s)
        affected_main_colors.sort()

        # O(alt_coloring)
        new_colors = []
        for (main_color, all_items_flag) in affected_main_colors:
            # The key insight here is alt_elements is already sorted by
            #   alt_color due to the way it was constructed.
            alt_elements = alt_elements_by_main_color[main_color]

            prev_main_elt = None
            prev_alt_c = None
            ignore = all_items_flag
            cell_size = None
            is_new_color = False
            curr_new_color = None

            for alt_elt in alt_elements:
                alt_c = alt_coloring[alt_elt]
                if (prev_alt_c is None) or prev_alt_c != alt_c:
                    if not ignore:
                        curr_new_color = self.__num_colors__
                        self.__num_colors__ += 1
                        new_colors.append(curr_new_color)
                        self.__cells__.append(self.__set_type__())

                        if cell_size == 1 and is_new_color:  # Prev new cell was a singleton.
                            assert prev_main_elt is not None
                            self.__singletons__.append(prev_main_elt)

                        is_new_color = True
                    else:
                        ignore = False
                    cell_size = 1  # The new cell has 1 element in it.
                else:
                    cell_size += 1
                    if cell_size == 2 and curr_new_color is not None:
                        # Curr new cell is a non-singleton.
                        self.__non_singleton_colors__.add(curr_new_color)

                prev_alt_c = alt_c

                if is_new_color:
                    main_elt = alt_relabeling.new_to_old(alt_elt)
                    self.__cells__[main_color].remove(main_elt)
                    self.__cells__[curr_new_color].add(main_elt)
                    self.__list__[main_elt] = curr_new_color
                    prev_main_elt = main_elt

            if cell_size == 1:  # Prev new cell was a singleton.
                assert prev_main_elt is not None
                self.__singletons__.append(prev_main_elt)

            # Last but not least, check to see if main_color is now a singleton.
            if len(self.__cells__[main_color]) == 1:
                self.__non_singleton_colors__.remove(main_color)
                self.__singletons__.append(obj_peak(self.__cells__[main_color]))

        if self.__record_point_set__:
            for nc in new_colors:
                self.__new_colors__.append(nc)

        return [c for (c, _) in affected_main_colors] + new_colors

    # O(1)
    def set_record_point(self):
        if self.__record_point_set__:
            raise RuntimeError("Error! Record point already set.")
        self.__record_point_set__ = True
        self.__new_colors__ = []
        self.__old_num_singletons__ = len(self.__singletons__)
        self.__marked_max_color__ = self.__num_colors__ - 1
        self.__shattered_colors__ = self.__set_type__()

    # O(1)
    def clear_record_point(self):
        if not self.__record_point_set__:
            raise RuntimeError("Error! Record point no set.")
        self.__record_point_set__ = False

    # O(1) -- Returns all colors which were in the coloring when
    #   set_record_point() was last called which have since been shattered.
    def post_record_point_shattered_colors(self):
        if not self.__record_point_set__:
            raise RuntimeError("Error! Cannot call post_record_point_sha" + \
                               "ttered_colors() unless a record point is set.")
        return self.__shattered_colors__

    # O(1) -- Returns all colors which were added after set_record_point()
    #   was last called.
    def post_record_point_colors(self):
        if not self.__record_point_set__:
            raise RuntimeError("Error! Cannot call post_record_point_" + \
                               "colors() unless a record point is set.")
        return self.__new_colors__

    # O(1) -- Returns all elements which became singletons after
    #   set_record_point() was last called.
    def post_record_point_singletons(self):
        if not self.__record_point_set__:
            raise RuntimeError("Error! Cannot call post_record_point_" + \
                               "singletons() unless a record point is set.")
        if len(self.__singletons__) == self.__old_num_singletons__:
            return []
        return self.__singletons__[self.__old_num_singletons__:]

    # O(1)
    def record_point_is_set(self):
        return self.__record_point_set__

    def __len__(self):
        return self.__size__

    def __iter__(self):
        return self.__list__.__iter__()

    def __str__(self):
        return self.__list__.__str__()

    def __consistency_check__(self):
        # Not sure why I added this col_dict at all. Remove from code...
        col_dict = default_dict()
        for i in range(0, len(self)):
            col_dict[i] = self[i]

        col_freq = default_dict()
        for elt, col in col_dict.items():
            if col not in col_freq:
                col_freq[col] = 0
            col_freq[col] += 1

        for col, freq in col_freq.items():
            cell = self.get_cell(col)
            assert len(cell) == freq
            for elt in cell:
                assert self[elt] == col

        total_len = 0
        for col in self.get_non_singleton_colors():
            assert len(self.get_cell(col)) > 1
            total_len += len(self.get_cell(col))
        for elt in self.get_singletons():
            assert len(self.get_cell(self[elt])) == 1
            total_len += 1
        assert total_len == len(self)


if __name__ == "__main__":
    c1 = Coloring([0, 3, 0, 2, 3, 5, 5, 5, 5])
    print("Expect: %s == %s" % (c1, [0, 2, 0, 1, 2, 3, 3, 3, 3]))
    c2 = Coloring(c1, use_sample_sets=True)
    print("Expect: %s == %s" % (c2, [0, 2, 0, 1, 2, 3, 3, 3, 3]))

    cell = c2.get_cell(2)
    for _ in range(0, 1000):
        assert cell.sample() in [1, 4]

    c1.refine_with(c2)
    print("Expect: %s == %s" % (c1, [0, 2, 0, 1, 2, 3, 3, 3, 3]))

    c3 = Coloring([0, 0, 0, 1])
    rA = Relabeling(Relabeling.SUB_COLLECTION_TYPE, [4, 5, 6, 7])
    c1.set_record_point()
    c1.refine_with(c3, alt_relabeling=rA)
    print("Expect: %s == %s" % (c1, [0, 2, 0, 1, 4, 5, 5, 6, 3]))
    print("Expect: %s == %s" % (sorted(list(c1.get_singletons())), [1, 3, 4, 7, 8]))
    print("Expect: %s == %s" % (sorted(list(c1.get_non_singleton_colors())), [0, 5]))
    assert sorted(list(c1.get_cell(0))) == [0, 2]
    assert sorted(list(c1.get_cell(1))) == [3]
    assert sorted(list(c1.get_cell(2))) == [1]
    assert sorted(list(c1.get_cell(3))) == [8]
    assert sorted(list(c1.get_cell(4))) == [4]
    assert sorted(list(c1.get_cell(5))) == [5, 6]
    assert sorted(list(c1.get_cell(6))) == [7]
    assert sorted(list(c1.post_record_point_shattered_colors())) == [2, 3]
    assert sorted(list(c1.post_record_point_colors())) == [4, 5, 6]
    assert sorted(list(c1.post_record_point_singletons())) == [1, 4, 7, 8]

    c4 = Coloring([0, 0, 0, 1, 0])
    rB = Relabeling(Relabeling.SUB_COLLECTION_TYPE, [4, 5, 6, 7, 8])
    c2.refine_with(c4, alt_relabeling=rB)
    print("Expect: %s == %s" % (c2, [0, 2, 0, 1, 4, 3, 3, 5, 3]))
