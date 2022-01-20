from HC_basic_container_types import default_set, default_dict, \
                                     SetLike, DictLike, Indexed, ContainerBase
from HC_relabeling import Relabeling
from HC_sampling import SampleSet, SampleDict, default_sample_set

# Classes:
#
# GraphView
# KHopSubGraphView
#
# ListView
# SubListView
#
# SetView
# SampleSetView
#
# RelabeledSetView
#
# DictView
# SubDictView
# SampleDictView
#
# EdgeDictView
# SubEdgeDictView

# Functions:
#
# obj_peak(obj) -- returns some element from obj

# Requires that the nodes are labeled from 0 to n-1
#
# If edge_types is not None, modifies edge_types in place.
class GraphView:

    def __init__(self, directed, nodes=None, edges=None, \
                 neighbors_list=None, edge_types=None):

        if (nodes is None and edges is not None) or \
                ((nodes is not None) and edges is None):
            raise RuntimeError("Error! `nodes` must be None if and only if " + \
                               "`edges` is also None.")
        if (nodes is None and neighbors_list is None):
            raise RuntimeError("Error! Must initialize GraphView with either" +\
                               "`nodes` and `edges` or with `neighbors_list`.")

        NE = nodes is not None

        if NE:
            # Change type of nodes if necessary.
            nodes = default_set(nodes, copy=False)
            self.__n__ = len(nodes)
            self.__m__ = len(edges)
            for n in nodes:
                if n < 0 or n >= len(nodes):
                    raise RuntimeError("Error! Nodes in a GraphView must be " +\
                                       "labeled from 0 to n-1.")
        else:
            self.__n__ = len(neighbors_list)
            self.__m__ = 0
            for neighbors in neighbors_list:
                self.__m__ += len(neighbors)
            if not directed:
                assert self.__m__ % 2 == 0
                self.__m__ /= 2

        # We give multiple names to the functions in case the expected interface
        #   is different.
        self.successors = self.out_neighbors
        self.predecessors = self.in_neighbors

        self.__directed__ = directed
        if directed:
            self.__in_neighbors__ = [default_set() for _ in range(0, self.__n__)]
            if NE:
                self.__out_neighbors__ = \
                    [default_set() for n in range(0, self.__n__)]
                for (a, b) in edges:
                    self.__out_neighbors__[a].add(b)
                    self.__in_neighbors__[b].add(a)
            else:
                self.__out_neighbors__ = neighbors_list
                for n in range(0, len(self.__out_neighbors__)):
                    for neighbor in self.__out_neighbors__[n]:
                        self.__in_neighbors__[neighbor].add(n)

            # Only store if asked for
            self.__neighbors__ = [None for _ in range(0, self.__n__)]
        else:
            if NE:
                self.__neighbors__ = [default_set() for n in nodes]
                for (a, b) in edges:
                    self.__neighbors__[a].add(b)
                    self.__neighbors__[b].add(a)
            else:
                self.__neighbors__ = neighbors_list

        self.__has_edge_types__ = edge_types is not None
        if self.__has_edge_types__:
            self.__edge_types__ = EdgeDictView(edge_types, directed)

    def is_directed(self):
        return self.__directed__

    def neighbors(self, node):
        n = self.__neighbors_func__(node)
        if type(n) is SetView:
            return n
        return SetView(n)

    def __neighbors_func__(self, node):
        if self.__directed__:
            if self.__neighbors__[node] is None:
                self.__neighbors__[node] = \
                    self.__out_neighbors__[node] | self.__in_neighbors__[node]
        return self.__neighbors__[node]

    def out_neighbors(self, node):
        if not self.__directed__:
            raise RuntimeError("Error! Called out_neighbors() on an " + \
                               "undirected GraphView.")
        o_n = self.__out_neighbors__[node]
        if type(o_n) is SetView:
            return o_n
        return SetView(o_n)

    def in_neighbors(self, node):
        if not self.__directed__:
            raise RuntimeError("Error! Called in_neighbors() on an " + \
                               "undirected GraphView.")
        i_n = self.__in_neighbors__[node]
        if type(i_n) is SetView:
            return i_n
        return SetView(i_n)

    def num_nodes(self):
        return self.__n__

    def num_edges(self):
        return self.__m__

    def nodes(self):
        return range(0, self.__n__)

    def has_edge_types(self):
        return self.__has_edge_types__

    def edge_type(self, a, b):
        return self.__edge_types__[(a, b)]

    def edge_types(self):
        return self.__edge_types__


class KHopSubGraphView(GraphView):

    def __init__(self, graph_view, nodes, k, restricting_coloring=None):
        (inner, outer, keep_outer_edges) = \
            self.__k_hop_nodes__(graph_view, nodes, k, restricting_coloring)
        nodes = list(inner) + list(outer)  # The inner nodes will have lower
                                           #    new labels than the outer nodes.
                                           # However, this fact should not matter.
        self.__relabeling__ = Relabeling(Relabeling.SUB_COLLECTION_TYPE, nodes)

        self.__directed__ = graph_view.is_directed()
        self.__n__ = len(nodes)
        self.__m__ = 0

        R_C = restricting_coloring is not None
        if R_C or keep_outer_edges:
            all_nodes = default_set(nodes)

        # The `outer_AND_set` is the collection of all nodes that outer nodes
        #   can have neighbors in.
        if keep_outer_edges:
            outer_AND_set = all_nodes
        else:
            outer_AND_set = inner

        if self.__directed__:
            self.__out_neighbors__ = [None for _ in range(0, self.__n__)]
            self.__in_neighbors__ = [None for _ in range(0, self.__n__)]
            self.__neighbors__ = [None for _ in range(0, self.__n__)]
            for main_node in inner:
                sub_node = self.__relabeling__.old_to_new(main_node)
                o_n = graph_view.out_neighbors(main_node)
                if R_C:
                    combo = o_n & all_nodes
                    if len(combo) < len(o_n):
                        o_n = combo
                self.__out_neighbors__[sub_node] = \
                    RelabeledSetView(o_n, self.__relabeling__)
                self.__m__ += len(self.__out_neighbors__[sub_node])

                i_n = graph_view.in_neighbors(main_node)
                if R_C:
                    combo = i_n & all_nodes
                    if len(combo) < len(i_n):
                        i_n = combo
                self.__in_neighbors__[sub_node] = \
                    RelabeledSetView(i_n, self.__relabeling__)

                n = graph_view.neighbors(main_node)
                if R_C:
                    combo = n & all_nodes
                    if len(combo) < len(n):
                        n = combo
                self.__neighbors__[sub_node] = \
                    RelabeledSetView(n, self.__relabeling__)

            for main_node in outer:
                sub_node = self.__relabeling__.old_to_new(main_node)
                self.__out_neighbors__[sub_node] = \
                    RelabeledSetView(graph_view.out_neighbors(main_node) & outer_AND_set, \
                                     self.__relabeling__)
                self.__m__ += len(self.__out_neighbors__[sub_node])

                self.__in_neighbors__[sub_node] = \
                    RelabeledSetView(graph_view.in_neighbors(main_node) & outer_AND_set, \
                                     self.__relabeling__)
        else:
            self.__neighbors__ = [None for _ in range(0, self.__n__)]
            for main_node in inner:
                sub_node = self.__relabeling__.old_to_new(main_node)
                n = graph_view.neighbors(main_node)
                if R_C:
                    combo = n & all_nodes
                    if len(combo) < len(n):
                        n = combo
                self.__neighbors__[sub_node] = \
                    RelabeledSetView(n, self.__relabeling__)
                self.__m__ += len(self.__neighbors__[sub_node])
            for main_node in outer:
                sub_node = self.__relabeling__.old_to_new(main_node)
                self.__neighbors__[sub_node] = \
                    RelabeledSetView(graph_view.neighbors(main_node) & outer_AND_set, \
                                     self.__relabeling__)
                self.__m__ += len(self.__neighbors__[sub_node])
            assert self.__m__ % 2 == 0
            old_m = self.__m__
            self.__m__ = int(self.__m__ / 2)
            assert self.__m__ * 2 == old_m

        self.__has_edge_types__ = graph_view.has_edge_types()
        if self.__has_edge_types__:
            self.__edge_types__ = SubEdgeDictView(graph_view.edge_types(), \
                                                  self.__relabeling__)

    def get_node_relabeling(self):
        return self.__relabeling__

    # Returns (inner, outer) where inner is all the nodes within k-1 hops
    #   and outer is all the nodes with a shortest distance to node of k hops.
    #   Thus: inner & outer == {}
    #
    # Returns (inner, outer, keep_outer_edges), where things work as follows:
    #
    #   `inner` contains all nodes within [ceiling(k / 2) - 1] hops.
    #   `outer` contains all nodes within exactly ceiling(k / 2) hops.
    #       (i.e. inner & outer == {})
    #   `keep_outer_edges` is set to True iff k is even.
    def __k_hop_nodes__(self, graph_view, nodes, k, coloring):
        if k <= 0:
            raise ValueError("Error! k_hop_nodes() requires k >= 1.")

        keep_outer_edges = k % 2 == 0
        hops = int((k + 1) / 2)  # Divide by 2, rounding up.

        inner = default_set()
        new_outer = default_set(nodes)
        for _ in range(0, hops):
            inner |= new_outer
            old_outer = new_outer
            new_outer = default_set()
            for n in old_outer:
                # Don't add neighbors of singletons.
                if coloring is None or \
                        len(coloring.get_cell(coloring[n])) > 1:
                    new_outer |= graph_view.neighbors(n) - inner
        return (inner, new_outer, keep_outer_edges)

class ListView(Indexed):

    def __init__(self, obj):
        self.__obj__ = obj

    def __getitem__(self, key):
        return self.__obj__[key]

    def __setitem__(self, key, value):
        raise ValueError("Error! Cannot edit a ListView of a collection.")

    def __len__(self):
        return len(self.__obj__)

    def __str__(self):
        return self.__obj__.__str__()

class SetView(ContainerBase):

    def __init__(self, obj):
        self.__obj__ = obj

    def __len__(self):
        return len(self.__obj__)

    def __iter__(self):
        return self.__obj__.__iter__()

    def __contains__(self, value):
        return self.__obj__.__contains__(value)

    def __str__(self):
        return self.__obj__.__str__()

    def __getitem__(self, idx):
        return self.__obj__[idx]

    def is_indexed(self):
        if type(self.__obj__) is set:
            return False
        return self.__obj__.is_indexed()

    def is_sorted(self):
        if type(self.__obj__) is set:
            return False
        return self.__obj__.is_sorted()

    # O(self + other)
    def __or__(self, other):
        return __views_py_safe_or__(self, other)

    # def __ror__(self, other):
    #     return __views_py_safe_or__(self, other)

    # O(min(self, other))
    def __and__(self, other):
        return __views_py_safe_and__(self, other)

    # def __rand__(self, other):
    #     return __views_py_safe_and__(self, other)

    # O(self)
    def __sub__(self, other):
        return __views_py_safe_sub__(self, other)

    # O(other)
    # def __rsub__(self, other):
    #     return __views_py_safe_sub__(other, self)

class SampleSetView(ContainerBase):

    def __init__(self, obj):
        self.__obj__ = obj
        self.sample = obj.sample
        self.sample_excluding = obj.sample_excluding

    def is_indexed(self):
        return self.__obj__.is_indexed()

    def is_sorted(self):
        return self.__obj__.is_sorted()

    # O(self + other)
    def __or__(self, other):
        s = self.__obj__ | other
        return default_sample_set(s, copy=False)

    # O(min(self, other))
    def __and__(self, other):
        s = self.__obj__ & other
        return default_sample_set(s, copy=False)

    # O(self)
    def __sub__(self, other):
        s = self.__obj__ - other
        return default_sample_set(s, copy=False)

    # O(other)
    # def __rsub__(self, other):
    #     s = other - self.__obj__
    #     return self.__return_func__(s)

    # O(1)
    def __iter__(self):
        return self.__obj__.__iter__()

    # O(1)
    def __contains__(self, value):
        return self.__obj__.__contains__(value)

    # O(1)
    def __len__(self):
        return len(self.__obj__)

class SampleDictView(ContainerBase):

    def __init__(self, obj):
        self.__obj__ = obj
        self.sample = obj.sample
        self.sample_excluding = obj.sample_excluding
        self.items = obj.items

    # O(1)
    def __getitem__(self, key):
        return self.__obj__.__getitem__(key)

    # O(1)
    def __iter__(self):
        return self.__obj__.__iter__()

    # O(1)
    def __contains__(self, value):
        return self.__obj__.__contains__(value)

    # O(1)
    def __len__(self):
        return len(self.__obj__)

def __views_py_safe_or__(A, B):
    # print("Views-Safe or (%s, %s)" % (A, B))
    s = default_set(A)
    for elt in B:
        s.add(elt)
    return s

def __views_py_safe_and__(A, B):
    # print("Views-Safe and (%s, %s)" % (A, B))
    s = default_set()
    if len(A) < len(B):
        for elt in A:
            if elt in B:
                s.add(elt)
    else:
        for elt in B:
            if elt in A:
                s.add(elt)
    return s

def __views_py_safe_sub__(A, B):
    # print("Views-Safe sub (%s, %s)" % (A, B))
    s = default_set()
    for elt in A:
        if elt not in B:
            s.add(elt)
    return s

# For this class, obj contains a SUBSET of elements in `relabeling`
#
# Thus, obj is really a sub-sub-set
class RelabeledSetView(ContainerBase):

    def __init__(self, obj, relabeling):
        if type(relabeling) is not Relabeling:
            raise TypeError("Error! `relabeling` must be of type " + \
                             "Relabeling when initializing a RelabeledSetView.")

        self.__relabeling__ = relabeling
        self.__obj__ = obj
        self.__size__ = len(self.__obj__)

    def is_indexed(self):
        if type(self.__obj__) is set:
            return False
        return self.__obj__.is_indexed()

    def is_sorted(self):
        if type(self.__obj__) is set:
            return False
        return self.__obj__.is_sorted()

    def __contains__(self, key):
        return type(key) is int and 0 <= key and key < self.__size__

    # Assumes `other` is in the relabeled space. Returns a set of values in the
    #   relabeled space.
    #
    # O(self + other)
    def __or__(self, other):
        return __views_py_safe_or__(self, other)

    # Assumes `other` is in the relabeled space. Returns a set of values in the
    #   relabeled space.
    #
    # O(min(self, other))
    def __and__(self, other):
        return __views_py_safe_and__(self, other)

    # Assumes `other` is in the relabeled space. Returns a set of values in the
    #   relabeled space.
    #
    # O(self)
    def __sub__(self, other):
        return __views_py_safe_sub__(self, other)

    # Assumes `other` is in the relabeled space. Returns a set of values in the
    #   relabeled space.
    #
    # O(other)
    # def __rsub__(self, other):
    #     return __views_py_safe_sub__(other, self)
        

    # Iterates over relabeled elements.
    def __iter__(self):
        return __IteratorForRelabeledSetView__(self)

    # Returns original labels of elements.
    def orig_elements(self):
        return [v for v in self.__obj__]

    def __contains__(self, idx):
        return self.__relabeling__.new_to_old(idx) in self.__obj__

    def __len__(self):
        return self.__size__

    def __str__(self):
        return "RelabeledSetView(%s)" % ([v for v in self])

class __IteratorForRelabeledSetView__:

    def __init__(self, ssv):
        self.__ssv__ = ssv
        self.__iter_obj__ = ssv.__obj__.__iter__()
        self.__length_hint__ = self.__iter_obj__.__length_hint__

    def __next__(self):
        key = self.__iter_obj__.__next__()
        return self.__ssv__.__relabeling__.old_to_new(key)

class DictView(ContainerBase):

    def __init__(self, obj):
        self.__obj__ = obj

    def __getitem__(self, key):
        return self.__obj__[key]

    def __setitem__(self, key, value):
        raise ValueError("Error! Cannot edit a DictView of a collection.")

    def __iter__(self):
        return self.__obj__.__iter__()

    def __contains__(self, key):
        return self.__obj__.__contains__(key)

    def items(self):
        return self.__obj__.items()

    def __str__(self):
        return self.__obj__.__str__()

# Uses the key relabeling
class SubListView(Indexed):

    def __init__(self, obj, relabeling):
        if type(relabeling) is not Relabeling:
            raise ValueError("Error! `relabeling` must be of type " + \
                             "Relabeling when initializing a SubListView.")
        self.__obj__ = obj
        self.__relabeling__ = relabeling
        self.__size__ = len(relabeling)

    def __getitem__(self, idx):
        return self.__obj__[self.__relabeling__.new_to_old(idx)]

    def __setitem__(self, key, value):
        raise ValueError("Error! Cannot edit a SubListView of a collection.")

    def __len__(self):
        return self.__size__

    def __str__(self):
        return str([self.__getitem__[i] for i in range(0, self.__size__)])

# NOTE: may modify edge_dict in place if not directed.
class EdgeDictView(ContainerBase):

    def __init__(self, edge_dict, directed):
        self.__directed__ = directed
        self.__edge_dict__ = edge_dict
        if not directed:
            edges = [edge for edge, _ in edge_dict.items()]
            for (a, b) in edges:
                if b < a:
                    t = self.__edge_dict__[(a, b)]
                    del self.__edge_dict__[(a, b)]
                    self.__edge_dict__[(b, a)] = t

    def __getitem__(self, key):
        if not self.__directed__:
            key = (min(key[0], key[1]), max(key[0], key[1]))
        return self.__edge_dict__[key]

    def __setitem__(self, key, value):
        raise ValueError("Error! Cannot edit an EdgeDictView.")

    def items(self):
        return self.__edge_dict__.items()

    def __contains__(self, key):
        if not self.__directed__:
            key = (min(key[0], key[1]), max(key[0], key[1]))
        return key in self.__edge_dict__

class SubEdgeDictView(ContainerBase):

    def __init__(self, edge_dict_view, relabeling):
        if type(relabeling) is not Relabeling:
            raise ValueError("Error! `relabeling` must be of type " + \
                         "Relabeling when initializing a SubEdgeDictView.")
        self.__edv__ = edge_dict_view
        self.__relabeling__ = relabeling

    def __setitem__(self, key, value):
        raise ValueError("Error! Cannot edit a SubEdgeDictView.")

    def __getitem__(self, key):
        (a, b) = key
        a = self.__relabeling__.new_to_old(a)
        b = self.__relabeling__.new_to_old(b)
        return self.__edv__[(a, b)]

    def items(self):
        return [((self.__relabeling__.old_to_new(a), \
                  self.__relabeling__.old_to_new(b)), \
                 value) for (a, b), value in self.__edv__.items()]

    def __contains__(self, key):
        (a, b) = key
        a = self.__relabeling__.new_to_old(a)
        b = self.__relabeling__.new_to_old(b)
        return (a, b) in self.__edv__

# O(1), hopefully, assuming the base set class is implemented nicely.
def obj_peak(obj):
    if len(obj) == 0:
        raise RuntimeError("Error! Cannot obj_peak() on an empty object.")
    for v in obj:
        return v

if __name__ == "__main__":
    # s = default_set(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    sub_set_list = ['a', 'b', 'e', 'g']
    r = Relabeling(Relabeling.SUB_COLLECTION_TYPE, sub_set_list)

    SS1 = default_set(['a', 'e'])
    SS2 = default_set(['a', 'b'])

    SSV1 = SetView(SS1)
    SSV2 = SetView(SS2)
    RSSV1 = RelabeledSetView(SSV1, r)
    RSSV2 = RelabeledSetView(SSV2, r)

    print(default_set(RSSV1))

    print(RSSV1)
    print(RSSV2)
    print(RSSV1 | RSSV2)
    print(RSSV1 & default_set([2]))
    print(RSSV1 & default_set([2, 3]))
    print(RSSV1 & default_set([3]))
    print(RSSV1 & RSSV2)
    s = default_set([2])
    s |= RSSV1 - RSSV2


    print("------------------------------------")
    nodes = [0, 1, 2, 3, 4, 5, 6, 7]
    edges = [(0, 1), (0, 2), (2, 1), (1, 3), (3, 2), (0, 4), \
             (4, 5), (4, 6), (6, 5), (5, 7), (7, 6), (3, 7)]
    edge_types = default_dict({(0, 1): "corner-mid", \
                  (0, 2): "corner-mid", \
                  (1, 3): "corner-mid", \
                  (2, 3): "corner-mid", \
                  (4, 5): "corner-mid", \
                  (4, 6): "corner-mid", \
                  (5, 7): "corner-mid", \
                  (6, 7): "corner-mid", \
                  (0, 4): "corner-corner", \
                  (3, 7): "corner-corner", \
                  (1, 2): "mid-mid", \
                  (5, 6): "mid-mid"})
    GV = GraphView(directed=False, nodes=nodes, edges=edges, edge_types=edge_types)
    print(GV.neighbors(0))
    print(GV.neighbors(3))
    print(GV.neighbors(6))
    assert GV.edge_type(2, 1) == "mid-mid"
    assert GV.edge_type(1, 2) == "mid-mid"
    assert GV.edge_type(0, 1) == "corner-mid"
    assert GV.edge_type(4, 0) == "corner-corner"
    SGV = KHopSubGraphView(GV, [0], 3)
    print(SGV.neighbors(0))
    print(SGV.neighbors(3))
    print(SGV.neighbors(6))
    print(SGV.num_nodes())
    print(SGV.num_edges())
    SSGV = KHopSubGraphView(SGV, [6], 1)
    print(SSGV.num_nodes())
    print(SSGV.num_edges())

    import random
    from basic_container_types import Set
    from list_containers import ListSet

    list_min_size = 0
    list_max_size = 10
    val_min = -7
    val_max = 22
    num_tests = 100000
    for i in range(0, num_tests):
        l1 = [random.randint(val_min, val_max) for __ in \
                range(0, random.randint(list_min_size, list_max_size))]
        l2 = [random.randint(val_min, val_max) for __ in \
                range(0, random.randint(list_min_size, list_max_size))]

        s1 = set(l1)
        s2 = set(l2)

        ls1 = [Set(l1), ListSet(l1), SetView(Set(l1)), SetView(ListSet(l1))][random.randint(0, 3)]
        ls2 = [Set(l2), ListSet(l2), SetView(Set(l2)), SetView(ListSet(l2))][random.randint(0, 3)]

        if type(ls1) is ListSet:
            assert sorted(list(s1)) == ls1.__elts__
        if type(ls2) is ListSet:
            assert sorted(list(s2)) == ls2.__elts__

        operation = ["|=", "|", "&=", "&", "-=", "-", "add", "remove", "discard", "in"][random.randint(0, 9)]
        if operation == "|=":
            res_s = set(s1)
            res_s |= s2
            res_ls = type(ls1)(ls1)
            res_ls |= ls2
        elif operation == "|":
            res_s = s1 | s2
            res_ls = ls1 | ls2
        elif operation == "&=":
            res_s = set(s1)
            res_s &= s2
            res_ls = type(ls1)(ls1)
            res_ls &= ls2
        elif operation == "&":
            res_s = s1 & s2
            res_ls = ls1 & ls2
        elif operation == "-=":
            res_s = set(s1)
            res_s -= s2
            res_ls = type(ls1)(ls1)
            res_ls -= ls2
        elif operation == "-":
            res_s = s1 - s2
            res_ls = ls1 - ls2
        elif operation == "add":
            if type(ls1) is SetView:
                continue
            v = random.randint(val_min, val_max)
            res_s = set(s1)
            res_ls = type(ls1)(ls1)
            res_s.add(v)
            res_ls.add(v)
        elif operation == "remove" and len(l1) > 0:
            if type(ls1) is SetView:
                continue
            v = l1[random.randint(0, len(l1) - 1)]
            res_s = set(s1)
            res_ls = type(ls1)(ls1)
            res_s.remove(v)
            res_ls.remove(v)
        elif operation == "discard":
            if type(ls1) is SetView:
                continue
            v = random.randint(val_min, val_max)
            res_s = set(s1)
            res_ls = type(ls1)(ls1)
            res_s.discard(v)
            res_ls.discard(v)
        elif operation == "in" and len(l1) > 0:
            for v in range(min(l1) - 1, max(l1) + 2):
                assert (v in ls1) == (v in l1)
            continue

        assert len(res_s) == len(res_ls)

        res_s = sorted(list(res_s))
        if type(res_ls) is Set or type(res_ls) is set:
            res_ls = sorted(list(res_ls))
        else:
            res_ls = list(res_ls)

        if res_s != res_ls:
            print("Error!")
            if operation in ["add", "remove", "discard"]:
                pass
            else:
                print("%s %s %s --> %s VS. %s %s %s --> %s" % \
                    (s1, operation, s2, res_s, ls1, operation, ls2, res_ls))
            raise RuntimeError(":(")
