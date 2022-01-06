from basic_container_types import Sorted, Indexed, SetLike, DictLike, \
                                    __BCT_generic_set_or__, \
                                    __BCT_generic_set_and__, \
                                    __BCT_generic_set_sub__, \
                                    __BCT_generic_set_ior__, \
                                    __BCT_generic_set_iand__, \
                                    __BCT_generic_set_isub__

# A set class. It uses lists rather than hash tables to save space.
#   However, this can incur large computation costs for certain operations.
#
# These sets are most efficient when elements are added to them in increasing
#   order.
#
# NOTE: These sets can only store elements which are orderd.
class ListSet(SetLike, Indexed, Sorted):

    def __init__(self, elts=None):
        if elts is None:
            self.__elts__ = []
        else:
            self.__elts__ = list(elts)
            needs_sorting = False
            for i in range(1, len(self.__elts__)):
                if self.__elts__[i - 1] > self.__elts__[i]:
                    needs_sorting = True
                    break
            if needs_sorting:
                self.__elts__.sort()
            has_duplicates = False
            for i in range(1, len(self.__elts__)):
                if self.__elts__[i - 1] == self.__elts__[i]:
                    has_duplicates = True
            if has_duplicates:
                l = self.__elts__
                last_elt = l[0]
                self.__elts__ = list()
                self.__elts__.append(last_elt)
                for elt in l:
                    if elt != last_elt:
                        self.__elts__.append(elt)
                    last_elt = elt


    def __contains__(self, elt):
        (result, _) = self.__bin_search__(elt)
        return result

    def __getitem__(self, idx):
        return self.__elts__[idx]

    def __len__(self):
        return len(self.__elts__)

    def __iter__(self):
        return self.__elts__.__iter__()

    # O(1) if elt needs to be added and will be the largest element of the set.
    # O(log n) if elt already in set.
    # O(n) if elt needs to be added elsewhere.
    def add(self, elt):
        (found, idx) = self.__bin_search__(elt)
        if not found:
            if idx == len(self.__elts__):
                self.__elts__.append(elt)
            else:
                self.__elts__.insert(idx, elt)

    # O(1) if element removed is the largest element in the set.
    # O(n) otherwise.
    def remove(self, elt):
        (found, idx) = self.__bin_search__(elt)
        if not found:
            raise ValueError("Error! Element %s not in ListSet %s." % \
                    (str(elt), str((self, self.__elts__))))
        self.__remove_helper__(idx)

    def discard(self, elt):
        (found, idx) = self.__bin_search__(elt)
        if not found:
            return
        self.__remove_helper__(idx)

    def __eq__(self, other):

        if len(self.__elts__) != len(other):
            return False
        if other.is_sorted() and other.is_indexed():
            for i in range(0, len(self.__elts__)):
                if self.__elts__[i] != other[i]:
                    return False
            return True
        elif other.is_sorted():
            return self.__elts__ == list(other)
        return self.__elts__ == sorted(list(other))

    def __ge__(self, other):
        raise RuntimeError("Error! Operator >= not implemented for ListSet")

    def __le__(self, other):
        raise RuntimeError("Error! Operator <= not implemented for ListSet")

    # O(self + other) assuming both are ListSets
    def __or__(self, other):
        if (not other.is_sorted()) or not other.is_indexed():
            return __BCT_generic_set_or__(self, other)

        new_set = type(self)()
        i = 0
        j = 0
        while i < len(self) or j < len(other):
            if i == len(self):
                new_set.add(other[j])
                j += 1
            elif j == len(other):
                new_set.add(self[i])
                i += 1
            elif self[i] < other[j]:
                new_set.add(self[i])
                i += 1
            else:
                new_set.add(other[j])
                j += 1

        return new_set

    # O(self + other) assuming both are ListSets
    def __ior__(self, other):
        if (not other.is_sorted()) or not other.is_indexed():
            return __BCT_generic_set_ior__(self, other)

        if len(other) == 0:
            return self
        if len(self) == 0:
            self.__elts__ = [e for e in other]
            return self
        if self[-1] < other[0]:
            self.__elts__ += [e for e in other]
            return self
        if self[0] > other[-1]:
            self.__elts__ = [e for e in other] + self.__elts__
            return self

        # Scan through self.__elts__ and other to see if other contains any new
        #   elements. If other contains a new element, put it into the correct
        #   place in self.__elts__ and shift the old content in self.__elts__
        #   into addendum.
        #
        # The process implicitly guarantees that:
        #   if k < len(addendum) and i < len(self) then addendum[k] < self[i]
        addendum = []
        i = 0
        j = 0
        k = 0
        # NOTE: i increments every iteration at bottom of loop.
        while i < len(self) or j < len(other) or k < len(addendum):
            if i == len(self) and j == len(other):
                self.__elts__.append(addendum[k])
                k += 1
            elif i == len(self) and k == len(addendum):
                self.__elts__.append(other[j])
                j += 1
            elif j == len(other) and k == len(addendum):
                break
            elif i == len(self):
                if addendum[k] == other[j]:
                    self.__elts__.append(addendum[k])
                    k += 1
                    j += 1
                elif addendum[k] < other[j]:
                    self.__elts__.append(addendum[k])
                    k += 1
                else:
                    self.__elts__.append(other[j])
                    j += 1
            elif j == len(other):
                if len(addendum) > 0:
                    addendum.append(self.__elts__[i])
                    self.__elts__[i] = addendum[k]
                    k += 1
                else:
                    break
            elif k == len(addendum):
                if self.__elts__[i] == other[j]:
                    j += 1
                elif self.__elts__[i] < other[j]:
                    pass
                else:
                    addendum.append(self.__elts__[i])
                    self.__elts__[i] = other[j]
                    j += 1
            else:
                # i < len(self), j < len(other), k < len(addendum)
                addendum.append(self.__elts__[i])
                if addendum[k] == other[j]:
                    self.__elts__[i] = addendum[k]
                    k += 1
                    j += 1
                elif addendum[k] < other[j]:
                    self.__elts__[i] = addendum[k]
                    k += 1
                else:
                    self.__elts__[i] = other[j]
                    j += 1
            i += 1

        return self

    # TODO: Optimize
    # NOTE: Even though the big-O is bad, the left-to-right list reads may be
    #   fast.
    # O(self + other) assuming both are ListSets
    def __and__(self, other):
        if (not other.is_sorted()) or not other.is_indexed():
            return __BCT_generic_set_and__(self, other)

        new_set = type(self)()
        i = 0
        j = 0
        while i < len(self) and j < len(other):
            if self[i] == other[j]:
                new_set.add(self[i])
                i += 1
                j += 1
            elif self[i] < other[j]:
                i += 1
            else:
                j += 1
        return new_set

    # O(self + other) assuming both are ListSets
    def __iand__(self, other):
        if (not other.is_sorted()) or not other.is_indexed():
            return __BCT_generic_set_iand__(self, other)

        if len(self) == 0 or len(other) == 0:
            self.__elts__ = []
            return self
        if self[-1] < other[0] or other[-1] < self[0]:
            self.__elts__ = []
            return self

        # The use of the shift allows this to be in-place.
        #   I am unsure if that is worth the effort.
        i = 0
        j = 0
        shift = 0
        while i + shift < len(self) and j < len(other):
            if shift > 0:
                self.__elts__[i] = self.__elts__[i + shift]

            if self.__elts__[i] == other[j]:
                i += 1
                j += 1
            elif self.__elts__[i] < other[j]:
                shift += 1
            else:
                # self.__elts__[i] > other[j]
                j += 1

        if i < len(self):
            self.__elts__ = self.__elts__[:i]
        return self

    # TODO: Optimize
    # NOTE: Even though the big-O is bad, the left-to-right list reads may be
    #   fast.
    # O(self + other) assuming both are ListSets
    def __sub__(self, other):
        if (not other.is_sorted()) or not other.is_indexed():
            return __BCT_generic_set_sub__(self, other)

        new_set = type(self)()
        i = 0
        j = 0
        while i < len(self):
            if j == len(other):
                new_set.add(self[i])
                i += 1
            elif other[j] == self[i]:
                i += 1
                j += 1
            elif other[j] < self[i]:
                j += 1
            else:
                # self[i] is less than ALL remaining elements in other
                new_set.add(self[i])
                i += 1

        return new_set

    # O(self + other) assuming both are ListSets
    def __isub__(self, other):
        if (not other.is_sorted()) or not other.is_indexed():
            return __BCT_generic_set_isub__(self, other)

        i = 0
        j = 0
        shift = 0
        while i + shift < len(self) and j < len(other):
            if shift > 0:
                self.__elts__[i] = self.__elts__[i + shift]

            if self.__elts__[i] == other[j]:
                shift += 1
            elif self.__elts__[i] < other[j]:
                i += 1
            else:
                j += 1

        if shift > 0:
            for k in range(i, len(self.__elts__) - shift):
                self.__elts__[k] = self.__elts__[k + shift]
            self.__elts__ = self.__elts__[:(len(self.__elts__) - shift)]
        return self

    def __remove_helper__(self, idx):
        if idx == len(self.__elts__) - 1:
            self.__elts__.pop()
        elif idx == 0:
            self.__elts__ = self.__elts__[1:]
        else:
            self.__elts__ = self.__elts__[:idx] + self.__elts__[(idx + 1):]

    # Returns (found, idx) where
    #   `found` is True/False indicating whether elt is in
    #       the set.
    #   If `found` is True, then `idx` is the index of the found element.
    #   If `found` is False, then `idx` is the index at which the element would
    #       go if it were to be inserted into the set's list.
    def __bin_search__(self, elt):
        if len(self.__elts__) == 0:
            return (False, 0)
        if elt > self.__elts__[-1]:
            return (False, len(self.__elts__))
        elif elt < self.__elts__[0]:
            return (False, 0)
        elif elt == self.__elts__[-1]:
            return (True, len(self.__elts__) - 1)
        elif elt == self.__elts__[0]:
            return (True, 0)

        # The indices that could contain elt are in the range [low, high)
        #   (i.e. inclusive, exclusive)
        high = len(self.__elts__)
        low = 0
        while low < high - 1:
            mid = int((high + low) / 2)
            mid_val = self.__elts__[mid]

            if mid_val == elt:
                return (True, mid)
            elif mid_val < elt:
                low = mid
            elif elt < mid_val:
                high = mid

        if elt < mid_val:
            return (False, mid)
        return (False, mid + 1)

    def __str__(self):
        return "ListSet(" + str(self.__elts__) + ")"

class ListDict(DictLike, Sorted):

    def __init__(self, elts=None):
        self.__elts__ = []
        if elts is None or len(elts) == 0:
            return

        keys = [k for k, v in elts.items()]
        needs_sorting = False
        for i in range(1, len(keys)):
            if keys[i - 1] > keys[i]:
                needs_sorting = True
                break
        if needs_sorting:
            keys.sort()

        prev_key = keys[0]
        self.__elts__.append(prev_key)
        self.__elts__.append(elts[prev_key])
        for i in range(1, len(keys)):
            key = keys[i]
            if key == prev_key:
                continue
            prev_key = key

            self.__elts__.append(key)
            self.__elts__.append(elts[key])

    def items(self):
        return __ListDictItemsIterator__(self)

    def __iter__(self):
        return __ListDictKeysIterator__(self)

    def __len__(self):
        return int(len(self.__elts__) / 2)

    def __contains__(self, key):
        (found, _) = self.__bin_search__(key)
        return found

    def __getitem__(self, key):
        (found, idx) = self.__bin_search__(key)
        if not found:
            raise KeyError("Error! Key %s not found." % (key,))
        return self.__elts__[idx + 1]

    def __setitem__(self, key, value):
        (found, idx) = self.__bin_search__(key)
        if found:
            self.__elts__[idx + 1] = value
            return

        self.__elts__.append(None)
        self.__elts__.append(None)
        shifted_slots = int((len(self.__elts__) - idx) / 2) - 1
        i = len(self.__elts__) - 2
        for _ in range(0, shifted_slots):
            self.__elts__[i] = self.__elts__[i - 2]
            self.__elts__[i + 1] = self.__elts__[i - 1]
            i -= 2
        self.__elts__[i] = key
        self.__elts__[i + 1] = value
        return

    def __delitem__(self, key):
        (found, idx) = self.__bin_search__(key)
        if not found:
            raise KeyError("Error! Key %s not found." % (key,))

        if len(self.__elts__) == 2:
            self.__elts__ = []
        elif idx == len(self.__elts__) - 2:
            self.__elts__.pop()
            self.__elts__.pop()
        elif idx == 0:
            self.__elts__ = self.__elts__[2:]
        else:
            self.__elts__ = self.__elts__[:idx] + self.__elts__[idx + 2:]

    def __str__(self):
        s = "ListDict: {"
        for i in range(0, len(self)):
            if i > 0:
                s += ",  "
            j = i * 2
            s += "%s: %s" % (self.__elts__[j], self.__elts__[j + 1])
        s += "}"
        return s

    # Returns (found, idx) where
    #   `found` is True/False indicating whether key is in
    #       the set.
    #   If `found` is True, then `idx` is the index of the found key.
    #   If `found` is False, then `idx` is the index at which the key would
    #       go if it were to be inserted into the dict's list.
    def __bin_search__(self, key):
        if len(self.__elts__) == 0:
            return (False, 0)
        if key > self.__elts__[-2]:
            return (False, len(self.__elts__))
        elif key < self.__elts__[0]:
            return (False, 0)
        elif key == self.__elts__[-2]:
            return (True, len(self.__elts__) - 2)
        elif key == self.__elts__[0]:
            return (True, 0)

        # The indices that could contain elt are in the range [low, high)
        #   (i.e. inclusive, exclusive)
        high = len(self.__elts__)
        low = 0
        while low < high - 2:
            mid = int((high + low) / 2)
            if mid % 2 == 1:
                mid -= 1
            mid_val = self.__elts__[mid]

            if mid_val == key:
                return (True, mid)
            elif mid_val < key:
                low = mid
            elif key < mid_val:
                high = mid

        if key < mid_val:
            return (False, mid)
        return (False, mid + 2)

class __ListDictItemsIterator__:

    def __init__(self, ld):
        self.__elts__ = ld.__elts__
        self.__i__ = 0

    def __next__(self):
        if self.__i__ == len(self.__elts__):
            raise StopIteration
        res = (self.__elts__[self.__i__], \
               self.__elts__[self.__i__ + 1])
        self.__i__ += 2
        return res

    def __iter__(self):  # I don't know why this is needed.
        return self

class __ListDictKeysIterator__:

    def __init__(self, ld):
        self.__elts__ = ld.__elts__
        self.__i__ = 0

    def __next__(self):
        if self.__i__ == len(self.__elts__):
            raise StopIteration
        res = self.__elts__[self.__i__]
        self.__i__ += 2
        return res

    def __iter__(self):  # I don't know why this is needed.
        return self

if __name__ == "__main__":

    S = ListSet([2, 0, 8, 4, 8])
    S.add(6)
    S.remove(8)
    assert S.is_sorted()
    assert S.is_indexed()
    T = ListSet([3, 2, 1, 1, 1])
    assert T.is_sorted()
    R = S | T
    print("%s | %s = %s" % (S, T, R))
    assert list(R) == [0, 1, 2, 3, 4, 6]
    assert list(__BCT_generic_set_or__(S, T)) == [0, 1, 2, 3, 4, 6]
    R = S & T
    print("%s & %s = %s" % (S, T, R))
    assert list(R) == [2]
    R = S - T
    print("%s - %s = %s" % (S, T, R))
    assert list(R) == [0, 4, 6]
    assert -1 not in R
    assert 0 in R
    assert 1 not in R
    assert 2 not in R
    assert 4 in R
    assert 6 in R
    assert 8 not in R
    assert 10 not in R

    from basic_container_types import Set
    Q = Set([4, 5, 6])
    Z = Q | S
    assert sorted(list(Z)) == [0, 2, 4, 5, 6]
    P = Set([0, 2, 8])
    P |= S
    assert sorted(list(P)) == [0, 2, 4, 6, 8]

    # |=

    S = ListSet([1, 2, 4, 6, 7, 10])
    T = ListSet([0, 1, 2, 3, 4, 8, 10, 11])
    S |= T
    assert list(S) == [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]
    S = ListSet([0, 1])
    T = ListSet([1, 2, 3])
    S |= T
    assert list(S) == [0, 1, 2, 3]
    S = ListSet([4, 10])
    T = ListSet([1, 2, 3])
    S |= T
    assert list(S) == [1, 2, 3, 4, 10]

    # &=

    S = ListSet([1, 2, 4, 6, 7, 10])
    T = ListSet([0, 1, 2, 3, 4, 8, 10, 11])
    S &= T
    assert list(S) == [1, 2, 4, 10]
    S = ListSet([0, 1])
    T = ListSet([1, 2, 3])
    S &= T
    assert list(S) == [1]
    S = ListSet([4, 10])
    T = ListSet([1, 2, 3])
    S &= T
    assert list(S) == []

    # -=

    S = ListSet([1, 2, 4, 6, 7, 10])
    T = ListSet([0, 1, 2, 3, 4, 8, 10, 11])
    S -= T
    assert list(S) == [6, 7]
    S = ListSet([0, 1])
    T = ListSet([1, 2, 3])
    S -= T
    assert list(S) == [0]
    S = ListSet([4, 10])
    T = ListSet([1, 2, 3])
    S -= T
    assert list(S) == [4, 10]

    S -= ListSet()
    assert list(S) == [4, 10]
    S |= ListSet()
    assert list(S) == [4, 10]
    S &= ListSet()
    assert list(S) == []
    S = ListSet([50, 10, 20, 40, 30])
    S |= Set([30, 31, 30, 20, 19])
    assert list(S) == [10, 19, 20, 30, 31, 40, 50]

    D = ListDict({3: 4, 5: 6, 8: 10, 7: 9})
    print(D.__str__())
    assert D.__elts__ == [3, 4, 5, 6, 7, 9, 8, 10]
    assert len(D) == 4
    assert 3 in D
    assert 4 not in D
    assert 5 in D
    assert 6 not in D
    assert 7 in D
    assert 8 in D
    assert 9 not in D
    assert 10 not in D
    D[8] = 11
    assert 8 in D
    assert 11 not in D
    assert D.__elts__ == [3, 4, 5, 6, 7, 9, 8, 11]
    D[2] = 5
    assert D.__elts__ == [2, 5, 3, 4, 5, 6, 7, 9, 8, 11]
    D[6] = 6.7
    assert D.__elts__ == [2, 5, 3, 4, 5, 6, 6, 6.7, 7, 9, 8, 11]
    assert len(D) == 6
    del D[3]
    assert D.__elts__ == [2, 5, 5, 6, 6, 6.7, 7, 9, 8, 11]
    assert D.is_sorted()
    assert D.is_dictlike()
    assert not D.is_setlike()

    import random
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

        ls1 = [Set(l1), ListSet(l1)][random.randint(0, 1)]
        ls2 = [Set(l2), ListSet(l2)][random.randint(0, 1)]

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
            v = random.randint(val_min, val_max)
            res_s = set(s1)
            res_ls = type(ls1)(ls1)
            res_s.add(v)
            res_ls.add(v)
        elif operation == "remove" and len(l1) > 0:
            v = l1[random.randint(0, len(l1) - 1)]
            res_s = set(s1)
            res_ls = type(ls1)(ls1)
            res_s.remove(v)
            res_ls.remove(v)
        elif operation == "discard":
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

    import sys

    for i in range(0, 10):
        s = int(2**i)
        print("s == %d" % s)

        # print(sys.getsizeof(set(range(0, s))))
        # print(sys.getsizeof(list(range(0, s))))
        # print(sys.getsizeof(Set(range(0, s))))
        # print(sys.getsizeof(ListSet(range(0, s)).__elts__))

        # print("size of int")
        # print(sys.getsizeof(90000))
        # print("size of int pair")
        # print(sys.getsizeof((9, 10)))
        print("    size of list of tuples")
        l = [(i*2, i*2 + 1) for i in range(0, s)]
        l2 = [i for i in range(0, s*2)]
        d = {i * 2 : i * 2 + 1 for i in range(0, s)}
        print("        " + str(sys.getsizeof(l) + sum([sys.getsizeof(t) for t in l])))
        print("    Size of double-length list")
        print("        " + str(sys.getsizeof(l2)))
        print("    Size of dict.")
        print("        " + str(sys.getsizeof(d)))
