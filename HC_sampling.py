from HC_basic_container_types import SetLike, DictLike, default_dict
from HC_list_containers import ListSet
import random

def sample_proportionately(the_list, the_values):
    cumulative_values = [0]
    for value in the_values:
        cumulative_values.append(cumulative_values[-1] + value)
    max_value = cumulative_values[-1]
    sampled_value = random.randint(1, max_value)

    # Binary Search:
    # TODO: Review logic.
    min_idx = 0
    max_idx = len(cumulative_values) - 1
    min_val = cumulative_values[min_idx] 
    max_val = cumulative_values[max_idx] 
    mid_idx = int((max_idx + min_idx) / 2)

    diff = max_idx - min_idx

    while max_idx > min_idx:
        mid_idx = int((max_idx + min_idx) / 2)
        mid_val = cumulative_values[mid_idx]
        if mid_val == sampled_value:
            return the_list[mid_idx]
        elif mid_idx == min_idx:
            return the_list[mid_idx]
        elif sampled_value < mid_val:
            max_idx = mid_idx
            max_val = mid_val
        else:
            min_idx = mid_idx
            min_val = mid_val
        new_diff = max_idx - min_idx
        assert new_diff < diff
        diff = new_diff

    return the_list[min_idx]

class SampleSet(SetLike):

    def __init__(self, collection=None):
        if collection is None:
            collection = []
        self.__elements__ = list(collection)
        self.__element_dict__ = default_dict()
        for i in range(0, len(self.__elements__)):
            self.__element_dict__[self.__elements__[i]] = i        

    # O(1)
    def add(self, element):
        if element not in self.__element_dict__:
            self.__element_dict__[element] = len(self.__elements__)
            self.__elements__.append(element)

    # O(1)
    def remove(self, element):
        if element not in self.__element_dict__:
            raise KeyError("Error! %s not found in SampleSet %s" % \
                           (element, self))
        idx = self.__element_dict__[element]
        del self.__element_dict__[element]
        if idx < len(self.__elements__) - 1:
            self.__elements__[idx] = self.__elements__[-1]
            self.__element_dict__[self.__elements__[idx]] = idx
        self.__elements__.pop()

    # O(1)
    def discard(self, element):
        if element not in self.__element_dict__:
            return
        # Now for the same code as in remove().
        idx = self.__element_dict__[element]
        del self.__element_dict__[element]
        if idx < len(self.__elements__) - 1:
            self.__elements__[idx] = self.__elements__[-1]
            self.__element_dict__[self.__elements__[idx]] = idx
        self.__elements__.pop()

    # O(1), assuming random.randint is O(1)
    def sample(self):
        if len(self.__elements__) == 0:
            raise RuntimeError("Error! Cannot sample from an empty SampleSet.")
        return self.__elements__[random.randint(0, len(self.__elements__) - 1)]

    # O(1), assuming random.randint is O(1)
    def sample_excluding(self, excluded_element):
        if len(self.__elements__) == 0:
            raise RuntimeError("Error! Cannot sample and exclude an element " +\
                               "when the SampleSet has size 1.")
        elif excluded_element not in self.__element_dict__:
            raise KeyError("Error! excluded_element %s not in SampleSet %s" \
                            % (excluded_element, self))
        excl_idx = self.__element_dict__[excluded_element]
        sample_idx = random.randint(0, len(self.__elements__) - 2)
        if sample_idx >= excl_idx:
            sample_idx += 1
        return self.__elements__[sample_idx]

    # O(1)
    def __iter__(self):
        return self.__elements__.__iter__()

    # O(1)
    def __len__(self):
        return len(self.__elements__)

    # O(1)
    def __contains__(self, key):
        return key in self.__element_dict__

    # O(self + other)
    def __or__(self, other):
        if len(self) > len(other):
            s = SampleSet(self)
            s |= other
        else:
            s = SampleSet(other)
            s |= self
        return s

    def __ror__(self, other):
        return self.__or__(other)

    # O(min(self, other))
    def __and__(self, other):
        if len(self) > len(other):
            s = SampleSet(other)
            s &= self
        else:
            s = SampleSet(self)
            s &= other
        return s

    def __rand__(self, other):
        return self.__and__(other)

    # O(other)
    def __ior__(self, other):
        for elt in other:
            if elt not in self:
                self.add(elt)
        return self

    # O(self)
    def __iand__(self, other):
        elts = [elt for elt in self]
        for elt in elts:
            if elt not in other:
                self.remove(elt)
        return self

    # O(min(self, other))
    def __isub__(self, other):
        if len(self) > len(other):
            for elt in other:
                if elt in self:
                    self.remove(elt)
        else:
            elts = [elt for elt in self]
            for elt in elts:
                if elt in other:
                    self.remove(elt)
        return self

    # O(self)
    def __sub__(self, other):
        s = SampleSet(self)
        s -= other
        return s

    # O(other)
    def __rsub__(self, other):
        s = SampleSet(other)
        s -= self
        return s

class SampleListSet(ListSet):

    def __init__(self, base_arg=None):
        ListSet.__init__(self, base_arg)

    # O(1), assuming random.randint is O(1)
    def sample(self):
        if len(self.__elts__) == 0:
            raise RuntimeError("Error! Cannot sample from an empty SampleListSet.")
        sample_idx = random.randint(0, len(self.__elts__) - 1)
        return self.__elts__[sample_idx]

    # O(log n), assuming random.randint is O(1) (or even O(log n))
    def sample_excluding(self, excluded_element):
        if len(self.__elts__) == 1:
            raise RuntimeError("Error! Cannot sample from empty SampleListSet.")
        (found, excluded_idx) = self.__bin_search__(excluded_element)
        if not found:
            raise KeyError(("Error! excluded_element %s" % excluded_element) + \
                           " not in this SampleListSet.")
        sample_idx = random.randint(0, len(self.__elts__) - 2)
        if sample_idx >= excluded_idx:
            sample_idx += 1
        return self.__elts__[sample_idx]

class SampleDict(DictLike):

    def __init__(self, d=None):
        if d is None:
            d = {}
        self.__items_list__ = [[k, v] for k, v in d.items()]
        self.__key_order_dict__ = {self.__key_list__[i]: i \
                                   for i in range(0, len(self.__key_list__))}

    # Returns a random (key, value) tuple.
    #
    # O(1), assuming random.randint is O(1)
    def sample(self):
        return tuple(self.__items_list__[\
                        random.randint(0, len(self.__items_list__) - 1)])

    # Returns a random (key, value) tuple.
    #
    # O(1), assuming random.randint is O(1)
    def sample_excluding(self, excluded_key):
        if len(self.__items_list__) == 0:
            raise RuntimeError("Error! Cannot sample and exclude a key " +\
                               "when the SampleDict has size 1.")
        elif excluded_key not in self.__key_order_dict__:
            raise KeyError("Error! excluded_key %s not in SampleSet %s" \
                            % (excluded_key, self))
        exlc_idx = self.__key_order_dict__[excluded_key]
        sample_idx = random.randint(0, len(self.__items_list__) - 2)
        if sample_idx >= excl_idx:
            sample_idx += 1
        return tuple(self.__items_list__[sample_idx])

    # O(1)
    def __setitem__(self, key, value):
        if key not in self.__key_order_dict__:
            self.__items_list__.append([key, value])
            idx = len(self.__items_list__) - 1
            self.__key_order_dict__[key] = idx
        else:
            idx = self.__key_order_dict__[key]
            self.__items_list__[key][1] = value

    # O(1)
    def __getitem__(self, key):
        if key not in self.__key_order_dict__:
            raise KeyError("Error! Key %s not in SampleDict %s" % (key, self))
        return self.__items_list__[self.__key_order_dict__[key]][1]

    # O(1)
    def __del__(self, key):
        if key not in self.__key_order_dict__:
            raise KeyError("Error! Key %s not in SampleDict %s" % (key, self))
        idx = self.__key_order_dict__[key]
        del self.__key_order_dict__[key]
        if idx < len(self.__items_list__) - 1:
            self.__items_list__[idx] = self.__items_list__[-1]
            self.__key_order_dict__[self.__items_list__[idx][0]] = idx
        self.__items_list__[idx].pop()

    # O(1)
    def __len__(self):
        return len(self.__items_list__)

    # Keys
    #
    # O(1)
    def __iter__(self):
        return __SampleDictIter__(self)

    # (key, value) tuples
    #
    # O(1)
    def items(self):
        return __SampleDictItems__(self)

    # O(1)
    def __contains__(self, key):
        return key in self.__key_order_dict__

class __SampleDictItems__:

    def __init__(self, sd):
        self.__iter_obj__ = sd.__items_list__.__iter__()
        self.__length_hint__ = self.__iter_obj__.__length_hint__

    def __next__(self):
        return tuple(self.__iter_obj__.next())

class __SampleDictIter__:
    def __init__(self, sd):
        self.__iter_obj__ = sd.__items_list__.__iter__()
        self.__length_hint__ = self.__iter_obj__.__length_hint__

    def __next__(self):
        return self.__iter_obj__.next()[0]




__default_sample_set_type__ = SampleSet

def default_sample_set(arg=None, copy=True):
    global __default_sample_set_type__

    if (not copy) and type(arg) is __default_sample_set_type__:
        return arg

    if arg is None:
        return __default_sample_set_type__()
    return __default_sample_set_type__(arg)

def set_default_sample_set_type(t):
    global __default_sample_set_type__
    __default_sample_set_type__ = t


if __name__ == "__main__":
    s = SampleSet([1, 2, 4, 8, 16])
    print([x for x in s])

    s.add(32)
    s.remove(2)
    s.remove(1)
    s.add(32)
    s.add(2)
    s.remove(8)
    assert sorted([x for x in s]) == [2, 4, 16, 32]

    results = {x: 0 for x in s}
    for _ in range(0, 100000):
        results[s.sample_excluding(2)] += 1
    print(results)
    assert sorted([x for x in s]) == [2, 4, 16, 32]

    s -= set([2, 10, 32])
    assert sorted([x for x in s]) == [4, 16]
    s = s | set([1, 2, 8])
    assert sorted([x for x in s]) == [1, 2, 4, 8, 16]

    results = {x: 0 for x in s}
    for _ in range(0, 100000):
        results[s.sample()] += 1
    print(results)
