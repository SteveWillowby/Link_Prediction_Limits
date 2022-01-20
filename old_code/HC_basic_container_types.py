# Properties:
#
# is_setlike -- guarantees the existence of the following:
#   Operators: |, &, -, |=, &=, -=, ==
#   Keywords: in, for x in obj, 
#   Functions: add(), remove(), discard(), len()
#
# is_indexed -- guarantees the existence of the following:
#   Operators: [] (getter)
#   Keywords: for x in obj,
#   Functions: len()
#
# is_sorted -- guarantees the existence of the following:
#   Operators: If the [] getter is present, elements are indexed in sorted order
#   Keywords: for x in obj (iterates in sorted order)
#   Functions: len()
#
# is_dictlike -- guarantees the existence of the following:
#   Operators: [] getter, [] setter
#   Keywords: for key in obj (iterates keys), del obj[key], in
#   Functions: len(), items()

class ContainerBase:

    def __init__(self):
        pass

    def is_setlike(self):
        return False

    def is_indexed(self):
        return False

    def is_dictlike(self):
        return False

    def is_sorted(self):
        return False

    def is_editable(self):  # Returns True to is_editable() by default.
        return True

class SetLike(ContainerBase):

    def __init__(self):
        pass

    def is_setlike(self):
        return True

class Indexed(ContainerBase):

    def __init__(self):
        pass

    def is_indexed(self):
        return True

class DictLike(ContainerBase):

    def __init__(self):
        pass

    def is_dictlike(self):
        return True

class Sorted(ContainerBase):

    def __init__(self):
        pass

    def is_sorted(self):
        return True

class View(ContainerBase):

    def __init__(self):
        pass

    def is_editable(self):
        return False

class Set(set, SetLike):
    def __init__(self, base_arg=None):
        if base_arg is None:
            set.__init__(self)
            return
        set.__init__(self, base_arg)

    def __or__(self, other):
        if type(other) is set or type(other) is Set:
            return Set(set.__or__(self, other))
        return __BCT_generic_set_or__(self, other)

    def __and__(self, other):
        if type(other) is set or type(other) is Set:
            return Set(set.__and__(self, other))
        return __BCT_generic_set_and__(self, other)

    def __sub__(self, other):
        if type(other) is set or type(other) is Set:
            return Set(set.__sub__(self, other))
        return __BCT_generic_set_sub__(self, other)

    def __ior__(self, other):
        if type(other) is set or type(other) is Set:
            return set.__ior__(self, other)
        return __BCT_generic_set_ior__(self, other)

    def __iand__(self, other):
        if type(other) is set or type(other) is Set:
            return set.__iand__(self, other)
        return __BCT_generic_set_iand__(self, other)

    def __isub__(self, other):
        if type(other) is set or type(other) is Set:
            return set.__isub__(self, other)
        return __BCT_generic_set_isub__(self, other)

class List(list, ContainerBase):
    def __init__(self, base_arg=None):
        if base_arg is None:
            list.__init__(self)
        else:
            list.__init__(self, base_arg)

    def is_indexed(self):
        return True

class Dict(dict, DictLike):
    def __init__(self, base_arg=None, identity_iterator=None):
        if identity_iterator is not None:
            if base_arg is not None:
                raise ValueError("Error! Cannot use both base_arg and " + \
                                 "identity_iterator to initialize Dict.")
            dict.__init__(self)
            for i in identity_iterator:
                self[i] = i
        elif base_arg is None:
            dict.__init__(self)
        else:
            dict.__init__(self, base_arg)

# TODO: Optimize
def __BCT_generic_set_or__(A, B, preferred_type=None):
    # print("BTC or (%s, %s)" % (type(A), type(B)))
    if preferred_type is None:
        new_set = type(A)(A)
    else:
        new_set = preferred_type(A)
    for elt in B:
        new_set.add(elt)
    return new_set

def __BCT_generic_set_ior__(A, B):
    for elt in B:
        A.add(elt)
    return A

# TODO: Optimize
def __BCT_generic_set_and__(A, B, preferred_type=None):
    # print("BTC and (%s, %s)" % (type(A), type(B)))
    if preferred_type is None:
        new_set = type(A)()
    else:
        new_set = preferred_type()

    if len(A) < len(B):
        for elt in A:
            if elt in B:
                new_set.add(elt)
    else:
        for elt in B:
            if elt in A:
                new_set.add(elt)
    return new_set

def __BCT_generic_set_iand__(A, B):
    # TODO: Can I avoid creating a copied list?
    A_elts = list(A)
    for elt in A_elts:
        if elt not in B:
            A.remove(elt)
    return A

# TODO: Optimize
def __BCT_generic_set_sub__(A, B, preferred_type=None):
    # print("BTC sub (%s, %s)" % (type(A), type(B)))
    if preferred_type is None:
        new_set = type(A)()
    else:
        new_set = preferred_type()

    for elt in A:
        if elt not in B:
            new_set.add(elt)
    return new_set

def __BCT_generic_set_isub__(A, B):
    if len(A) * 4 < len(B):
        # Only make a copy if A is significantly smaller.
        A_elts = list(A)
        for elt in A_elts:
            if elt in B:
                A.remove(elt)
    else:
        for elt in B:
            A.discard(elt)
    return A

__default_set_type__ = set
__default_dict_type__ = dict
__default_list_type__ = list

def default_set(arg=None, copy=True):
    global __default_set_type__
    if (not copy) and type(arg) is __default_set_type__:
        return arg
    if arg is None:
        return __default_set_type__()
    return __default_set_type__(arg)

def default_dict(arg=None, copy=True):
    global __default_dict_type__
    if (not copy) and type(arg) is __default_dict_type__:
        return arg
    if arg is None:
        return __default_dict_type__()
    return __default_dict_type__(arg)

def default_list(arg=None, copy=True):
    global __default_list_type__
    if (not copy) and type(arg) is __default_list_type__:
        return arg
    if arg is None:
        return __default_list_type__()

    return __default_list_type__(arg)

def set_default_set_type(t):
    global __default_set_type__
    __default_set_type__ = t

def set_default_dict_type(t):
    global __default_dict_type__
    __default_dict_type__ = t

def set_default_list_type(t):
    global __default_list_type__
    __default_list_type__ = t

if __name__ == "__main__":

    S = Set()
    S = Set([1, 2])
    S.add(3)
    print(S.is_setlike())
    print(S == set([1, 2, 3]))
    S = Set(range(0, 20))
    T = Set([1, 2, 4, 8, 16])
    R = S & T

    import time
    import random
    test_size = 20000
    start_1 = time.time()
    for i in range(0, test_size):
        d = {j: j for j in range(0, i)}
        assert len(d) == i
    end_1 = time.time()
    start_2 = time.time()
    for i in range(0, test_size):
        d = Dict({j: j for j in range(0, i)})
        assert len(d) == i
    end_2 = time.time()
    start_3 = time.time()
    for i in range(0, test_size):
        d = Dict(identity_iterator=range(0,i))
        assert len(d) == i
    end_3 = time.time()
    print("Time 1 (regular dict):                      %f" % (end_1 - start_1))
    print("Time 2 (passing {} into Dict):              %f" % (end_2 - start_2))
    print("Time 3 (using identity_iterator with Dict): %f" % (end_3 - start_3))

    data_access = [random.randint(0, 2) for _ in range(0, test_size)]
    start_1 = time.time()
    for i in range(0, test_size):
        # d = [j for j in range(0, i)]
        d = list(range(0, i))
        assert len(d) == i
    end_1 = time.time()
    start_2 = time.time()
    for i in range(0, test_size):
        d = List([j for j in range(0, i)])
        assert len(d) == i
    end_2 = time.time()
    start_3 = time.time()
    for i in range(0, test_size):
        d = List(range(0, i))
        assert len(d) == i
    end_3 = time.time()
    print("Time 1 (regular list):                %f" % (end_1 - start_1))
    print("Time 2 (passing [] into List):        %f" % (end_2 - start_2))
    print("Time 3 (using iterator with List): %f" % (end_3 - start_3))

