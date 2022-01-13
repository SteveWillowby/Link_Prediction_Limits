from HC_basic_container_types import default_dict

# Assigns the sub-keys to be 0 through size - 1.
class Relabeling(object):

    IDENTITY_TYPE = 0
    SUB_COLLECTION_TYPE = 1

    # O(main_keys)
    def __init__(self, relabel_type, arg):
        self.__type__ = relabel_type
        if self.__type__ == Relabeling.IDENTITY_TYPE:
            self.__size__ = arg
        elif self.__type__ == Relabeling.SUB_COLLECTION_TYPE:
            main_keys = arg
            self.__sub_to_main__ = main_keys
            self.__main_to_sub__ = default_dict()
            for i in range(0, len(main_keys)):
                self.__main_to_sub__[main_keys[i]] = i
            self.__size__ = len(main_keys)
        else:
            raise ValueError("Error! Unknown value for `relabel_type`. Pass" + \
                "Relabeling.IDENTITY_TYPE or Relabeling.SUB_COLLECTION_TYPE")

    # O(1)
    def new_to_old(self, new_val):
        if self.__type__ == Relabeling.IDENTITY_TYPE:
            if new_val < 0 or new_val >= self.__size__:
                raise ValueError("Error! This identity 'relabeling' is for " + \
                          ("elements from 0 to %d only." % (self.__size__ - 1)))
            return new_val
        elif self.__type__ == Relabeling.SUB_COLLECTION_TYPE:
            return self.__sub_to_main__[new_val]

    # O(1)
    def old_to_new(self, old_val):
        if self.__type__ == Relabeling.IDENTITY_TYPE:
            if old_val < 0 or old_val >= self.__size__:
                raise ValueError("Error! This identity 'relabeling' is for " + \
                          ("elements from 0 to %d only." % (self.__size__ - 1)))
            return old_val
        elif self.__type__ == Relabeling.SUB_COLLECTION_TYPE:
            return self.__main_to_sub__[old_val]

    # O(1)
    def __len__(self):
        return self.__size__

if __name__ == "__main__":
    R = Relabeling(Relabeling.IDENTITY_TYPE, 10)
    assert R.new_to_old(5) == 5
    assert R.old_to_new(3) == 3
    assert len(R) == 10
    R.new_to_old(-1)  # Raises error
