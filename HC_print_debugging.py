__debug_indent_strs__ = [""]
__debug_printing_flag__ = False
__debug_print_min_depth__ = 0
__debug_print_max_depth__ = None
__debug_printing_conditional_flag__ = False

def debug_printing_conditional_on():
    global __debug_printing_conditional_flag__
    __debug_printing_conditional_flag__ = True

def debug_printing_conditional_off():
    global __debug_printing_conditional_flag__
    __debug_printing_conditional_flag__ = False

def turn_on_debug_printing(min_depth=0, max_depth=None):
    global __debug_printing_flag__
    global __debug_print_min_depth__
    global __debug_print_max_depth__
    __debug_printing_flag__ = True
    __debug_print_min_depth__ = min_depth
    __debug_print_max_depth__ = max_depth

def turn_off_debug_printing():
    global __debug_printing_flag__
    __debug_printing_flag__ = False

def debug_print(s, debug_depth=0):
    global __debug_printing_flag__
    global __debug_printing_conditional_flag__
    global __debug_indent_strs__
    global __debug_print_min_depth__
    global __debug_print_max_depth__

    if (not __debug_printing_flag__) and \
            not __debug_printing_conditional_flag__:
        return

    if debug_depth < __debug_print_min_depth__ and \
            not __debug_printing_conditional_flag__:
        return
    if __debug_print_max_depth__ is not None and \
            debug_depth > __debug_print_max_depth__ and \
            not __debug_printing_conditional_flag__:
        return

    while debug_depth >= len(__debug_indent_strs__):
        indent = ""
        for _ in range(0, len(__debug_indent_strs__)):
            indent += "  "
        __debug_indent_strs__.append(indent)

    if type(s) is str:
        print(__debug_indent_strs__[debug_depth] + s)
    else:
        print(__debug_indent_strs__[debug_depth] + str(s))


if __name__ == "__main__":
    debug_print("You shouldn't see this.")
    turn_on_debug_printing()
    debug_print("You should see")
    debug_print("this", debug_depth=2)
    debug_print("indented by 4 spaces.")
    turn_off_debug_printing()
    debug_print("You shouldn't see this.", debug_depth=10)
    turn_on_debug_printing()
    debug_print("Also, this line should be indented by 6", debug_depth=3)
    debug_print("and this line by 2.", debug_depth=1)
