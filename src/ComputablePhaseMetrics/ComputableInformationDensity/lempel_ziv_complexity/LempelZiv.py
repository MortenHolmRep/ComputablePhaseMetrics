def lz77_py(seq):
    """Lempel-Ziv 77 complexity \n
    Note: The input sequence is cast as string
    and not tuple (for performance reasons).
    """
    complexity, ind, inc = 1, 1, 0
    while ind + inc < len(seq):
        if seq[ind : ind + inc + 1] in seq[: ind + inc]:
            inc += 1
        else:
            complexity += 1
            ind += inc + 1
            inc = 0
    return complexity + 1 if inc != 0 else complexity


def lz78_py(seq):
    """Lempel-Ziv 78 complexity \n
    Note: The input sequence is cast as string
    and not tuple (for performance reasons).
    """
    sub_strings = set()
    ind, inc = 0, 1
    while ind + inc <= len(seq):
        sub_str = seq[ind : ind + inc]
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)
