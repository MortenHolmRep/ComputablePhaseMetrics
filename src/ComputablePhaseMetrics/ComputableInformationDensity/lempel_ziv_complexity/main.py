from . import LempelZivModule
from .LempelZiv import lz77_py, lz78_py


def lz77(sequence, version="C"):
    seq = "".join(map(str, sequence))
    if version == "C":
        return LempelZivModule.lz77(seq)
    elif version == "python":
        return lz77_py(seq)
    else:
        raise ValueError(f"Bad value for version: {version}.")


def lz78(sequence, version="C"):
    seq = "".join(map(str, sequence))
    if version == "C":
        raise NotImplementedError("lz78 C-version")
    elif version == "python":
        lz78_py(seq)
    else:
        raise ValueError(f"Bad value for version: {version}.")
