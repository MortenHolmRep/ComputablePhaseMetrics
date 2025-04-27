import logging
from math import log2
from multiprocessing import Pool, cpu_count

import numpy as np

try:
    from ComputablePhaseMetrics.ComputableInformationDensity.lempel_ziv_complexity.LempelZivModule import (
        lz77,
    )

    logging.info("Using C implementation of Lempel-Ziv (`LempelZivModule`).")
except ImportError:
    from ComputablePhaseMetrics.ComputableInformationDensity.lempel_ziv_complexity.LempelZiv import (
        lz77_py as lz77,
    )

    logging.warning(
        "C implementation of Lempel-Ziv not found. Falling back to Python implementation (`LempelZiv_py`)."
    )


def cid(sequence):
    """Computable Information Density

    Args: one-dimensional data array.
    Returns: CID measure of the sequence.
    """
    if isinstance(sequence, np.ndarray):
        sequence = "".join(map(str, sequence))

    C, L = lz77(sequence), len(sequence)
    return C * (log2(C) + 2 * log2(L / C)) / L


def cid_shuffle(sequence, nshuff):
    """Computable Information Density

    Args:
        sequence: one-dimensional data array.
        nshuff: number of shuffles of the sequence.
    Returns:
        CID measure of the randomly shuffled sequence.
    """
    # get new instance of NumPy's random number generator
    rng = np.random.default_rng()  # this can be seeded!

    # generator yielding shuffled copy of sequence:
    def shuffle():
        for _ in range(nshuff):
            rng.shuffle(sequence)
            # yield copy so not to share memory
            # among process-pool of workers:
            yield sequence.copy()

    # create and configure a pool of workers:
    with Pool(min(cpu_count(), nshuff)) as pool:
        cid_pool = pool.map_async(cid, shuffle())
        pool.close()  # close the process pool
        pool.join()  # wait for all workers to complete

    return np.mean(cid_pool.get())


def cid_correlation(sequence):
    """CID Correlation Length \n
    Args: one-dimensional data array.
    Returns: correlation length using CID measure.
    """
    # Implement metthod from:
    # https://doi.org/10.1103/PhysRevLett.125.170601
    raise NotImplementedError("cid_correlation")
