from abc import ABC, abstractmethod
from multiprocessing import Pool

import numpy as np

from ComputablePhaseMetrics.ComputableInformationDensity.computable_information_density import (
    cid,
    cid_shuffle,
)
from ComputablePhaseMetrics.ComputableInformationDensity.hilbert_curve import (
    hilbert_curve,
)


class ComputableInformationDensity(ABC):
    """Abstract base class for the Computable Information Density"""

    def __init__(self, dim, nbits, nshuff):
        self.principal_hcurve = hilbert_curve(dim, nbits)
        self.size = 1 << nbits
        self.nshuff = nshuff
        self.dim = dim

    @abstractmethod
    def itter_hscan(self, data):
        pass

    def hscan(self, data, hcurve):
        """Hilbert scaned view of data."""
        return data[tuple(hcurve)].T.ravel()

    def cid_shuffle(self, data):
        """CID of randomly shuffled data"""
        # flattened- and hilbert scanned- view are
        # statistically equivalent after shuffling:
        return cid_shuffle(data.ravel(), self.nshuff)

    def __call__(self, data, n_workers):
        # reshape data to be compatible with Hilbert scan:
        data = np.transpose(data).reshape((-1,) + (self.size,) * self.dim).T
        # create and configure a pool of workers:
        with Pool(n_workers) as pool:
            cid_pool = pool.map_async(cid, self.itter_hscan(data))
            pool.close()  # close the process pool
            pool.join()  # wait for all workers to complete

        return np.mean(cid_pool.get()), self.cid_shuffle(data)
