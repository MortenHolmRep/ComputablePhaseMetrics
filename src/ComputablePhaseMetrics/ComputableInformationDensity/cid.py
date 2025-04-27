from ComputablePhaseMetrics.ComputableInformationDensity.cid_abc import (
    ComputableInformationDensity,
)


class CID(ComputableInformationDensity):
    hamiltonian_cycle = [0, 1, 0, 2, 1, 0, 1, 2]
    length = len(hamiltonian_cycle)

    def __init__(self, dim, nbits, nshuff):
        super().__init__(dim, nbits, nshuff)

    def itter_hscan(self, data):
        """yields all 8 distinct Hilbert scanned views of the data.
        Since a view is returned, this operation is O(1)."""
        for k in self.hamiltonian_cycle:
            hcurve = self.principal_hcurve  # view of principal_hcurve i.e. O(1)
            if k == 0:
                hcurve[0] = (self.size - 1) - hcurve[0]
            if k == 1:
                hcurve[1] = (self.size - 1) - hcurve[1]
            if k == 2:
                hcurve[[0, 1]] = hcurve[[1, 0]]
            yield self.hscan(data, hcurve)  # view of data, i.e. O(1)

    def __call__(self, data):
        return super().__call__(data, n_workers=self.length)


def cid2d(nbits, nshuff):
    """Two-dimensional CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class"""
    return CID(2, nbits, nshuff)


def sequential_time(nbits, nshuff):
    """Spatiotemporal CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class"""
    return CID(2, nbits, nshuff)


def interlaced_time(nbits, nshuff):
    """Spatiotemporal CID Analysis \n
    Args:
        order: linear system size (log2) (int).
        nshuff: number of radnom shuffles of data.
    Returns: instance of the CID class"""
    return CID(3, nbits, nshuff)
