from .HilbertCurve import hilbert_2D_curves
from multiprocessing import Pool
from .CID import CID_Q
import numpy as np

class SequentialTime():

    def __init__(self, order, alpha=2, nshuff=2):
        self.dim = 3
        self.order = order
        self.Q = CID_Q(alpha, nshuff)
        self.hilbert_curves = hilbert_2D_curves(order)
        
    def __call__(self, data):

        if type(data) is not np.ndarray:
            raise TypeError('Data must be a numpy array, got {}.'.format(type(data)))
        elif data.ndim != self.dim:
            raise ValueError('Dimensions of numpy-data-array must be %d, got %d' %(self.dim, data.ndim))
        elif data.shape[1:] != (1 << self.order, ) * (self.dim - 1):
            raise ValueError('Spatial size must be {}, got {}.'.format((1 << self.order, ) * (self.dim - 1), data.shape[1:]))
        
        def hilbert_scans():
            for hilbert_curve in self.hilbert_curves:
                yield [[lattice[point] for point in hilbert_curve] for lattice in data]
        
        with Pool(len(self.hilbert_curves)) as p:
            results = p.map_async(self.get_cids, hilbert_scans())
            p.close()
            p.join()
            
        cids, qs = ( np.mean(res, axis=0) for res in zip(*results.get()) )
        
        return np.mean(cids), np.std(cids), np.mean(qs), np.std(qs)
    
    def get_cids(self, hscans):
        res = [self.Q(hscan) for hscan in hscans]
        return [list(a) for a in zip(*res)]