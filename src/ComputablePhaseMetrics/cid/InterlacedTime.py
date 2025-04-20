from .HilbertCurve import hilbert_3D_curves
from multiprocessing import Pool
from .CID import CID_Q
import numpy as np

class InterlacedTime():

    def __init__(self, order, alpha=2, nshuff=2):
        self.dim = 3
        self.order = order
        self.Q = CID_Q(alpha, nshuff)
        self.hilbert_curves = hilbert_3D_curves(order)
        
    def __call__(self, data):

        if type(data) is not np.ndarray:
            raise TypeError('Data must be a numpy array, got {}.'.format(type(data)))
        elif data.ndim != self.dim:
            raise ValueError('Dimensions of numpy-data-array must be %d, got %d' %(self.dim, data.ndim))
        elif data.shape[1:] != (1 << self.order, ) * (self.dim - 1):
            raise ValueError('Spatial size must be {}, got {}.'.format((1 << self.order, ) * (self.dim - 1), data.shape[1:]))
        elif data.shape[0] % data.shape[1] != 0:
            raise ValueError('Temporal size must be a multiplum of the Spatial size; data.shape = {}'.format(data.shape))
        
        def split_data():
            temporal_sections = data.shape[0] // (1 << self.order)
            return np.split(data, temporal_sections, axis=0)
        
        def hilbert_scans():
            for hilbert_curve in self.hilbert_curves:
                yield [cube[point] for cube in split_data() for point in hilbert_curve]
                
        with Pool(len(self.hilbert_curves)) as p:
            qs = p.map_async(self.Q, hilbert_scans())
            p.close()
            p.join()
        
        return np.mean([q[0] for q in qs.get()]), np.mean([q[1] for q in qs.get()])