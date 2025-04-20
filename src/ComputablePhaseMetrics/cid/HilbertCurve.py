class HilbertCurve():
    """
    This is a Pythonized version of the **principle** Hilbert curve implementation
    by John Skilling as described in:
    
    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
        Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Algorithm:
    ----------
    A single global Gray code [i.e. Reflected binary Code (RBC)] is being applied
    to the np-bit binary representation of the Hilbert distance/index H. This over-
    transforms the distance H and the **excess work** is undone by a single traverse
    through the np-bit Gray code representation of H.

    Parameters:
    -----------
        p   -   The number of bits for each dimension (int).
        n   -   The dimensionality of the hypercube (int).

    Returns:
    --------
        An iterable list of tuples: The i'th tuple contains the coordinates (int)
        to the i'th point along the Hilbert curve.
    """

    def __init__(self, order, dim):

        if type(order) is not int:
            raise TypeError('order must be an integer, got {}.'.format(type(order)))
        elif order <= 0:
            raise ValueError('order must be >= 1, got %d.' %(order))

        if type(dim) is not int:
            raise TypeError('dim must be an integer, got {}.'.format(type(order)))
        elif dim <= 1:
            raise ValueError('dim must be >= 2, got %d.' %(dim))

        self.dim = dim
        self.order = order
        self.bits = dim * order
    
    def __iter__(self):
        yield from self.points_from_distances()

    def points_from_distances(self):
        for gray in [dist ^ (dist >> 1) for dist in range(1 << self.bits)]:
            point = [self.extract_bits(gray, n) for n in range(self.dim)]
            yield self.undo_excess_work(point)
    
    def extract_bits(self, num, n):
        binary = format(num, '0'+str(self.bits)+'b')
        extracted_bits = binary[n::self.dim]
        return int(extracted_bits, 2)

    def undo_excess_work(self, point):
        for pot in (2 << p for p in range(self.order-1)):
            for n in reversed(range(self.dim)):
                if (point[n] & pot) != 0:
                    point[0] ^= pot - 1
                else:
                    exchange_low_bits = (point[0] ^ point[n]) & pot - 1
                    point[0] ^= exchange_low_bits
                    point[n] ^= exchange_low_bits
        return tuple(point)


def principle_curve(dim):
    return lambda order : [point for point in HilbertCurve(order, dim)]

def hilbert_2D_curves(order):
    def permute(points):
        return [((p2, p1)) for p1, p2 in points]
    
    def flip(points, axis):
        L = (1 << order) - 1
        if axis == 1:
            return [(L - p1, p2) for p1, p2 in points]
        elif axis == 2:
            return [(p1, L - p2) for p1, p2 in points]
        else:
            raise ValueError('Axis has to be either 1 or 2, got %d.' %(axis))
        
    def rotate90(points, k):
        k %= 4
        if k == 0:
            return points
        elif k == 1:
            return permute(flip(points, 1))
        elif k == 2:
            return flip(flip(points, 1), 2)
        elif k == 3:
            return flip(permute(points), 1)
        
    pc2D = principle_curve(2)

    return [rotate90(pc2D(order), k) for k in range(4)]
        
def hilbert_3D_curves(order):
    def permute(points):
        return [(p0, p2, p1) for p0, p1, p2 in points]

    def flip(points, axis):
        L = (1 << order) - 1
        if axis == 1:
            return [(p0, L - p1, p2) for p0, p1, p2 in points]
        elif axis == 2:
            return [(p0, p1, L - p2) for p0, p1, p2 in points]
        else:
            raise ValueError('Axis has to be either 1 or 2, got %d.' %(axis))

    def rotate90(points, k):
        k %= 4
        if k == 0:
            return points
        elif k == 1:
            return permute(flip(points, 1))
        elif k == 2:
            return flip(flip(points, 1), 2)
        else:
            return flip(permute(points), 1)
    
    pc3D = principle_curve(3)
    curves = (pc3D(order), permute(pc3D(order)))

    return [rotate90(curve, k) for curve in curves for k in range(4)]