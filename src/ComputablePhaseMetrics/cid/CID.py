from numpy import log2, mean, random

class CID_Q():
    
    def __init__(self, alpha, nshuff):
        self.alpha = alpha
        self.num_shuffles = nshuff
        
    def __call__(self, hscan):
        cid = self.cid(hscan)
        return cid, 1 - cid / self.cid_shuffles(hscan)
        
    def lz77(self, sequence):
        sequence = ''.join(map(str, sequence))
        sub_strings = set()
        ind, inc = 0, 1
        while True:
            if ind + inc > len(sequence):
                break
            sub_str = sequence[ind : ind + inc]
            if sub_str in sub_strings:
                inc += 1
            else:
                sub_strings.add(sub_str)
                ind += inc
                inc = 1
        return len(sequence), len(sub_strings)

    def cid(self, hscan):
        L, C = self.lz77(hscan)
        return C*(log2(C) + log2(self.alpha)) / L

    def cid_shuffles(self, hscan):
        shuffles = []
        rng = random.default_rng()
        for i in range(self.num_shuffles):
            rng.shuffle(hscan)
            shuffles.append( self.cid(hscan) )
        return mean(shuffles)