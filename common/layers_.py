from common.np import *


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        
        for i, wordId in enumerate(self.idx):
            dW[wordId] += dout[i]
        
        # 혹은
        # np.add.ad(dW, self.idx, dout)
        return None