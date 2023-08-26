import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.layers import MatMul

# 샘플 맥락 데이터
c0 = np.array([1, 0, 0, 0, 0, 0, 0])
c1 = np.array([0, 0, 1, 0, 0, 0, 0])

# 가중치 초기화
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 계층 생성
inputLayer0 = MatMul(W_in)
inputLayer1 = MatMul(W_in)
outputLayer = MatMul(W_out)

# 순전파
h0 = inputLayer0.forward(c0)
h1 = inputLayer1.forward(c1)
h = 0.5 * (h0 + h1)
s = outputLayer.forward(h)

print(s)