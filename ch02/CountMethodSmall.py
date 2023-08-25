import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, createCoMatrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, wordToId, idToWord = preprocess(text)
vocabSize = len(idToWord)
C = createCoMatrix(corpus, vocabSize)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print("Co-Occurence Matrix")
print(C[0])

print("PPMI")
print(W[0])

print("SVD")
print(U[0])

# 2차원 벡터로 변환하기 위해
# 첫 두개 원소 추출
print("첫 두개 원소")
print(U[0, :2])

# 각 단어 2차원 벡터로 표현 후 그래프로 시각화
for word, wordId in wordToId.items():
    plt.annotate(word, (U[wordId, 0], U[wordId, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha= 0.5)
plt.show()