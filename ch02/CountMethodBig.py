import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import ptb
from common.util import mostSimilar, createCoMatrix, ppmi
from sklearn.utils.extmath import randomized_svd

windowSize = 2
wordVecSize = 100

corpus, wordToId, idToWord = ptb.load_data('train')
vocabSize= len(wordToId)
print("동시발생 수 계산...")
C = createCoMatrix(corpus, vocabSize)
print("PPMI 계산...")
W = ppmi(C, verbose= True)

print('SVD 계산...')
try:
    # truncated SVD (Fast)
    U, S, V = randomized_svd(W, n_components= wordVecSize, n_iter= 5, random_state= None)

except:
    # SVD (Slow)
    U, S, V = np.linalg.svd(W)

wordVecs = U[:, :wordVecSize]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    mostSimilar(query, wordToId, idToWord, wordVecs, top= 5)