import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess

text = "You say goodbye and I say hello."

corpus, wordToId, idToWord = preprocess(text= text)

print(corpus)
print(idToWord)

C = np.array([[0, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 1, 1, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 1, 0]
              ], dtype= np.int32)

print(C.shape)

print(C[0]) # ID가 0인 단어의 벡터 표현
print(C[4]) # ID가 4인 단어의 벡터 표현
print(C[wordToId['goodbye']])   # "goodbye"의 벡터 표현

