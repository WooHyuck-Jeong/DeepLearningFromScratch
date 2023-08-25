import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.util import preprocess, createCoMatrix, cosSimilarity, ppmi

text = "You say goodbye and I say hello."
corpus, wordToId, idToWord = preprocess(text= text)
vocabSize = len(wordToId)
C = createCoMatrix(corpus, vocabSize)
W = ppmi(C)

np.set_printoptions(precision= 3)       # 유효 자릿수를 세 자리로 표시
print('Co-Occurence Matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)