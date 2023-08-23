import sys
sys.path.append('..')
from common.util import preprocess, createCoMatrix, cosSimilarity

text = 'You say goodbye and I say hello.'
corpus, wordToId, idToWord = preprocess(text)

vocabSize= len(wordToId)
C = createCoMatrix(corpus, vocabSize= vocabSize)

c0 = C[wordToId['you']]     # "you"의 단어 벡터
c1 = C[wordToId['i']]       # "i" 의 단어 벡터

print(cosSimilarity(c0, c1))