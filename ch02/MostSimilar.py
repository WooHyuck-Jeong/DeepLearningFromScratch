import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.util import preprocess, createCoMatrix, mostSimilar

text = "You say goodbye and I say hello."
corpus, wordToId, idToWord = preprocess(text)
vocabSize = len(wordToId)
C = createCoMatrix(corpus= corpus, vocabSize= vocabSize)

mostSimilar('you', wordToId, idToWord, C, top= 5)