import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.util import preprocess, createContextsTarget, convertOneHot

text = "You say goodbye and I say hello."

corpus, wordToId, idToWord = preprocess(text)

# print(corpus)

# print(idToWord)

# Contexts and Target 생성
contexts, target = createContextsTarget(corpus, windowSize= 1)

print(contexts)

print(target)

# Contexts and Target Convert One-Hot
vocabSize = len(wordToId)
target = convertOneHot(target, vocabSize)
contexts = convertOneHot(contexts, vocabSize)

print(contexts)
print(target)