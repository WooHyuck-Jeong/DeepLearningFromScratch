import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.trainer import Trainer
from common.optimizer import Adam
from Simeple_CBOW import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

windowSize = 1
hiddenSize = 5
batchSize = 3
maxEpoch = 1000

text = "You say goodbye and I say hello."

corpus, wordToId, idToWord = preprocess(text)

vocabSize = len(wordToId)

contexts, target = create_contexts_target(corpus, windowSize)
target = convert_one_hot(target, vocabSize)
contexts = convert_one_hot(contexts, vocabSize)

model = SimpleCBOW(vocabSize, hiddenSize)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, maxEpoch, batchSize)
trainer.plot()

wordVecs = model.word_vecs
for wordId, word in idToWord.items():
    print(word, wordVecs[wordId])