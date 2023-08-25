import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import ptb

corpus, wordToId, idToWord = ptb.load_data('train')

print('말뭉치 크기 :', len(corpus))
print('corpus[:30] : ', corpus[:30])
print()
print('id to word [0] : ', idToWord[0])
print('id to word [1] : ', idToWord[1])
print('id to word [2] : ', idToWord[2])
print()
print("word to id ['car'] : ", wordToId['car'])
print("word to id ['happy'] :", wordToId['happy'])
print("word to id ['lexus'] : ", wordToId['lexus'])
