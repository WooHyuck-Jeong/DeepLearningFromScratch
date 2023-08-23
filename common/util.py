import numpy as np

def preprocess(text):
    """말뭉치 전처리 함수

    Args:
        text (text): text

    Returns:
        corpus : arr
        wordToID : dict
        idToWord : dict
    """
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    wordToId = {}
    idToWord = {}
    for word in words:
        if word not in wordToId:
            newId = len(wordToId)
            wordToId[word] = newId
            idToWord[newId] = word
            
    corpus = np.array([wordToId[w] for w in words])

    return corpus, wordToId, idToWord

def createCoMatrix(corpus, vocabSize, windowSize= 1):
    """동시발생 행렬 생성 함수

    Args:
        corpus (array): corpus array
        vocabSize (int): 어휘 수
        windowSize (int, optional): _description_. Defaults to 1.
    """
    corpusSize= len(corpus)
    coMatrix = np.zeros((vocabSize, vocabSize), dtype= np.int32)

    for idx, wordId in enumerate(corpus):
        for i in range(1, windowSize + 1):
            leftIdx = idx - 1
            rightIdx = idx + 1
            
            if leftIdx >= 0:
                leftWordId = corpus[leftIdx]
                coMatrix[wordId, leftWordId] += 1
                
            if rightIdx < corpusSize:
                rightWordId = corpus[rightIdx]
                coMatrix[wordId, rightWordId] += 1
                
    return coMatrix

def cosSimilarity(x, y, eps= 1e-8):
    """코사인 유사도 측정 함수

    Args:
        x (vec): word vector
        y (vec): word vector
        eps : epsilon 0으로 나누는 오류 해결을 위한 인수
    """
    
    nX = x / (np.sqrt(np.sum(x ** 2)) + eps)        # x 정규화
    nY = y / (np.sqrt(np.sum(y ** 2)) + eps)        # y 정규화
    
    return np.dot(nX, nY)

    