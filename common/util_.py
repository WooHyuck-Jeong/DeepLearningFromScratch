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


def mostSimilar(query, wordToId, idToWord, wordMatrix, top= 5):
    """검색어로 주어진 단어에 대해 비슷한 단어를 유사도 순으로 출력

    Args:
        query (str): 검색할 단어
        wordToId (dict): 단어 - 단어 ID 딕셔너리
        idToWord (dict): 단어 ID - 단어 딕셔너리
        wordMatrix (mat): 단어 벡터 행렬. 각 행에 대응하는 단어의 벡터 저장
        top (int, optional): 상위 표현 개수. Defaults to 5.
    """
    
    # 검색어 추출
    if query not in wordToId:
        print("%s(을)를 찾을 수 없습니다."%query)
        return
    
    print('\n[query] ' + query)
    queryId = wordToId[query]
    queryVec = wordMatrix[queryId]

    # 코사인 유사도 계산
    vocabSize = len(idToWord)
    similarity = np.zeros(vocabSize)
    for i in range(vocabSize):
        similarity[i] = cosSimilarity(wordMatrix[i], queryVec)

    # 코사인 유사도 기준으로 내림차순 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if idToWord[i] == query:
            continue
        print(' %s: %s' %(idToWord[i], similarity[i]))

        count += 1
        if count >= top:
            return
        
def ppmi(C, verbose= False, eps= 1e-8):
    """Positive PMI

    Args:
        C (mat): Co-Occurence Matrix
        verbose (bool, optional): Progress Print. Defaults to False.
        eps (double): epsilon. Defaults to 1e-8.
    """
    
    M = np.zeros_like(C, dtype= np.float32)
    N = np.sum(C)
    S = np.sum(C, axis= 0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
    
    return M

def createContextsTarget(corpus, windowSize):
    """맥락과 타겟을 만드는 함수

    Args:
        corpus (matrix): 말뭉치 벡터
        windowSize (int): 주변 맥락 확인 갯수
    """
    
    target = corpus[windowSize : -windowSize]
    contexts = []
    
    for idx in range(windowSize, len(corpus) - windowSize):
        cs = []
        for t in range(-windowSize, windowSize + 1) :
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

def convertOneHot(corpus, vocabSize):
    """원-핫 표현으로 변환

    Args:
        corpus (array): 단어 ID 목록 (1차원 또는 2차원 넘파이 배열)
        vocabSize (int): 어휘 수
        return : 원-핫 표현 (2차원 또는 3차원 넘파이 배열)
    """

    N = corpus.shape[0]

    if corpus.ndim == 1:
        oneHot = np.zeros((N, vocabSize), dtype= np.int32)
        for idx, wordId in enumerate(corpus):
            oneHot[idx, wordId] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        oneHot = np.zeros((N, C, vocabSize), dtype= np.int32)
        for idx0, wordIds in enumerate(corpus):
            for idx1, wordId in enumerate(wordIds):
                oneHot[idx0, idx1, wordId] = 1
    
    return oneHot