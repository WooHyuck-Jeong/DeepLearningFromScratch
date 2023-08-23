import numpy

text = "You say goodbye and I say hello."

# 단어 단위 분할
text = text.lower()     # 단어 소문자화
text = text.replace('.', ' .')
print(text)

# 공백 기준 분할
words = text.split(' ')
print(words)

# 단어 ID 부여
# ID의 리스트로 이용하기 위해 딕셔너리 활용
word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
        
print(id_to_word)
print(word_to_id)

print(id_to_word[1])
print(word_to_id['hello'])