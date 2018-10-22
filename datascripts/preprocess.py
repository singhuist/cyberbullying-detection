import csv
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
#from gensim.models import Word2Vec
import math

#ques = []
ans1 = []
isbully1 = []
severity1 = []
bully1 = []
ans2 = []
isbully2 = []
severity2 = []
bully2 = []
ans3 = []
isbully3 = []
severity3 = []
bully3 = []

with open('formspring_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if len(row)>=1:
                r = row[0].split('\t')

                if len(r)==14:
                    #print(r)
                    #ques.append(r[2])
                    ans1.append(r[1])
                    isbully1.append(r[5])
                    severity1.append(r[6])
                    bully1.append(r[7])
                    ans2.append(r[2])
                    isbully2.append(r[8])
                    severity2.append(r[9])
                    bully2.append(r[10])
                    ans3.append(r[3])
                    isbully3.append(r[11])
                    severity3.append(r[12])
                    bully3.append(r[13])

                    line_count += 1

ans = ans1+ans2+ans3
isbully = isbully1+isbully2+isbully3
severity = severity1+severity2+severity3
bully = bully1+bully2+bully3

#######################################################################################################
#preprocessing and cleanup

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
specialsym = set(['~','`','@','$','#','%','^','&','*','``', "''", '..', '...', 'n/a', 'na'])

for a in range(len(ans)):
    ans[a] = ans[a].lower() #convert to lowercase

    word_tokens = word_tokenize(ans[a])
    no_punc = [w for w in word_tokens if not w in punctuations] #remove punctuation
    no_sym = [w for w in no_punc if not w in specialsym] #remove special symbols
    filtered_sentence = [w for w in no_sym if not w in stop_words] #remove stop words

    #ans[a] = ' '.join(filtered_sentence)
    ans[a] = filtered_sentence

    #print(ans[a])

for i in range(len(isbully)):
    if isbully[i] == 'No':
        isbully[i] = 0
    elif isbully[i] == 'Yes':
        isbully[i] = 1

"""model = Word2Vec(ans, size=100, window=5, min_count=3, workers=3)
vectors = model.wv

print(vectors.similarity('music','face'))
print(vectors.most_similar('book'))
print(vectors['sing'])
print(model['book'])

#del model"""

#vectorising the sentences and words
dictionary = {}
counter = 0
for a in ans:
    for w in range(len(a)):
        word = a[w]
        if a[w] not in dictionary:
            dictionary[word] = counter + 1
            counter = counter + 1
        a[w] = dictionary[word]

print(ans[:5])

train_data = ans[:math.floor(0.8*len(ans))]
train_label = isbully[:math.floor(0.8*len(isbully))]

test_data = ans[len(ans)-100:len(ans)]
test_label = isbully[len(isbully)-100:len(isbully)]

