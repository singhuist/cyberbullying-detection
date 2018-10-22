import csv
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

ques = []
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

    '''print(ques)
    print(ans1)
    print(severity1)
    print(bully1)
    print(ans2)
    print(severity2)
    print(bully2)
    print(ans3)
    print(severity3)
    print(bully3)'''

ans = ans1+ans2+ans3
isbully = isbully1+isbully2+isbully3
severity = severity1+severity2+severity3
bully = bully1+bully2+bully3

#print(fullans)
#print(len(ans))
#print(isbully)
#print(severity)
#print(bully)

#preprocessing

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
specialsym = set(['~','`','@','$','#','%','^','&','*','``', "''", '..', '...', 'n/a', 'na'])

for a in range(len(ans)):
    ans[a] = ans[a].lower() #convert to lowercase

    word_tokens = word_tokenize(ans[a])
    no_punc = [w for w in word_tokens if not w in punctuations] #remove punctuation
    no_sym = [w for w in no_punc if not w in specialsym] #remove special symbols
    filtered_sentence = [w for w in no_sym if not w in stop_words] #remove stop words

    ans[a] = ' '.join(filtered_sentence)

    #print(ans[a])

