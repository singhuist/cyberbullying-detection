import string
import pickle

def load_data():
    filename = "../data/twitter_data.pkl"
    print("Loading data from file: " + filename)
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    labels = [] 
    ids = []
    for i in range(len(data)):
        text = "".join(l for l in data[i]['text'] if l not in string.punctuation)
        x_text.append((data[i]['text']).encode('utf-8'))
        labels.append(data[i]['label'])
    return x_text,labels