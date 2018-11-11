def simpleEncode(sentenceList):
    '''
    Encodes sentences into numbers, simply adding a new entry for a new word
    :param sentenceList: List of sentences to be vectorised
    :return: encoded list of sentences
    '''

    dictionary = {}
    counter = 0
    encodedData = []
    for s in sentenceList:
        encodedsent = []
        s = s.split()
        for w in range(len(s)):
            word = s[w]
            if word not in dictionary:
                dictionary[word] = counter + 1
                counter = counter + 1
            s[w] = dictionary[word]
            encodedsent.append(s[w])
        encodedData.append(encodedsent)

    return encodedData

