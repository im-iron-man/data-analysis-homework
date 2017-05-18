def get_word(x):
    '''split the email'''
    return x.split()

def get_vocab(words):
    '''list the distinct vacab'''
    s = set([])
    for word in words:
        s |= set(word)
    return list(s)

def vocabvec(word, vocab):
    vector = [0] * len(vocab)
    for i in range(len(vocab)):
        if vocab[i] in word:
            vector[i] = 1
    return vector



X = [
    'my dog has flea problems help please',
    'maybe not take him to dog park stupid',
    'my dalmation is so cute I love him',
    'stop posting stupid worthless garbage',
    'mr licks ate my steak how to stop him',
    'quit buying worthless dog food stupid'
]
train_data_Y = [0, 1, 0, 1, 0, 1]

words = [get_word(x) for x in X]
vocab = get_vocab(words)
train_data_X = [vocabvec(word, vocab) for word in words]



from sklearn import naive_bayes as nb
clf = nb.BernoulliNB()

clf.fit(train_data_X, train_data_Y)

mail = 'my dog stupid'
word = get_word(mail)
vector = vocabvec(word, vocab)
print clf.predict([vector])