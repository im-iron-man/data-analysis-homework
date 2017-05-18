import os
from sklearn import naive_bayes as nb


######################################
# find train_data_X and train_data_Y #
######################################

def get_word(filename):
    import re
    f = open(filename, 'r')
    List = re.split(r'\W*', f.read())
    return [word.lower() for word in List if len(word) > 2]

def get_vocab(words):
    '''list the distinct vacab'''
    s = set([])
    for word in words:
        s |= set(word)
    return s

def vocabvec(word, vocab):
    vector = [0] * len(vocab)
    for i in range(len(vocab)):
        if vocab[i] in word:
            vector[i] = 1
    return vector



foldernames = os.listdir('email')
words = []
vocab = set([])

for i in range(2):
    filenames = os.listdir('email' + '/' + foldernames[i+1])
    wordsi = [get_word(os.path.join('email', foldernames[i+1], filenames[j])) for j in range(15)]
    words += wordsi
    vocabi = get_vocab(wordsi)
    vocab |= vocabi


train_data_X = [vocabvec(word, list(vocab)) for word in words]
train_data_Y = [0] * 15 + [1] * 15



#########
# train #
#########

clf = nb.BernoulliNB()
clf.fit(train_data_X, train_data_Y)



##########################
# predict and error rate #
##########################

test_words = []
for i in range(2):
    filenames = os.listdir('email' + '/' + foldernames[i+1])
    wordsi = [get_word(os.path.join('email', foldernames[i+1], filenames[j])) for j in range(15,25)]
    test_words += wordsi

test_data_X = [vocabvec(word, list(vocab)) for word in test_words]
result_data_Y = clf.predict(test_data_X)
real_data_Y = [0] * 10 + [1] * 10

error_data = [0] * 20
for i in range(20):
    if result_data_Y[i] != real_data_Y[i]:
        error_data[i] = 1

error_rate = error_data.count(1)/float(len(error_data))
print 'error_rate is %f' % error_rate