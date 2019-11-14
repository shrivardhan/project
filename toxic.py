
import sys, os, re, csv, codecs, numpy as np, pandas as pd
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, GRU
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
# import matplotlib.pyplot as plt
import gensim.models.keyedvectors as word2vec
import gc
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = 1
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

    def on_test_begin(self, logs={}):
        return

    def on_test_end(self, logs={}):
        return

    def on_test_batch_begin(self, batch, logs={}):
        return

    def on_test_batch_end(self, batch, logs={}):
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_batch_begin(self, batch, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_train_batch_end(self, batch, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        super(Callback, self).__init__()

    def on_test_begin(self, logs={}):
        return

    def on_test_end(self, logs={}):
        return

    def on_test_batch_begin(self, batch, logs={}):
        return

    def on_test_batch_end(self, batch, logs={}):
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_batch_begin(self, batch, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_train_batch_end(self, batch, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



def loadEmbeddingMatrix(typeToLoad):
        #load different embedding file from Kaggle depending on which embedding
        #matrix we are going to experiment with
        if(typeToLoad=="glove"):
            EMBEDDING_FILE='./glove.twitter.27B.100d.txt'
            embed_size = 100
        elif(typeToLoad=="word2vec"):
            word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)
            embed_size = 300
        elif(typeToLoad=="fasttext"):
            EMBEDDING_FILE='../input/fasttext/wiki.simple.vec'
            embed_size = 300

        if(typeToLoad=="glove" or typeToLoad=="fasttext" ):
            embeddings_index = dict()
            #Transfer the embedding weights into a dictionary by iterating through every line of the file.
            f = open(EMBEDDING_FILE)
            for line in f:
                #split up line into an indexed array
                values = line.split()
                #first index is word
                word = values[0]
                #store the rest of the values in the array as a new array
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs #50 dimensions
            f.close()
            print('Loaded %s word vectors.' % len(embeddings_index))
        else:
            embeddings_index = dict()
            for word in word2vecDict.wv.vocab:
                embeddings_index[word] = word2vecDict.word_vec(word)
            print('Loaded %s word vectors.' % len(embeddings_index))

        gc.collect()
        #We get the mean and standard deviation of the embedding weights so that we could maintain the
        #same statistics for the rest of our own random generated weights.
        # print(len(list(embeddings_index.values())))
        # temp =0
        # for i in list(list(embeddings_index.values())):
        #     if not len(i) == 100 :
        #         print(len(i))
        #         print(temp)
        #     temp = temp + 1
        print(len(list(embeddings_index.values())))
        all_embs = np.stack([x for i,x in enumerate(list(embeddings_index.values())) if len(x)!=100])
        print(len([x for i,x in enumerate(list(embeddings_index.values())) if len(x)!=100]))
        # all_embs = np.stack(list(embeddings_index.values()))
        emb_mean,emb_std = all_embs.mean(), all_embs.std()

        nb_words = len(tokenizer.word_index) + 1
        #We are going to set the embedding size to the pretrained dimension as we are replicating it.
        #the size will be Number of Words in Vocab X Embedding Size
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        gc.collect()

        #With the newly created embedding matrix, we'll fill it up with the words that we have in both
        #our own dictionary and loaded pretrained embedding.
        embeddedCount = 0
        for word, i in tokenizer.word_index.items():
            i-=1
            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
            embedding_vector = embeddings_index.get(word)
            #and store inside the embedding matrix that we will train later on.
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                embeddedCount+=1
        print('total embedded:',embeddedCount,'common words')

        del(embeddings_index)
        gc.collect()

        #finally, return the embedding matrix
        return embedding_matrix




train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# print(train.head())

train.isnull().any(),test.isnull().any()

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

max_features = 20000
# tokenizer = Tokenizer(num_words=max_features)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(list_sentences_train))

# print(len(tokenizer.word_counts))
count_thres = 20
low_count_words = [w for w,c in tokenizer.word_counts.items() if c < count_thres]
# print(tokenizer.texts_to_sequences(comments))
for w in low_count_words:
    del tokenizer.word_index[w]
    del tokenizer.word_docs[w]
    del tokenizer.word_counts[w]
# print(len(tokenizer.word_counts))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

# print(len(X_t))

# totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
# vocab_size = len(tokenizer.word_index) + 1
# embeddings_index = dict()
# f = open('./glove.twitter.27B.100d.txt')
# for line in f:
# 	values = line.split()
# 	word = values[0]
# 	coefs = asarray(values[1:], dtype='float32')
# 	embeddings_index[word] = coefs
# f.close()
#
# isNone = 0
# notNone = 0
# embedding_matrix = zeros((vocab_size, 100))
# for word, i in tokenizer.word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         notNone = notNone + 1
#         embedding_matrix[i] = embedding_vector
#     else:
#         isNone = isNone +1

# print(vocab_size)
# print(notNone)
# print(isNone)

inp = Input(shape=(maxlen, ))
embed_size = 128
# print(len(inp))
# x = Embedding(max_features, embed_size)(inp)
load_embedding_matrix = loadEmbeddingMatrix('glove')
# x = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)(inp)
print('dropout')
print(0.5)
print('lstm dropout')
print(0.5)
x = Embedding(len(tokenizer.word_index)+1, load_embedding_matrix.shape[1],weights=[load_embedding_matrix],trainable=False)(inp)
# x = Embedding(len(tokenizer.word_index)+1, 100, embeddings_initializer='glorot_normal')(inp)
x = Dropout(0.5)(x)
# x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
# x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.5, recurrent_dropout=0.5,name='lstm_layer'))(x)
# x = GRU(60, return_sequences=True,name="gru_layer")(x)
x = GRU(60, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, name="gru_layer")(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size = 100
epochs = 100
# model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# RocAuc = roc_callback((X_t, y),(,))
X_tra,X_val,Y_tra,Y_val = train_test_split(X_t,y,train_size=0.75,random_state=233)

RocAuc = RocAucEvaluation(validation_data=(X_val, Y_val), interval=1)
model.fit(X_tra, Y_tra, batch_size=batch_size, epochs=epochs, validation_data=[X_val,Y_val],callbacks=[RocAuc], verbose=1)
