import pandas as pd
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

df=pd.read_csv('C:\\Users\\dell\\Downloads\\compiled_aggressive.csv')
list_classes = ["aggressive"]
y = df[list_classes].values

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer=WordNetLemmatizer()
stop_words=stopwords.words('english')

token_list=[]
sentences=[]
token2_list=[]
lemma_list=[]
lemma2_list=[]
blank_list=[]
for i,m in zip(df['commentText'],df['replies.commentText']):
    if(type(i) is not float):
        #print(i)

        i=i.lower()
        j=nltk.word_tokenize(i)
        words = [word for word in j if word.isalpha()and word not in stop_words]
        token_list.append(words)
        lemmas=[lemmatizer.lemmatize(word) for word in words]
        lemmas=[stemmer.stem(word) for word in words]
        
        lemma_list.append(lemmas)
        
        if(len(lemmas)!=0):
            #print(lemmas)
            sentences.append(lemmas)
        else:
            sentences.append(blank_list)
        #print(sentences)
        token2_list.append(blank_list)
        lemma2_list.append(blank_list)
        #print(words)
    else:
        m=m.lower()
        j=nltk.word_tokenize(m)
        words = [word for word in j if word.isalpha()and word not in stop_words]
        token2_list.append(words)
        lemmas=[lemmatizer.lemmatize(word) for word in words]
        lemmas=[stemmer.stem(word) for word in words]
        
        if(len(lemmas)!=0):
            #print(lemmas)
            sentences.append(lemmas)
        else:
            sentences.append(blank_list)
        lemma2_list.append(lemmas)
        token_list.append(blank_list)
        lemma_list.append(blank_list)
        



df['commentText']=lemma_list
df['replies.commentText']=lemma2_list

#print(sentences)
maxlen=50
max_features=300
batch_size = 128
epochs = 2

tokenizer = Tokenizer(num_words=maxlen)

tokenizer.fit_on_texts(sentences)

list_tokenized_train = tokenizer.texts_to_sequences(sentences)

X_tr = pad_sequences(list_tokenized_train ,maxlen=maxlen,padding='post', truncating='post')


#model = Word2Vec(sentences, min_count=1,size=300)

#print(model)

#words = list(model.wv.vocab)
#print(words)

inp = Input(shape=(maxlen,))
x = Embedding(5052,max_features,input_length=maxlen)(inp)
x = Bidirectional(LSTM(batch_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(batch_size, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [early]
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
model.summary()
model.fit(X_tr, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)




