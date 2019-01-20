import pandas as pd 
import numpy as np 
from data_reader import *
from evaluate_new import *
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn import metrics 
from keras_contrib.layers import CRF
from sklearn.metrics import f1_score
import keras
import re
import os

doc_dir = "../../data/test_docs/"
def get_doc_test(gold, text):
    ## gold: gold data
    ## text: full text file
    test_labels = []
    test_doc = []
    with open(doc_dir+gold, 'r') as doc_labels, open(doc_dir+text, 'r') as doc_text:
        d_labels = doc_labels.readlines()
        d_text = doc_text.readlines()
        assert len(d_labels) == len(d_text), "Mismatch"
        for i in range(len(d_labels)):
            ## label: start_id end_id data_id pub_id
            test_labels.append(d_labels[i].strip())
            
            text = d_text[i].strip()
            text = re.sub('\d', '0', text)
            text = re.sub('[^ ]- ', '', text)
            
            test_doc.append(text)
    return test_labels, test_doc

def read_doc(doc, labels):
    doc = doc.strip().split()
    labels = labels.strip().split('|')
    labels = [la.split() for la in labels]
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            labels[i][j] = int(labels[i][j])

    res_labels = [0]*len(doc)
    for la in labels:
        if la[2]!=0:
            start = la[0]
            end = la[1]
            res_labels[start : end+1] = [1]*(end+1-start)
    return [(doc[i], str(res_labels[i])) for i in range(len(doc))]

# predict one doc
def doc_pred(model, doc, MAXLEN):
    splits = []
    for i in range(0, len(doc), MAXLEN):
        splits.append(doc[i : i+MAXLEN])
    splits = tokenizer.texts_to_sequences(splits)
    splits = pad_sequences(splits, maxlen=MAXLEN)
    preds = model.predict(splits)
    
    preds = [1 if p>=0.5 else 0 for pd in preds for p in pd]
    return preds

def doc_eval(model, doc_test, doc_out_dir, gold_dir, MAXLEN=30):
    '''
    model: trained model 
    doc_test: processed doc test input
    doc_out_dir: dir to store predicted results
    gold_dir: gold data dir, for evaluation
    prediction format: start_id end_id
    '''
    doc_preds = [doc_pred(model, d, MAXLEN) for d in doc_test]
    doc_preds = [[int(a) for a in x] for x in doc_preds]
    with open(doc_out_dir, 'w') as fout:
        for i in range(len(doc_preds)):
            first = 0
            j = 0
            string = ''
            no_mention = True
            while j<len(doc_preds[i]):
                while j<len(doc_preds[i]) and doc_preds[i][j]== 0:
                    j+=1
                if j<len(doc_preds[i]) and doc_preds[i][j] == 1:
                    no_mention=False
                    start = j
                    while j+1<len(doc_preds[i]) and doc_preds[i][j+1]==1:
                        j+=1
                    end = j 
                    if first > 0:
                        string += " | "
                    string += (str(start)+' '+str(end))
                    j+=1
                    first += 1
            if no_mention:
                fout.write("-1 -1"'\n')
            else:
                fout.write(string+'\n')
    print ('evaluating data from: ', doc_out_dir)
    print ('doc exact: ', doc_exact_match(doc_out_dir, gold_dir))
    print ('doc partial: ', doc_partial_match(doc_out_dir, gold_dir))


def sent2labels(sent):
    return [int(label) for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

# ##load glove
embedding_index = {}
f = open('../../glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

##hyperparameters
vocab_size = 100000
maxlen = 30
emb_dim = 300


def run(train_dir, val_dir):
    train_sents = get_sents(train_dir)
    val_sents = get_sents(val_dir)
    
    X_train = [sent2tokens(s) for s in train_sents]
    Y_train = [sent2labels(s) for s in train_sents]

    X_val = [sent2tokens(s) for s in val_sents]
    Y_val = [sent2labels(s) for s in val_sents]
    
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    print ('Total vocab found: ', len(word_index))
    
    embedding_matrix = np.zeros((vocab_size, emb_dim))
    counter = 0
    for word, i in word_index.items():
        if i >= vocab_size:
            break
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            counter += 1
        else:
            embedding_matrix[i] = np.random.randn(emb_dim)
    print ("{}/{} words covered in glove".format(counter, vocab_size))
    
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)

    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_val = pad_sequences(X_val, maxlen=maxlen)
        
    Y_train = np.asarray(Y_train)
    Y_val = np.asarray(Y_val)
    
    Y_train = np.expand_dims(Y_train, axis=2)
    Y_val = np.expand_dims(Y_val, axis=2)
    
    ##build model
    input = Input(shape=(maxlen,))
    model = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(100, return_sequences=True))(model)
    out = TimeDistributed(Dense(1, activation='sigmoid'))(model)

    model = Model(input, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    earlyStop = [EarlyStopping(monitor='val_loss', patience=1)]
    history = model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_data=(X_val, Y_val), 
        callbacks=earlyStop) 
    
    Y_pred = model.predict(X_val)
    print (f1_score(Y_val, Y_pred))
    
    return model, history


if __name__=='__main__':
	doc_test_y, doc_test_x = get_doc_test('test_doc_gold', 'test_docs')
	doc_tests = [read_doc(doc_test_x[d], doc_test_y[d]) for d in range(len(doc_test_x))]
	doc_tests = [sent2tokens(s) for s in doc_tests]

	zero_shot_y, zero_shot_x = get_doc_test('zero_shot_doc_gold', 'zero_shot_docs')
	zero_shot_tests = [read_doc(zero_shot_x[d], zero_shot_y[d]) for d in range(len(zero_shot_x))]
	zero_shot_tests = [sent2tokens(s) for s in zero_shot_tests]

	for i in [400, 300, 200, 100, 10]:
		print ("Neg ratio: {}/1000".format(i))
		DIR = '../../data/data_30_'+str(i)+'neg/'
	    out = '../../BiLSTM_outputs/'
	    if not os.path.exists(out):
	        os.makedirs(out)
	    model, history = run(DIR+'train.txt', DIR+'validate.txt')
	    doc_eval(model, doc_tests, out+'doc_30_'+str(i)+'neg', '../../data/test_docs/test_doc_gold')
	    doc_eval(model, zero_shot_tests, out+'zeroshot_30_'+str(i)+'neg', '../../data/test_docs/zero_shot_doc_gold')













