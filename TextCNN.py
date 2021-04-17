#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datatable as dt
import numpy as np
import tensorflow as tf
import os, gc, random, time
import pandas as pd
import gensim

from gensim.models import Word2Vec
from tensorflow.keras.layers import (
    Bidirectional,
    Embedding,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Concatenate,
    SpatialDropout1D,
    BatchNormalization,
    Dropout,
    Dense,
    Conv1D,
    Activation,
    Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), len(logical_gpus))
    except RuntimeError as e:
        print(e)


# In[ ]:


train_df = dt.fread('/home/liuchh/kaggle/input/train_set.csv', sep='\t').to_pandas()
test_df = dt.fread('/home/liuchh/kaggle/input/test_a.csv', sep='\t').to_pandas()

new_data = np.load('/home/liuchh/kaggle/input/pl_ensemble_0.95.npy')
new_data_x = test_df.iloc[new_data[:,0]].text.values
new_data_y = new_data[:,1]


# In[ ]:


tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=7000,
    lower=False,
    filters=""
)
tokenizer.fit_on_texts(list(train_df['text'].values) + list(test_df['text'].values))
train_ = tokenizer.texts_to_sequences(train_df['text'])
test_ = tokenizer.texts_to_sequences(test_df['text'])
new_ = tokenizer.texts_to_sequences(new_data_x)
train_ = tf.keras.preprocessing.sequence.pad_sequences(train_, maxlen=2400)
test_ = tf.keras.preprocessing.sequence.pad_sequences(test_, maxlen=2400)
new_ = tf.keras.preprocessing.sequence.pad_sequences(new_,maxlen=2400)
word_vocab = tokenizer.word_index


# In[ ]:


all_data = pd.concat([train_df['text'], test_df['text']])
file_name = '/home/liuchh/kaggle/input/word2vec.bin'
if not os.path.exists(file_name):
    print('Training Word2Vec ......')
    model = Word2Vec(
        [[word for word in document.split(' ')] for document in all_data.values],
        size=200,
        window=5,
        iter=10,
        workers=12,
        seed=2021,
        min_count=2
    )
    model.save(file_name)
else:
    print('Loading Word2Vec ......')
    model = Word2Vec.load(file_name)
print('Add word2vec finished ......')


# In[ ]:


Glove_model = gensim.models.KeyedVectors.load_word2vec_format('/home/liuchh/kaggle/input/Glove_200.txt',binary=False)

count = 0
embedding_matrix = np.zeros((len(word_vocab) + 1, 400))
for word, i in word_vocab.items():
    embedding_vector = np.concatenate((model.wv[word],Glove_model[word])) if word in model.wv else None
    if embedding_vector is not None:
        count += 1
        embedding_matrix[i] = embedding_vector
    else:
        unk_vec = np.random.random(400) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_matrix[i] = unk_vec


# In[ ]:


def TextCNN(sent_length, embeddings_weight):
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=True)
    x = SpatialDropout1D(0.2)(embedding(content))
    convs = []
    for kernel_size in [2,3,4,5]:
        c = Conv1D(1024, kernel_size, activation='relu')(x)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)
    x = Dense(512,activation="relu")(x)
    output = Dense(14, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=content, outputs = output)
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    return model


# In[ ]:


def category_performance_measure(labels_right, labels_pred):
    text_labels = list(set(labels_right))
    test_pred_labels = list(set(labels_pred))
    
    TP = dict.fromkeys(text_labels, 0)
    TP_FP = dict.fromkeys(text_labels, 0)
    TP_FN = dict.fromkeys(text_labels, 0)
    
    for i in range(0, len(labels_right)):
        TP_FP[labels_right[i]] += 1
        TP_FN[labels_right[i]] += 1
        if labels_right[i] == labels_pred[i]:
            TP[labels_right[i]] += 1
        
    for key in TP_FP:
        P = float(TP[key]) / float(TP_FP[key] + 1)
        R = float(TP[key]) / float(TP_FN[key] + 1)
        F1 = P * R * 2 / (P + R) if (P + R) != 0 else 0
        print("%s:\t P:%f\t R:%f\t F1:%f" % (key,P,R,F1))


# In[ ]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2021)

train_label = train_df['label'].values
train_label = to_categorical(train_label)
new_data_y = to_categorical(new_data_y)


# In[ ]:


from sklearn.model_selection import train_test_split
with tf.device('/gpu:1'):
    X_train, X_valid, y_train, y_valid = train_test_split(train_, train_label, shuffle=True, random_state=2021, stratify=train_label)
    
#     X_train = train_
#     y_train = train_label
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(128)
    val_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(128)
    test_ds = tf.data.Dataset.from_tensor_slices((test_, np.zeros((test_.shape[0], 14)))).batch(128)
    
    checkpoint_dir = './TextCNN_400_cv_finetune_checkpoints/cv_'+str(i)+'/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    model = TextCNN(2400, embedding_matrix)
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    plateau = ReduceLROnPlateau(monitor='val_accuracy', verbose=1, mode='max', factor=0.5, patience=3)
    checkpoint = ModelCheckpoint(checkpoint_prefix, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max', save_weights_only=True)
    
    model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[early_stopping, plateau, checkpoint],
        verbose=1
    )
    
    valid_prob = model.predict(val_ds)
    valid_pred = np.argmax(valid_prob,axis=1)
    y_valid = np.argmax(y_valid, axis=1)

    f1_score_ = f1_score(y_valid,valid_pred,average='macro') 
    print ("valid's f1-score: %s" %f1_score_)
    
    test_pre_matrix = model.predict(test_ds)
    
    np.save("TextCNN_128_400finetune_test_result.npy", test_pre_matrix)
    
    model.save('my_model_no_cv.h5')
    del model; gc.collect()
    tf.keras.backend.clear_session()

