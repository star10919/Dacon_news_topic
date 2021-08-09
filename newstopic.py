import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from icecream import ic

# Data
train      = pd.read_csv('./dacon/data/train_data.csv', header=0, encoding='UTF8')
test       = pd.read_csv('./dacon/data/test_data.csv', header=0, encoding='UTF8')
submission = pd.read_csv('./dacon/data/sample_submission.csv')
topic_dict = pd.read_csv('./dacon/data/topic_dict.csv')

ic(train, test)

# 정규화
def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

ic(train, test)

train_text = train["cleaned_title"].tolist()
test_text = test["cleaned_title"].tolist()

# ic(train_text)

train_label = np.asarray(train.topic_idx)

# 벡터화
# tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=150000, binary=False)
tfidf = TfidfVectorizer(analyzer='char_wb', sublinear_tf=True, ngram_range=(1, 2), max_features=45000, binary=False)
# tfidf = TfidfVectorizer(analyzer='char', sublinear_tf=True, ngram_range=(1, 2), max_features=45000, binary=False)

tfidf.fit(train_text)

train_tf_text = tfidf.transform(train_text).astype('float32')
test_tf_text  = tfidf.transform(test_text).astype('float32')
y_train = np.array([x for x in train['topic_idx']])
ic(train_tf_text.shape, test_tf_text.shape)     #(45654, 45000), (9131, 45000)
# ic(train_tf_text[:1])
ic(train_label.shape)    #(45654,)


# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(100, input_dim=45000, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(49, activation='relu'))
model.add(Dense(7, activation='softmax'))


from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
# optimizer = Adam(lr=0.01)    # learning_rate (커스터마이징)적용   /   learning_rate를 줄이면 epochs는 그만큼 늘려줘야 된다.(다 돌게 하려면)
# optimizer = Adagrad(lr=0.01)
# optimizer = Adadelta(lr=0.01)
# optimizer = Adamax(lr=0.01)
# optimizer = RMSprop(lr=0.01)
# optimizer = SGD(lr=0.01)
optimizer = Nadam(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])     # SGD:경사하강법

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='auto', patience=1, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=1, mode='auto', verbose=1, factor=0.1)

model.fit(train_tf_text[:45000], train_label[:45000], epochs=6, batch_size=3000, callbacks=[es, reduce_lr], validation_data=(train_tf_text[45000:], train_label[45000:]))
# model.fit(train_tf_text, train_label, epochs=15, batch_size=800, validation_split=0.2)


# Predict
y_predict = model.predict(test_tf_text)
y_predict = np.argmax(y_predict, axis=1)

# Results make to_csv submissions
# ic(len(test_tf_text))
# topic = []
# for i in range(len(test_tf_text)):
#     topic.append(np.argmax(test_tf_text[i]))   # np.argmax -> 최대값의 색인 위치

submission['topic_idx'] = y_predict
# ic(submission.shape)
ic(y_predict)



submission.to_csv('./dacon/predict43.csv', index=False)

'''
*26-0.8127053669
57/57 [==============================] - 0s 5ms/step - loss: 0.5560 - acc: 0.8287 - val_loss: 0.3668 - val_acc: 0.8945

*27-0.813143483   ***
15/15 [==============================] - 0s 11ms/step - loss: 0.4004 - acc: 0.8825 - val_loss: 0.3447 - val_acc: 0.8945

*28-0.812924425
15/15 [==============================] - 0s 10ms/step - loss: 0.4903 - acc: 0.8536 - val_loss: 0.3533 - val_acc: 0.8976

*29
15/15 [==============================] - 0s 11ms/step - loss: 0.4994 - acc: 0.8513 - val_loss: 0.3649 - val_acc: 0.8960

*30-0.8070098576
15/15 [==============================] - 0s 11ms/step - loss: 0.4953 - acc: 0.8517 - val_loss: 0.3643 - val_acc: 0.8976

*31
15/15 [==============================] - 0s 11ms/step - loss: 0.4030 - acc: 0.8798 - val_loss: 0.3421 - val_acc: 0.8914

*32-0.81117
15/15 [==============================] - 0s 11ms/step - loss: 0.4220 - acc: 0.8741 - val_loss: 0.3431 - val_acc: 0.8945

*33-0.8107338445
9/9 [==============================] - 0s 15ms/step - loss: 0.4798 - acc: 0.8555 - val_loss: 0.3593 - val_acc: 0.8991

*34-0.8094194962
12/12 [==============================] - 0s 12ms/step - loss: 0.4098 - acc: 0.8767 - val_loss: 0.3495 - val_acc: 0.8960

*35
9/9 [==============================] - 0s 16ms/step - loss: 0.4812 - acc: 0.8579 - val_loss: 0.3749 - val_acc: 0.8930

*36-0.8124863089
Epoch 18/18
9/9 [==============================] - 0s 17ms/step - loss: 0.3014 - acc: 0.9127 - val_loss: 0.3328 - val_acc: 0.8960

*37
Epoch 18/18
9/9 [==============================] - 0s 17ms/step - loss: 0.2977 - acc: 0.9138 - val_loss: 0.3512 - val_acc: 0.8884

*38
Epoch 18/18
9/9 [==============================] - 0s 16ms/step - loss: 0.2971 - acc: 0.9140 - val_loss: 0.3489 - val_acc: 0.8914

*39 - 0.80941
Epoch 18/18
9/9 [==============================] - 0s 17ms/step - loss: 0.2943 - acc: 0.9142 - val_loss: 0.3394 - val_acc: 0.8914

*40
9/9 [==============================] - 0s 16ms/step - loss: 0.2729 - acc: 0.9142 - val_loss: 0.3625 - val_acc: 0.8838

*41
Epoch 6/6
23/23 [==============================] - 0s 8ms/step - loss: 0.4214 - acc: 0.8572 - val_loss: 0.3472 - val_acc: 0.8960

*42-0.8120481928
15/15 [==============================] - 0s 10ms/step - loss: 0.5224 - acc: 0.8300 - val_loss: 0.3415 - val_acc: 0.8960
'''