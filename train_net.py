
# coding: utf-8

# In[1]:

import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def embedding(indcs, padding):
    res = np.array([])
    size = len(indcs)
    
    for i in range(size):
        l = len(indcs[i])
        res = np.append(res, np.pad(indcs[i], (0, padding - l), 'constant', constant_values=0))
    
    #size = len(indcs)
    #res = np.zeros([ size, dict_size ])
    #for i in range(size):
    #    ind = indcs[i]
    #    res[i][ind] = 1
    return res.reshape(size, padding)

def prepare_dataset(df):
    df['platform'] = df['platform'].apply(pd.to_numeric).astype(float)
    #df['platform'] = (df['platform'].apply(pd.to_numeric).astype(float) - 1.5) / 1.5
    df['advertiser'] = (df['advertiser'].astype(float) - 2266) / 2266
    df['province'] = (df['province'].astype(float) - 202) / 202
    df['state'] = (df['state'].astype(float) - 126) / 116
    df['month'] = (df['month'].astype(float) - 6) / 6
    df['dow'] = (df['dow'].astype(float) - 3.5) / 3.5
    
    size = len(df)
    
    
    Y = df['clicked'].values
    
    
    scalar_X = df[['platform', 'dow', 'month', 'state', 'province', 'advertiser']].values
    
    topics_ind = df['topics'].apply(lambda x: x.split(',')).apply(pd.to_numeric).values
    topics_X = embedding(topics_ind, 40)
        
    categories_ind = df['categories'].apply(lambda x: x.split(',')).apply(pd.to_numeric).values
    halfmax = 2100 / 2
    categories_X = (embedding(categories_ind, 2) - halfmax) / halfmax
    
    return (Y, scalar_X, topics_X, categories_X)

import thread
import Queue

    
# In[5]:

import time

current_time = lambda: int(round(time.time()))

def DataLoader(size, max_frames):
   
    for tf in pd.read_csv('data/train_set.csv', chunksize = size):
        max_frames -= 1
        if max_frames == 0:
            break

        yield tf

import thread
import Queue

FRAME_SIZE = 10000
MAX_FRAMES = 5
NB_EPOCH = 30
PROCESSORS_COUNT = 10

def thread_loader(q, nb_epoch):
    for i in range(nb_epoch):
        for tf in DataLoader(FRAME_SIZE, MAX_FRAMES):
            try:
                q.put( tf , block = True)
            except ValueError:
                pass  

def thread_data_processing(q_in, q_out):
    while True:
        try:
            frame = q_in.get(block = True, timeout = 5 * 60)
            q_out.put( prepare_dataset(frame), block = True)
        except Queue.Empty:
            break
        except:
            pass        

csv_queue = Queue.Queue(maxsize = 10)
frame_queue = Queue.Queue(maxsize = 10)

thread.start_new_thread(thread_loader, (csv_queue, NB_EPOCH))

for i in range(PROCESSORS_COUNT):
    thread.start_new_thread(thread_data_processing, (csv_queue, frame_queue))


# In[94]:

from keras.layers import Input, Embedding, Dense, merge, Flatten, Dropout, Activation
from keras.models import Model
from keras.regularizers import l2, activity_l2, l1
from keras.layers.normalization import BatchNormalization

topic_inp = Input(shape=(40,))
x_topic = Embedding(output_dim=10, input_dim=3000)(topic_inp)
topic_out = Flatten()(x_topic)

category_inp = Input(shape=(2,))

sc_inp = Input(shape=(6,))

x = merge([topic_out, category_inp, sc_inp], mode='concat')

x = Dense(30, W_regularizer=l1(0.01))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.1)(x)

#x = Dense(30)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
#x = Dropout(0.1)(x)

#x = Dense(10)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)

main_output = Dense(1, activation='hard_sigmoid', name='main_output')(x)

model = Model(input=[topic_inp, category_inp, sc_inp], output=[main_output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

batch_count = 0
epoch_counter = 0
while True:
    try:
        if (batch_count % MAX_FRAMES) == 0:
            epoch_counter += 1
            print "EPOCH %d" % epoch_counter

        train_Y, train_X, topics_X, categories_X = frame_queue.get(block = True, timeout = 60 * 5)
        model.fit([topics_X, categories_X, train_X], train_Y, nb_epoch=1, batch_size=64, verbose=1)
       
        batch_count += 1
        
    except Queue.Empty:
        break

print "Totall batches: %d" % batch_count


# In[101]:

yaml = model.to_yaml()
f = open('model.yaml', 'w')
f.write(yaml)
f.close()

model.save_weights('model.hd5')


# In[ ]:
engine = create_engine('postgresql://postgres@localhost:5432/click_pred')    

sql = 'SELECT * FROM train_set OFFSET 10000 LIMIT 10000;'
validation_frame = pd.read_sql(sql, engine)


# In[96]:

valid_Y, valid_X, valid_topics_X, valid_categories_X = prepare_dataset(validation_frame)


# In[97]:

prediction = model.predict([valid_topics_X, valid_categories_X, valid_X])


# In[98]:

correct = np.logical_and(prediction.reshape(-1) >= 0.4, valid_Y == 1)
#correct.shape


# In[99]:

print "Validation: %f " % ((float(np.count_nonzero(correct)) / len(correct)) * 100)


# In[100]:

#0.935


# In[ ]:



