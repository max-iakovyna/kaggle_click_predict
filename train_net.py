
# coding: utf-8

# In[1]:

import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import gc


TRAIN_SET_FILE_NAME = 'data/train_set.csv'
FRAME_SIZE = 1000
MAX_FRAMES = 50000
NB_EPOCH = 1
PROCESSORS_COUNT = 10

def prepare_dataset(df):
    df.replace('\N', 4, inplace = True)
    #X_adv = df['advertiser'].astype(float).values.reshape(-1, 1)
    X_dow = df['dow'].astype(float).values.reshape(-1, 1)
    X_month = df['month'].astype(float).values.reshape(-1, 1)
    X_state = df['state'].values.reshape(-1, 1)
    X_platform = df['platform'].astype(float).values.reshape(-1, 1)
      
    Y = df['ad_id'].apply(pd.to_numeric).values

    X = np.concatenate((X_dow, X_state, X_platform, X_month), axis=1)
      
    return (X, Y)

import thread
import Queue



data = pd.read_csv(TRAIN_SET_FILE_NAME)
classes = data['ad_id'].unique().astype(int)

print 'Nuber of classes %d' % len(classes)
    
# In[5]:

import time

current_time = lambda: int(round(time.time() * 1000))

TRAIN_SET_FILE_NAME = 'data/train_set.csv'

def DataLoader(size, max_frames):
   
    for tf in pd.read_csv(TRAIN_SET_FILE_NAME, chunksize = size):
        if max_frames == 0:
            break

        max_frames -= 1

        yield tf

import thread
import Queue


def thread_loader(q, nb_epoch):
    for i in range(nb_epoch):
        for tf in DataLoader(FRAME_SIZE, MAX_FRAMES):
            try:
                q.put( (tf, False) , block = True)
            except ValueError:
                pass 

    q.put( (None, True) , block = True) 

def thread_data_processing(q_in, q_out):
    while True:
        try:
            frame, poisen_pill = q_in.get(block = True, timeout = 1 * 60)
            #if poisen_pill:
            #    q_out.put( (None, poisen_pill), block = True)
            #    break
            frame.fillna(value = 0, inplace = True)    
            q_out.put( (prepare_dataset(frame), poisen_pill), block = True)
        except Queue.Empty:
            break
        except:
            print "Exception"
            pass        

csv_queue = Queue.Queue(maxsize = 10)
frame_queue = Queue.Queue(maxsize = 10)

thread.start_new_thread(thread_loader, (csv_queue, NB_EPOCH))

for i in range(PROCESSORS_COUNT):
    thread.start_new_thread(thread_data_processing, (csv_queue, frame_queue))


from sklearn.naive_bayes import MultinomialNB

bnb = MultinomialNB()

batch_count = 0
epoch_counter = 0


while True:
    try:
        if (batch_count % MAX_FRAMES) == 0:
            epoch_counter += 1
            print "EPOCH %d" % epoch_counter

        start = current_time()    
            
        XY, poisen_pill = frame_queue.get(block = True, timeout = 60 * 5)

        if poisen_pill:
            print 'Got poisen pill'
            break

        train_X, train_Y = XY

        bnb.partial_fit(train_X, train_Y, classes = classes)
       
        batch_count += 1

        print "Frame processed in: %d" % (current_time() - start)

        #gc.collect()
        
    except Queue.Empty:
        break

print "Totall batches: %d" % batch_count

import cPickle
# save the classifier
with open('mnb_classifier.pkl', 'wb') as fid:
    cPickle.dump(bnb, fid)    



#print "Classes %d" % len(bnb.classes_)

#validation_frame = pd.read_csv('data/train_set.csv', skiprows = 70000000, nrows = 100000)

#valid_X, valid_Y = prepare_dataset(validation_frame)


# In[97]:


#print "Score: %f " % bnb.score(valid_X, valid_Y)


# In[100]:

#0.935


# In[ ]:



