import pandas as pd
import numpy as np
import cPickle
import thread
import Queue
import time

from sqlalchemy import create_engine

FRAME_SIZE = 10000
MAX_FRAMES = 50000
NB_EPOCH = 1
PROCESSORS_COUNT = 10
submission_queue = Queue.Queue(maxsize=10)

OUTPUT_FILE = 'submission_result.csv'

engine = create_engine('postgresql://postgres@localhost:5432/click_pred') 

mdodel_pkl = open('mnb_classifier.pkl', 'r') 
model = cPickle.load(mdodel_pkl) 

print "Classes count %d" % len(model.classes_)

current_time = lambda: int(round(time.time() * 1000))

def prepare_dataset(df):

	df = df.replace('\N', 4)
	#X_adv = df['advertiser'].astype(float).values.reshape(-1, 1)
	X_month = df['month']
	X_dow = df['dow']
	X_state = df['state']
	X_platform = float(df['platform'])
      
   	X = np.array([X_dow, X_state, X_platform, X_month])
      
	return X

TRAIN_SET_FILE_NAME = "data/sample_submission.csv"

def dataLoader(size, max_frames):
   
    for tf in pd.read_csv(TRAIN_SET_FILE_NAME, chunksize = size):
        if max_frames == 0:
            break

        max_frames -= 1

        yield tf.fillna(value = 0)

def query_add_info(frame):
	start = current_time()
	display_ids = ', '.join(frame['display_id'].astype(str))
	sql = '''
		SELECT 
  			
  			events.display_id as display_id,
			events.platform as platform,
				
			EXTRACT(DOW FROM to_timestamp(events.timestamp + 1465876799998) ) as dow,
			EXTRACT(MONTH FROM to_timestamp(events.timestamp + 1465876799998) ) as month,
	
			countries_view.index as state,
			provinces_view.index as province
	
			 
		FROM events
	
		LEFT JOIN countries_view ON (split_part(events.geo_location, '>', 1) = countries_view.state)
		LEFT JOIN provinces_view ON (split_part(events.geo_location, '>', 2) = provinces_view.province)

		WHERE events.display_id in (%s)
		;
	''' % display_ids

	res = pd.read_sql(sql, engine).fillna(value = 0)

	print "Query parameters, done in %d ms" % (current_time() - start)
	return res


FIRST_ITERATION = True

def predict(ad_ids_str, addinf):

	#start = current_time()

	ad_ids = np.array(map(int, ad_ids_str.split()))
	X = prepare_dataset(addinf)

	#print "ad_ids: "
	#print ad_ids
	#print

	#print "X: "
	#print X
	#print
	indcs = np.in1d(model.classes_, ad_ids)
	indcs = np.where(indcs)

	#print "Indicies:" 
	#print indcs
	#print

	prob = model.predict_proba([X])[0]
	prob = prob[indcs]

	#print "Proba:" 
	#print prob
	#print

	res = ' '.join( map(str, ad_ids[prob.argsort()]) )

	#print "Result: "
	#print res
	#print


	#print "Prediction done in %d ms" % (current_time() - start)
	#print '------------------------------------------------'
	#print


	return res

	

for frame in dataLoader(FRAME_SIZE, MAX_FRAMES):
	addinfo = query_add_info(frame)
	
	start = current_time()

	merged = pd.merge(frame, addinfo, on='display_id')

	print "Merge done in %d ms" % (current_time() - start)

	start = current_time()

	for row in merged.iterrows():
		#nv = predict(row[1]['ad_id'], addinfo[addinfo['display_id'] == row[1]['display_id']] )
		nv = predict(row[1]['ad_id'], row[1] )
		frame.set_value(row[0], 'ad_id', nv)

	print "Prediction on frame done in %d ms" % (current_time() - start)
	
	if FIRST_ITERATION:
		frame.to_csv(OUTPUT_FILE, header = True, index=False, columns = ['display_id', 'ad_id'] )
		FIRST_ITERATION = False
	else:
		frame.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, columns = ['display_id', 'ad_id'])

	print


