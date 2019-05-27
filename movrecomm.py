 #script that reads in a data set of movies and recommends a movie
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
data = fetch_movielens(min_rating = 4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#store model in variable'model'
model  = LightFM(loss='warp') #Weighted Approximate-Rank Pairwise

#now we wil train the model
model.fit(data['train'],epochs = 30, num_threads=2)
#epoch=no. of runs threads = no. of parallel computation


#after training generate a recommendation
def sample_recommendation(model,data, user_ids):
	#no. of users and movies in training data
	n_users, n_items = data['train'].shape

	#generate recommendations for each user we input
	# say from for loops we want to know the list of known positives
	for user_id in user_ids:

		#movies they already likes
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#movies our model predicts they will like
		scores = model.predict(user_id, np.arange(n_items))

		#rank them in order of most liked to least
		top_items = data['item_labels'][np.argsort(-scores)]

		#print ort the results
		print("User %s" % user_id)
		print("		Known positives:")

		for x in known_positives[:3]:
			print("			%s" % x)

		print("			Recommended:")

		for x in top_items[:3]:
			print("				%s" % x)


sample_recommendation(model, data, [3, 25, 450])
