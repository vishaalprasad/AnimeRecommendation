"""A class of recommender systems working on the anime dataset found at:
https://www.kaggle.com/CooperUnion/anime-recommendations-database/discussion/30036
The (abstract) base class defines many useful methods, but leaves the two main
pieces, training and predicting, to be implemented as subclasses.

For now, only the standard latent factor model, with no user or item features,
has been implemented.

Written by: Vishaal Prasad, 04/2017
"""



from collections import defaultdict

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import abc


from util.data import get_data, remove_users
from util.user_info import get_user_info


class RecSysModel(object):
	__metaclass__ = abc.ABCMeta


	def __init__(self, username, min_num):
		self.users, self.anime = get_data()
		self.username = username

		self.train = remove_users(self.users, min_num)
		self.add_to_users(username)

	#------Setup------#

	def add_to_users(self, username):
		"""Explicitly add info of user for whom we're building the RecSys."""
		ids, ratings = get_user_info(username)
		tmp = np.vstack(
				[np.tile(max(self.users.user_id)+1, len(ids)), 
				np.array(ids), np.array(ratings)])
		tmp2 = pd.DataFrame(np.transpose(tmp))
		tmp2.columns = ['user_id', 'anime_id', 'rating']
		self.train = pd.concat([self.train, tmp2], ignore_index=True)
		self.uid = max(self.users.user_id)+1
		self.seen_anime = ids


	def train_test_split(self):
		"""Split the train, test data if necessary."""
		train, self.test = train_test_split(self.users, test_size = 0.1) 
		train, self.valid = train_test_split(train, test_size = 0.2)

	#-------Methods to be defined by every subclass------#

	@abc.abstractmethod
	def train_model(self):
		"""Trains the tensorflow model."""
		pass

	@abc.abstractmethod
	def predict(self, mask):
		"""
		Predicts, for a given user, a certain set of shows. 
		Massk is an array of 0s and 1s (or Falses and Trues) of length 
		len(nAnime) representing if that anime is to be predicted or not.
		"""
		pass

	#-------Helper methods to convert anime headings-------#

	def convert_ids_to_names(self, ids):
		lookup = pd.Series(self.anime.name.values,index=self.anime.anime_id).to_dict()
		namer = lambda id_: lookup[id_]
		vfunc = np.vectorize(namer)
		return vfunc(ids)

	def convert_names_to_ids(self, names):
		lookup = pd.Series(self.anime.anime_id,index=self.anime.name.values).to_dict()
		namer = lambda name: lookup[name]
		vfunc = np.vectorize(namer)

		return vfunc(names)
 

	#-------Various "predict" wrappers------#

	def predict_unseen_shows(self):
		mask = np.zeros(len(self.anime), dtype=bool)
		for idx,anime_id in enumerate(self.anime['anime_id']):
			if anime_id not in self.seen_anime:
				mask[idx] = True
		return self.predict(mask)

	def predict_seen_shows(self):

		mask = np.zeros(len(self.anime), dtype=bool)
		for idx,anime_id in enumerate(self.anime['anime_id']):
			if anime_id in self.seen_anime:
				mask[idx] = True
		return self.predict(mask)


	def predict_title(self, title):
		
		mask = np.zeros(len(self.anime), dtype=bool)
		for idx,anime_id in enumerate( \
			self.convert_ids_to_names(self.anime['anime_id'])):
			if anime_id  == title:
				mask[idx] = True
				return self.predict(mask)

		print('Warning: No show with title "%s" found.' %title)
		print('Ensure that the name matches exactly the MAL title.')
		return





class LatentFactorMethod(RecSysModel):
	def __init__(self, username, k=3, min_num=10):
		super().__init__(username, min_num)
		self.k = k

	#-------Create Local Data Structures to vectorize computations------#

	def setup_user_index(self):
		"""Creates a datastructure that maps a user_id to an index
		for standard arrays."""
		user_ids = self.train.user_id
		user_index = defaultdict(lambda: -1)
		counter = 0

		for user in user_ids:
			if user_index[user] == -1:
				user_index[user] = counter
				counter += 1 

		self.user_index = user_index

	def setup_item_index(self):
		"""Creates a datastructure that maps an item_id to an index
		for standard arrays."""
		item_ids = self.train.anime_id
		item_index = defaultdict(lambda: -1) # maps an anime_id to the index.
		counter = 0

		for item in item_ids:
			if item_index[item] == -1:
				item_index[item] = counter
				counter += 1 

		self.item_index = item_index


	#-------Setup various TensorFlow components------#

	def setup_tf_vars(self):
		"""Declares tensorflow variables."""
		nUsers = len(self.train.user_id.unique())
		nAnime = len(self.train.anime_id.unique())

		self.alpha = tf.Variable(tf.constant([6.9], shape=[1, 1]))
		self.Bi = tf.Variable(tf.constant([0.0]*nAnime, shape=[nAnime, 1]))
		self.Bu = tf.Variable(tf.constant([0.0]*nUsers, shape=[nUsers, 1]))
		self.Gi = tf.Variable(tf.random_normal([nAnime, self.k], stddev=0.35))
		self.Gu = tf.Variable(tf.random_normal([nUsers, self.k], stddev=0.35))

		self.y = tf.cast(tf.constant(self.train['rating'].as_matrix(), 
									shape=[len(self.train),1]), tf.float32)


	def objective(self, alpha, Bi, Bu, Gi, Gu, y, lam):
		"""Definition of MAE objective function, with L2 normalization."""
		train = self.train
		item_index = self.item_index
		user_index = self.user_index

		pred = tf.gather(Bi, train.anime_id.map( #item biases
							lambda id_: item_index[id_]).as_matrix()) 
		pred += tf.gather(Bu, train.user_id.map( #user biases
							lambda id_: user_index[id_]).as_matrix()) 

		Gi_full = tf.gather(Gi, train.anime_id.map( #item latent factors 
							lambda id_: item_index[id_]).as_matrix())
		Gu_full = tf.gather(Gu, train.user_id.map( #user latent factors
							lambda id_: user_index[id_]).as_matrix()) 
		pred += tf.expand_dims(tf.einsum('ij,ji->i', # dot product
								Gi_full, tf.transpose(Gu_full)), 1)

		pred += tf.tile(alpha, (len(train), 1)) #overall bias 
		obj = tf.reduce_sum(abs(pred-y)) #L1 loss

		# regularization
		obj += lam * tf.reduce_sum(Bi**2)
		obj += lam * tf.reduce_sum(Bu**2) 
		obj += lam * tf.reduce_sum(Gi**2) 
		obj += lam * tf.reduce_sum(Gu**2)

		return obj

	def setup_tf_optimizers(self):
		"""Create the tf nodes needed to actually minimize objective func."""
		self.optimizer = tf.train.AdamOptimizer(0.01)
		self.obj = self.objective(self.alpha, self.Bi, self.Bu, 
								self.Gi, self.Gu, self.y, 1)
		self.trainer = self.optimizer.minimize(self.obj)

	#-------Wrapper for convenience------#

	def setup(self):
		"""Sets up the class so that it may run."""
		self.setup_user_index()
		self.setup_item_index()
		self.setup_tf_vars()
		self.setup_tf_optimizers()

	#-------Implement abstract methods------#

	def train_model(self):
		"""Trains model, if cache is not found."""
		# Check first if cached
		import os.path
		import pickle
		fname = './cache/lfm-%s.p' % self.username
		if os.path.isfile(fname):
			self.cAlpha, self.cBi, self.cBu, self.cGi, self.cGu \
			= pickle.load(open(fname, 'rb'))
			print('Using cached model.')
			return


		#Convert to local var for convenience
		alpha = self.alpha; Bi = self.Bi; Bu = self.Bu; 
		Gi = self.Gi; Gu = self.Gu; obj = self.obj

		#Set up TF session
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			tLoss = []
			for iteration in range(500):
				cvalues = sess.run([self.trainer, self.obj])
				print("objective = " + str(cvalues[1]))
				tLoss.append(cvalues[1])
	    

			self.tLoss = tLoss; 
			self.cAlpha, self.cBi, self.cBu, self.cGi, self.cGu, self.cLoss = \
						sess.run([alpha, Bi, Bu, Gi, Gu, obj])

			print("\nFinal train loss is ", self.cLoss)

			pickle.dump( [self.cAlpha, self.cBi, self.cBu, self.cGi, self.cGu], \
				open( fname, 'wb' ) )



	def predict(self, mask):
		ids = self.anime['anime_id'][mask]
		indices = ids.map(lambda x: self.item_index[x])

		bi = indices.map(lambda x: 0.0 if x == -1 else float(self.cBi[x])).as_matrix()
		gi = indices.map(lambda x: np.zeros(self.k) if x == -1 else self.cGi[x])
		gi = np.vstack(gi.as_matrix()).astype(np.float)

		user_index = self.user_index[self.uid]
		alpha = self.cAlpha; alpha = np.squeeze(np.tile(alpha, (1, len(bi))))
		bu = self.cBu[user_index]; bu = np.tile(bu, len(bi))
		gu = self.cGu[user_index]; 
		pred = alpha + bu + bi + np.squeeze(np.dot(gi, gu))

		sorted_indices = pred.argsort()[::-1]
		ids = ids.as_matrix()
		ranked_ids = ids[sorted_indices]
		return super().convert_ids_to_names(ranked_ids), pred[sorted_indices]

		
class LatentFactorWithFeatures(LatentFactorMethod):
	def __init__(self, username, k=3, min_num=10):
		super().__init__(username, k, min_num)

	def setup_feature_vector(self):
		genres = self.anime.genre.apply(lambda x: str(x).split(","))
		genres2 = genres.apply(pd.Series)
		all_genres = []
		for i in range(len(genres2.columns)):
		    genres2[i] = genres2[i].str.strip()
		    all_genres += map(lambda s: str(s).strip(), list(genres2[i].unique()))
		all_genres = list(np.unique(all_genres))

		genre_map = {}
		for i, x in enumerate(all_genres): genre_map.update({x:i})
		nGenres = len(all_genres)

		indexed = self.anime.set_index('anime_id')
		indexed.index.name = None
		ind = indexed.ix[self.train.anime_id.get_values()]

		train_genres = ind.genre.map(lambda x: [genre_map[j.strip()] for j in str(x).split(',')])
		ohe_genres = np.zeros((len(train_genres), nGenres))
		for i, row in enumerate(train_genres): ohe_genres[i, row] = 1

		self.ohe_genres = ohe_genres
		self.nGenres = nGenres
		self.all_genres = all_genres
		self.indexed = indexed
		self.genre_map = genre_map

	def setup_feature_vector_test(self, ids):
		test_ind = self.indexed.ix[ids]
		test_genres = test_ind.genre.map(lambda x: [self.genre_map[j.strip()] for j in str(x).split(',')])

		test_ohe_genres = np.zeros((len(test_genres), self.nGenres))
		for i, row in enumerate(test_genres): test_ohe_genres[i, row] = 1

		return test_ohe_genres




	def objective(self, alpha, Bi, Bu, Gi, Gu, Pi, y, lam):
		"""Definition of MAE objective function, with L2 normalization."""
		train = self.train
		item_index = self.item_index
		user_index = self.user_index
		ohe_genres = self.ohe_genres

		pred = tf.gather(Bi, train.anime_id.map( #item biases
							lambda id_: item_index[id_]).as_matrix()) 
		pred += tf.gather(Bu, train.user_id.map( #user biases
							lambda id_: user_index[id_]).as_matrix()) 

		Gi_full = tf.gather(Gi, train.anime_id.map( #item latent factors 
							lambda id_: item_index[id_]).as_matrix())
		Gu_full = tf.gather(Gu, train.user_id.map( #user latent factors
							lambda id_: user_index[id_]).as_matrix()) 
		Pi_full = tf.matmul(tf.constant(self.ohe_genres, dtype=tf.float32), Pi) 

		pred += tf.expand_dims(tf.einsum('ij,ji->i', (Gi_full+Pi_full), 
								tf.transpose(Gu_full)), 1)


		pred += tf.tile(alpha, (len(train), 1)) #overall bias 
		obj = tf.reduce_sum(abs(pred-y)) #L1 loss

		# regularization
		obj += lam * tf.reduce_sum(Bi**2)
		obj += lam * tf.reduce_sum(Bu**2) 
		obj += lam * tf.reduce_sum(Gi**2) 
		obj += lam * tf.reduce_sum(Gu**2)
		obj += lam * tf.reduce_sum(Pi**2)

		return obj


	def setup_tf_vars(self):
		"""Declares tensorflow variables."""
		super().setup_tf_vars()
		self.Pi = tf.Variable(tf.random_normal([self.nGenres, self.k], stddev=0.35))

		'''nUsers = len(self.train.user_id.unique())
		nAnime = len(self.train.anime_id.unique())

		self.alpha = tf.Variable(tf.constant([6.9], shape=[1, 1]))
		self.Bi = tf.Variable(tf.constant([0.0]*nAnime, shape=[nAnime, 1]))
		self.Bu = tf.Variable(tf.constant([0.0]*nUsers, shape=[nUsers, 1]))
		self.Gi = tf.Variable(tf.random_normal([nAnime, self.k], stddev=0.35))
		self.Gu = tf.Variable(tf.random_normal([nUsers, self.k], stddev=0.35))
		self.Pi = tf.Variable(tf.random_normal([self.nGenres, self.k], stddev=0.35))
		self.y = tf.cast(tf.constant(self.train['rating'].as_matrix(), 
									shape=[len(self.train),1]), tf.float32)'''

	def setup_tf_optimizers(self):
		"""Create the tf nodes needed to actually minimize objective func."""
		self.optimizer = tf.train.AdamOptimizer(0.01)
		self.obj = self.objective(self.alpha, self.Bi, self.Bu, 
								self.Gi, self.Gu, self.Pi, self.y, 1)
		self.trainer = self.optimizer.minimize(self.obj)


	def setup(self):
		"""Sets up the class so that it may run."""
		self.setup_feature_vector()
		super().setup_user_index()
		super().setup_item_index()
		self.setup_tf_vars()
		self.setup_tf_optimizers()

	def train_model(self):
		"""Trains model, if cache is not found."""
		# Check first if cached
		import os.path
		import pickle
		fname = './cache/lfmf-%s.p' % self.username
		if os.path.isfile(fname):
			self.cAlpha, self.cBi, self.cBu, self.cGi, self.cGu, self.cPi \
			= pickle.load(open(fname, 'rb'))
			print('Using cached model.')
			return


		#Convert to local var for convenience
		alpha = self.alpha; Bi = self.Bi; Bu = self.Bu; 
		Gi = self.Gi; Gu = self.Gu; Pi = self.Pi; obj = self.obj

		#Set up TF session
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			tLoss = []
			for iteration in range(500):
				cvalues = sess.run([self.trainer, self.obj])
				print("objective = " + str(cvalues[1]))
				tLoss.append(cvalues[1])
	    

			self.tLoss = tLoss; 
			self.cAlpha, self.cBi, self.cBu, self.cGi, self.cGu, self.cPi, self.cLoss = \
						sess.run([alpha, Bi, Bu, Gi, Gu, Pi, obj])

			print("\nFinal train loss is ", self.cLoss)

			pickle.dump( [self.cAlpha, self.cBi, self.cBu, self.cGi, self.cGu, self.cPi], \
				open( fname, 'wb' ) )


	def predict(self, mask):
		ids = self.anime['anime_id'][mask]
		indices = ids.map(lambda x: self.item_index[x])

		bi = indices.map(lambda x: 0.0 if x == -1 else float(self.cBi[x])).as_matrix()
		gi = indices.map(lambda x: np.zeros(self.k) if x == -1 else self.cGi[x])
		gi = np.vstack(gi.as_matrix()).astype(np.float)
		pi = np.dot(self.setup_feature_vector_test(ids), self.cPi)

		user_index = self.user_index[self.uid]
		alpha = self.cAlpha; alpha = np.squeeze(np.tile(alpha, (1, len(bi))))
		bu = self.cBu[user_index]; bu = np.tile(bu, len(bi))
		gu = self.cGu[user_index]; 
		pred = alpha + bu + bi + np.squeeze(np.dot((gi+pi), gu))

		sorted_indices = pred.argsort()[::-1]
		ids = ids.as_matrix()
		ranked_ids = ids[sorted_indices]
		return super().convert_ids_to_names(ranked_ids), pred[sorted_indices]



if __name__ == '__main__':
	lfm = LatentFactorWithFeatures('username', 3) #generic username for public ;-)
	lfm.setup()
	lfm.train_model()
	#lfm.predict_title('Gintama')
	recs, ratings = lfm.predict_unseen_shows()

	top = 0; bottom=5
	for rec, rating in zip(recs[top:bottom], ratings[top:bottom]): print (rec, rating)
	import pdb; pdb.set_trace()

