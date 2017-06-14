import pandas as pd
import numpy as np

def get_data(clean = True):
	""" Load in data, and then optionally clean it.

	Parameters:
	-----------
	clean : boolean (optional)
		whether to clean the data or not.

	Returns:
	--------
	users, anime : (Pandas dataframe, Pandas dataframe) 
		users: the dataframe corresponding to user/anime pairs (and the rating given)
		anime: the dataframe corresponding to a given anime and all its associated 
		information. 
	"""

	users = pd.read_csv('rating.csv')
	anime = pd.read_csv('anime.csv')
	if not clean:
		return users, anime

	#Remove all "-1" ratings
	users.drop(users[users.rating == -1].index, inplace=True)

	#Remove non-TV anime, as well as anime with a rating of NaN
	#bad_ids = anime[(anime.type!='TV') | (anime.rating.isnull())].anime_id.values
	bad_ids = anime[(anime.type!='TV')].anime_id.values
	users.drop(users[users.anime_id.map(lambda x: x in bad_ids)].index, inplace=True)
	anime.drop(anime[(anime.type != 'TV') | (anime.rating.isnull())].index, inplace=True)
	return users, anime

def remove_users(users, minNumRatings = 10):

	""" Remove users with fewer ratings than minNumRatings.

	Parameters:
	-----------
	users : Pandas DataFrame
		the dataframe corresponding to user/anime pairs (and the rating given)
	minNumRatings : int
		minimum number of anime a user must have rated to not be removed.

	Returns:
	--------
	users : Pandas dataframe 
		the dataframe corresponding to user/anime pairs (and the rating given)
	"""

	#Remove users with fewer than minNumRatings views
	vc = users.user_id.value_counts()
	low_ratings = vc[vc.map(lambda x: x < minNumRatings)].index
	users.drop(users[users.user_id.map(lambda x: x in low_ratings)].index, inplace=True)

	return users


def create_anime_image_data(anime):

	"""Create (or load) a dict for each anime that has a high level CNN 
	representation of the associated MAL image.

	Parameters:
	-----------
	anime : Pandas dataframe
		the dataframe corresponding to the list of all anime in the dataset.

	Returns:
	--------
	image_data : dict 
		A dict where each title is a key and the CNN representation of its MAL 
		image is the value.
	"""

	from PIL import Image
	from lxml import etree

	import sys
	import os.path
	import pickle
	import time
	import urllib.request
	import io
	import requests


	dir_path = os.path.dirname(os.path.realpath(__file__))
	fname = dir_path + '/../data/image_data.p'

	if os.path.isfile(fname):
		print('Using cached image data.')
		return pickle.load(open(fname, 'rb'))
	
	# To import mynet from a directory below, I must add that directory to path
	sys.path.insert(0, dir_path + '/../')

	import tensorflow as tf
	from mynet import CaffeNet

	#MAL credentials
	username = 'username'; password = 'password'


	#Get the tensorflow model started
	images = tf.placeholder(tf.float32, [None, 224, 224, 3])
	net = CaffeNet({'data':images})
	sesh = tf.Session()
	sesh.run(tf.global_variables_initializer())
	# Load the data
	net.load('mynet.npy', sesh)

	image_data = {}


	width, height = (225, 350) #all MAL images are this size
	new_width, new_height = (224, 224)
	left = int((width - new_width)/2)
	top = int((height - new_height)/2)
	right = (left+new_width)
	bottom = (top + new_height)


	# Now to actually construct the dataset 
	for name in anime.name:

		#First, get the full anime XML from MAL's search query
		title = "+".join(name.split() )
		query = 'https://%s:%s@myanimelist.net/api/anime/search.xml?q=%s'  \
			% (username, password, title)
		r = requests.get(query)

		#Make sure that the request goes through
		while r.status_code != requests.codes.ok:
			r = requests.get(query)
			time.sleep(1.0)    # don't overload their server...

		#From the XML file, pull all images that fit the query
		doc = etree.fromstring(r.content)
		image = doc.xpath('.//image/text()')
		''' For sake of simplicity, I assume that the first image,
			corresponding to the first matching response to the query, is what 
			we want. This isn't strictly correct, but for my goals here it's 
			good enough.'''
		
		URL = image[0] 

		with urllib.request.urlopen(URL) as url:
			f = io.BytesIO(url.read())
		img = Image.open(f, 'r')

		#Center crop image so it's 225x225x3, and convert to numpy.
		img = np.array(img.crop((left, top, right, bottom)))

		#Now use the Illustration2Vec pre-trained model to extract features.

		output = sesh.run(net.get_output(), feed_dict={images: img[None,:]})

		image_data[name] = output

		print('Finished with ' + anime.name)


	pickle.dump(image_data, open(fname, 'wb'))
	sesh.close()


if __name__ == '__main__':
	users, anime = get_data()
	create_anime_image_data(anime)



