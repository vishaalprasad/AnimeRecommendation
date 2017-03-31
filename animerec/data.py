import pandas as pd
import numpy as np

def get_data(clean = True, minNumRatings = 10):
	""" Load in data, and then optionally clean it.

	Parameters:
	-----------
	clean : boolean (optional)
		whether to clean the data or not.
	minNumRatings : int (optional)
		minimum number of anime a user must have rated to not be removed.

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
	bad_ids = anime.anime_id[(anime.type != 'TV') | (anime.rating.isnull())]
	users.drop(users[users.anime_id.map(lambda x: x in bad_ids)].index, inplace=True)
	anime.drop(anime[(anime.type != 'TV') | (anime.rating.isnull())].index, inplace=True)

	#Remove users with fewer than minNumRatings views
	vc = users.user_id.value_counts()
	low_ratings = vc[vc.map(lambda x: x < minNumRatings)].index
	users.drop(users[users.user_id.map(lambda x: x in low_ratings)].index, inplace=True)

	return users, anime



