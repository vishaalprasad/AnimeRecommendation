def get_user_info(username):

	""" Retrieves information via MyAnimeList's API about a user's viewed anime and the
	corresponding ratings. It only takes into account non-zero ratings.

	Parameters:
	-----------
	username : string
		Username of the MyAnimeList user whose data to pull.

	Returns:
	--------
	seen_titles, seen_ratings : (List of strings, List of ints) 
		seen_id: a list with anime_id which the user has seen
		seen_ratings: a list with the ratings for each corresponding title.
	"""

	#First, get XML data based on username
	import requests
	query = 'https://myanimelist.net/malappinfo.php?u=%s&status=all&type=anime' % username
	r = requests.get(query)
	if r.status_code != requests.codes.ok:
		print ("Error processing request. Try again")
		import sys; sys.exit()

	#Now, parse XML data
	from lxml import etree
	doc = etree.fromstring(r.content)
	ids = doc.xpath('.//series_animedb_id/text()')
	ratings = doc.xpath('.//my_score/text()')

	#Now take the data and construct a rating for them.
	from itertools import compress
	mask = [rating != '0' for rating in ratings]
	seen_id = list(map(int, compress(ids, mask)))
	seen_ratings = list(map(int, compress(ratings, mask)))

	return seen_id, seen_ratings
