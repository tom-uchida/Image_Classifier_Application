from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# Information of API key
key = "00b33a732084b1a48f55ed3296205a2c"
secret = "7ca7c98b12c7e7e0"
wait_time = 1

# Directory to save images downloaded from flickr
data_name = sys.argv[1]
save_dir = "../images/" + data_name

# Setting of FlickrAPI
flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = data_name,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

photos =result['photos']
pprint(photos)