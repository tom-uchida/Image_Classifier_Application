from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# Information of API key
key = "00b33a732084b1a48f55ed3296205a2c"
secret = "7ca7c98b12c7e7e0"
wait_time = 1

# Directory to save images downloaded from flickr
search_name = sys.argv[1]
save_dir = "../images/" + search_name
if not os.path.exists(save_dir): os.mkdir(save_dir)

# Setting of FlickrAPI
flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = search_name,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

photos = result['photos']
# pprint(photos)

# Download images from flickr
num_of_imgs = 0
for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    file_path = save_dir + '/' +  photo['id'] + '.jpg'

    # skip
    if os.path.exists(file_path): continue

    # do
    urlretrieve(url_q, file_path)

    # show progress
    if i % 100 == 0:
        num_of_imgs = num_of_imgs + 100
        print(num_of_imgs, "images done.")

    time.sleep(wait_time)