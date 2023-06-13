from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import nltk
from nltk.corpus import wordnet
import gensim.downloader
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import cv2
import os
import tqdm
#1. Query should be adj. + noun(ex. curly hair boy,
#blonde lady, tall gentleman, freckles girl…)
#2. Collect images from services like google
#image search, instagram, pinterest, imgur…
#3. Save final images classified by query

#steps
#1. gensim to get search query
#1a. confirm valid search query validation
#2. use beautiful soup(web scraper) to search valid query into an image website
#2a. identify faces using dlib and landmark. If face exist save if not, delete
#2b. create new directory if search is new if not, add to existing directory.(try to narrow down to adjective)

#1
#gensim
#download wordnet corpus
nltk.download('wordnet')
# def is_adjective(word):
#   synsets = wordnet.synsets(word)
#   for synset in synsets:
#     if synset.pos() == 'a':
#       return True
#   return False

# adj = input("enter an adjective: ")
# if is_adjective(adj):
#   print(f"{adj} is an adjective.")
# else:
#   if print(f"{adj} is not an adjective"):
#     print("Please enter an adjective")
#     exit()
word = input("enter a search: ")
query = word.split()
print(query)
# model = gensim.downloader.load('glove-twitter-200')

# target_appearance = [query[0]]
# similar_words = model.most_similar(positive=target_appearance, topn=100)

# for word, similarity in similar_words:
# #   print(word, is_adjective(word), similarity)
#     print(word, similarity)


#2
#selenium
# import and mix dlib face identifier to this.
driver = webdriver.Chrome()

search_keyword = word
search_url = f'https://www.pinterest.com/search/pins/?q={search_keyword}'
driver.get(search_url)
time.sleep(5)

soup = BeautifulSoup(driver.page_source, 'html.parser')
image_tags = soup.find_all('img', {'src': True})

if not os.path.exists('pinterest_images'):
  os.makedirs('pinterest_images')

for idx, img in enumerate(image_tags):
  image_url = img['src']
  image_url = image_url.replace("236x", "736x", 1)
  image_name = f"{search_keyword}_{idx}.jpg"
  image_path = os.path.join('pinterest_images', image_name)

  response = requests.get(image_url)
  with open(image_path, 'wb') as f:
    f.write(response.content)

driver.quit()