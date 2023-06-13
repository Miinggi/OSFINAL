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
import dlib
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



nltk.download('wordnet')

#noun detector
def is_noun(word):
    synsets = wordnet.synsets(word)
    for synset in synsets:
        if synset.pos() == 'n':
            return True
    return False

#adj detector
def is_adjective(word):
    synsets = wordnet.synsets(word)
    for synset in synsets:
        if synset.pos() == 'a':
            return True
    return False

#search query
adj = input("enter an adjective: ")
noun = input("enter a noun: ")
target_appearance = [adj, noun]

query = (f'{adj} {noun}')
print(query)
#------------------gensim------------------

#image collector
def image_collection(search_keyword):
  driver = webdriver.Chrome()
  search_url = f'https://www.pinterest.com/search/pins/?q={search_keyword}'
  driver.get(search_url)
  time.sleep(5)

  soup = BeautifulSoup(driver.page_source, 'html.parser')
  image_tags = soup.find_all('img', {'src': True})

  if not os.path.exists(f'pinterest_images {search_keyword}'):
      os.makedirs(f'pinterest_images {search_keyword}')

  for idx, img in enumerate(image_tags):
      image_url = img['src']
      image_url = image_url.replace("236x", "736x", 1)
      image_name = f"{search_keyword}_{idx}.jpg"
      image_path = os.path.join(f'pinterest_images {search_keyword}', image_name)
      print(image_path)

      response = requests.get(image_url)
      with open(image_path, 'wb') as f:
          f.write(response.content)

  driver.quit()

#------------------dlib------------------

# drawing landmarks function definition
def draw_landmark(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    image = cv2.imread(image_path)
    
    # for face_rect in detector(image, 0):
    #     shape = predictor(image, face_rect)
    #     for x in range(68):
    #         pts = (shape.part(x).x, shape.part(x).y)
    #         cv2.circle(image, pts, 1, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)
    #         cv2.putText(image, f"{x}", pts, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("landmark", image)
    cv2.waitKey()



#alpha blending def
def alpha_blending():
    blendImg = cv2.imread('trial.png', cv2.IMREAD_UNCHANGED)
    alpha = blendImg[:, :, 3]
    blendImg_bgr = blendImg[:, :, :3]

    baseImg = cv2.imread('cat.jpg')

    h, w, _ = blendImg.shape
    pos_y = baseImg.shape[0] - h - 10
    pos_x = 10

    alpha = np.stack([alpha, alpha, alpha], axis=2)/255

    output = np.array(baseImg)
    bg = output[pos_y: pos_y + h, pos_x: pos_x + w]
    output[pos_y:pos_y + h, pos_x: pos_x + w] = (blendImg_bgr * alpha) + bg * (1 - alpha)

    cv2.imshow('blendImg', blendImg)
    cv2.imshow('alpha', alpha)
    cv2.imshow('origin', baseImg)
    cv2.imshow('output', output)
    cv2.waitKey()

imgs = image_collection(query)

dname = f'pinterest_images {query}'
print(dname)
fnames = os.listdir(dname)
print(fnames)

for fname in fnames:
    image_path = os.path.join(dname, fname)
    print(image_path)
    is_face_present(image_path)
