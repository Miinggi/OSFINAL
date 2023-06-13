import gensim.downloader
import time
import requests
import os
import cv2
import numpy as np
import pandas as pd
import dlib
import nltk
from nltk.corpus import wordnet
from selenium import webdriver
from bs4 import BeautifulSoup


#downloads for search query generation
nltk.download('wordnet')
model = gensim.downloader.load('glove-twitter-200')

#webdriver
driver = webdriver.Chrome()

#adjective determiner
def is_adjective(word):
    synsets = wordnet.synsets(word)
    for synset in synsets:
        if synset.pos() == 'a':
            return True
    return False

#noun determiner
def is_noun(word):
    synsets = wordnet.synsets(word)
    for synset in synsets:
        if synset.pos() == 'n':
            return True
    return False

#search query generation
adjective = "beautiful"
noun = "girl"
target_appearance = [adjective, noun]
similar_words = model.most_similar(positive=target_appearance, topn=100)

for word, similarity in similar_words:
    print(word, is_adjective(word), similarity)

#webdriver
driver = webdriver.Chrome('chromedriver/chromedriver')

#store generated search queries
#pandas stuff

#image collection part
def image_collection(search_keyword):
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

#drawing landmarks function definition
def draw_landmark(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    image = cv2.imread(image_path)
    for face_rect in detector(image, 0):
        shape = predictor(image, face_rect)

        for x in range(68):
            pts = (shape.part(x).x, shape.part(x).y)
            cv2.circle(image, pts, 1, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)
            cv2.putText(image, f"{x}", pts, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
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