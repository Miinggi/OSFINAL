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

nltk.download('wordnet')
word = input("enter a search: ")
query = word.split()
print(query)
model = gensim.downloader.load('glove-twitter-200')

target_appearance = [query]
similar_words = model.most_similar(positive=target_appearance, topn=100)

for word, similarity in similar_words:
    print(word, similarity)