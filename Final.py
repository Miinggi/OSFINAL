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
    faces = list(detector(image, 0))

    if len(faces) == 0:
        os.remove(image_path)
        print(f'Deleted {image_path} because no face was detected')
        return None
    
    for face_rect in faces:
        shape = predictor(image, face_rect)
        for x in range(68):
            pts = (shape.part(x).x, shape.part(x).y)
            cv2.circle(image, pts, 1, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)
            cv2.putText(image, f"{x}", pts, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("landmark", image)
    # cv2.waitKey()
    
    return image



def alpha_blending(landmarked_image, watermark, output_filename):
    blendImg = cv2.imread(watermark, cv2.IMREAD_UNCHANGED)
    alpha = blendImg[:, :, 3]
    blendImg_bgr = blendImg[:, :, :3]

    h, w, _ = blendImg.shape
    pos_y = landmarked_image.shape[0] - h - 10
    pos_x = 10

    # Check if the blend image exceeds the boundaries of the base image
    if pos_y < 0 or pos_x < 0 or pos_y + h > landmarked_image.shape[0] or pos_x + w > landmarked_image.shape[1]:
        raise ValueError("Blend image exceeds the boundaries of the base image.")

    # Resize the blend image to match the region of interest in the base image
    blendImg_bgr = cv2.resize(blendImg_bgr, (w, h))

    alpha = np.stack([alpha, alpha, alpha], axis=2) / 255

    output = np.array(landmarked_image)
    bg = output[pos_y: pos_y + h, pos_x: pos_x + w]
    output[pos_y:pos_y + h, pos_x: pos_x + w] = (blendImg_bgr * alpha) + bg * (1 - alpha)

    # Save the output image after alpha blending with a unique filename
    output_path = os.path.join(os.path.dirname(image_path), output_filename)
    cv2.imwrite(output_path, output)

    cv2.imshow('output', output)
    # cv2.waitKey()








from nltk.corpus import wordnet



if __name__ == "__main__":
    counter = 0  # Counter to keep track of the number of alpha-blended images
    
    while counter < 100:
        # Search query
            adj = input("enter an adjective: ")
            noun = input("enter a noun: ")
            # target_appearance = [adj, noun]

            query = (f'{adj} {noun}')
            print(query)

            imgs = image_collection(query)

            dname = f'pinterest_images {query}'
            fnames = os.listdir(dname)

            watermark = 'pepesmile.png'

            for fname in fnames:
                image_path = os.path.join(dname, fname)
                landmarked_image = draw_landmark(image_path)
                if landmarked_image is not None:
                    output_filename = f"alpha_blended_{fname}"
                    alpha_blending(landmarked_image, watermark, output_filename)
                    os.remove(image_path)  # Remove the original image

                    counter += 1  # Increment the counter
                    print(counter)

                    if counter == 100:
                        break  # Stop further processing once 100 alpha-blended images are generated

            updated_fnames = os.listdir(dname)
            print(updated_fnames)
else:
            print("Invalid adjective or noun.")


