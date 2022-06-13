import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pathlib

dataset_path = os.path.dirname(__file__) + "/Dataset"
train_path = dataset_path + "/Train"
test_path = dataset_path + "/Test"
IMG_HEIGHT = 32
IMG_WIDTH = 32
CATEGORIES_COUNT = len(os.listdir(train_path))
print(CATEGORIES_COUNT)

img_dir = pathlib.Path(train_path)
plt.figure(figsize=(14,14))
index = 0
for i in range(CATEGORIES_COUNT):
    plt.subplot(7, 7, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    traffic_sign = list(img_dir.glob(f'{i}/*'))[0]
    image = tf.keras.preprocessing.image.load_img(traffic_sign, target_size=(IMG_WIDTH, IMG_HEIGHT))
    plt.imshow(image)
plt.show()
