import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, array_to_img, load_img
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential

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
    image = load_img(traffic_sign, target_size=(IMG_WIDTH, IMG_HEIGHT))
    plt.imshow(image)
plt.show()

def load_images(path):
    images_list = list()
    label_list = list()
    for category in range(CATEGORIES_COUNT):
        categories_path = os.path.join(path, str(category))
        for img in os.listdir(categories_path):
            img = load_img(os.path.join(categories_path, img), target_size=(IMG_WIDTH, IMG_HEIGHT))
            image = img_to_array(img)
            images_list.append(image)
            label_list.append(category)
    return images_list,label_list

images_loaded, labels_loaded = load_images(train_path)
labels_loaded = to_categorical(labels_loaded)
x_train, x_test, y_train, y_test = train_test_split(np.array(images_loaded), labels_loaded, test_size=0.4)




convModel = Sequential()


convModel.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
convModel.add(MaxPool2D(pool_size=(2, 2)))
convModel.add(Dropout(rate=0.25))


convModel.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
convModel.add(MaxPool2D(pool_size=(2, 2)))
convModel.add(Dropout(rate=0.25))

convModel.add(Conv2D(filters=64, kernel_size=3, activation='relu'))

convModel.summary()

convModel.add(Flatten())
convModel.add(Dense(units=64, activation='relu'))
convModel.add(Dense(CATEGORIES_COUNT, activation='softmax'))

convModel.summary()

print("Kreiran model")
convModel.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Kompajliran model")
EPOCHS = 30
fitted_model = convModel.fit(x_train, 
                    y_train,
                    validation_data = (x_test, y_test), 
                    epochs=EPOCHS, 
                    steps_per_epoch=60
                   )

print("Zavrsio model")

