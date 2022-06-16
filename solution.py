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
from keras.callbacks import History
from sklearn.metrics import accuracy_score
import pandas as pd





dataset_path = os.path.dirname(__file__) + "/Dataset"
train_path = dataset_path + "/Train"
test_path = dataset_path + "/"
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




conv_model = Sequential()


conv_model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
conv_model.add(MaxPool2D(pool_size=(2, 2)))
conv_model.add(Dropout(rate=0.25))


conv_model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
conv_model.add(MaxPool2D(pool_size=(2, 2)))
conv_model.add(Dropout(rate=0.25))

conv_model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))

conv_model.summary()

conv_model.add(Flatten())
conv_model.add(Dense(units=64, activation='relu'))
conv_model.add(Dense(CATEGORIES_COUNT, activation='softmax'))

conv_model.summary()

print("Kreiran model")
conv_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Kompajliran model")
EPOCHS = 30
fitted_model = conv_model.fit(x_train, 
                    y_train,
                    validation_data = (x_test, y_test), 
                    epochs=EPOCHS, 
                    steps_per_epoch=60
                   )

print("Zavrsio model")


model_loss, model_accuracy = conv_model.evaluate(x_test, y_test)

print('Preciznost na dataset-u za testiranje: ', model_accuracy * 100)

accuracy = fitted_model.history['accuracy']
val_accuracy = fitted_model.history['val_accuracy']

loss=fitted_model.history['loss']
val_loss=fitted_model.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Preciznost pri treniranju')
plt.plot(epochs_range, val_accuracy, label='Preciznost pri validaciji')
plt.legend(loc='lower right')
plt.title('Preciznost pri treniranju i validaciji')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Gubici pri treniranju')
plt.plot(epochs_range, val_loss, label='Gubici pri validaciji')
plt.legend(loc='upper right')
plt.title('Gubici pri treniranju i validaciji')
plt.show()



Y_test = pd.read_csv(test_path + 'Test.csv')
test_labels = Y_test["ClassId"].values
test_images = Y_test["Path"].values

output_images = list()
for img in test_images:
    image = load_img(os.path.join(test_path, img), target_size=(32, 32))
    output_images.append(np.array(image))

X_test=np.array(output_images)
predictions = np.array(np.argmax(conv_model.predict(X_test), axis=-1))


print('Preciznost predvidjenih vrednosti: ',accuracy_score(test_labels, predictions)*100)

plt.figure(figsize = (13, 13))

start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = predictions[start_index + i]
    actual = test_labels[start_index + i]
    col = 'g'
    if prediction != actual:
        col = 'r'
    plt.xlabel('Stvarna vr.={} || Predvidjena vr={}'.format(actual, prediction), color = col)
    plt.imshow(X_test[start_index + i])
    
plt.show()
