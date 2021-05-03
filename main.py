import matplotlib
#matplotlib.use("Agg")

# подключаем необходимые пакеты
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Reshape
from tensorflow.keras.optimizers import SGD
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for training on gpu
print("[INFO] loading data...")
dataset_path = "datasets/full_dataset"
data = []
labels = []
muap_paths = []
for folder in os.listdir(dataset_path):
    for muap_path in os.listdir("{}/{}".format(dataset_path, folder)):
        path = "{}\\{}\\{}".format(dataset_path, folder, muap_path)
        muap_paths.append(path)
print("[INFO] data loaded")
random.seed(42)
random.shuffle(muap_paths)
print("[INFO] creating data list and labels list...")
for muap_path in muap_paths:
    muap = np.loadtxt(muap_path)
    data.append(muap)
    label = muap_path.split(os.path.sep)[-2]
    labels.append(label)
print("[INFO] lists created")
data = np.asarray(data, dtype=np.float64)
labels = np.asarray(labels)
print("[INFO] splitting to train and test data...")
# разбиваем данные на обучающую и тестовую выборки, используя 75%
# данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
print("[INFO] data splitted")
# trainY = to_categorical(trainY)
# testY = to_categorical(testY)


# конвертируем метки из целых чисел в векторы (для 2х классов при
# бинарной классификации вам следует использовать функцию Keras
# “to_categorical” вместо “LabelBinarizer” из scikit-learn, которая
# не возвращает вектор)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

num_classes = 4
num_sensors = 1

input_size = trainX.shape[1]


model = Sequential()
model.add(Reshape((input_size, num_sensors), input_shape=(input_size,)))
model.add(Conv1D(50, 10, activation='relu', input_shape=(input_size, num_sensors)))
model.add(Conv1D(25, 10, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Conv1D(100, 10, activation='relu'))
model.add(Conv1D(50, 10, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Dropout(0.5))
# next layers will be retrained
model.add(Conv1D(100, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(num_classes, activation='softmax'))

EPOCHS = 100

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#start_time = time.time()

# обучаем нейросеть
# To make sure that you have "at least steps_per_epoch * epochs batches", set the steps_per_epoch to
# steps_per_epoch = len(X_train)//batch_size
# validation_steps = len(X_test)//batch_size # if you have test data
H = model.fit(trainX, trainY,
                    epochs=EPOCHS,
                    batch_size=32,
                    steps_per_epoch=len(trainX)//32,
                    validation_data=(testX, testY),
                    validation_steps=len(testX)//32
                    )

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
H.history["loss"][0] = 1
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_accuracy")
plt.plot(N, H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("output/plot.png")

# сохраняем модель и бинаризатор меток на диск
print("[INFO] serializing network and label binarizer...")
model.save("output/m.model")
f = open("output/lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()
