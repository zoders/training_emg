from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
# x  = np.loadtxt("power_dataset_test/2/male22_1.txt")
#
# plt.plot(x)
# plt.figure(1)
# x = np.asarray(x, dtype=np.float64)
# x = x.reshape((1, x.shape[0]))
#
# print(x)
#
# model = load_model("output/m.model")
# lb = pickle.loads(open("output/lb.pickle", "rb").read())
# preds = model.predict(x)
#
# print(preds)
# i = preds.argmax(axis=1)[0]
# label = lb.classes_[i]
# print("{}: {:.2f}%".format(label, preds[0][i] * 100))
#
# plt.show()

dataset_path = "full_dataset_test"
data = []
labels = []
muap_paths = []
for folder in os.listdir(dataset_path):
    for muap_path in os.listdir("{}/{}".format(dataset_path, folder)):
        path = "{}\\{}\\{}".format(dataset_path, folder, muap_path)
        muap_paths.append(path)
not_nice = []
for p in muap_paths:
    x = np.loadtxt(p)
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape((1, x.shape[0]))
    model = load_model("output/m.model")
    lb = pickle.loads(open("output/lb.pickle", "rb").read())
    preds = model.predict(x)
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    if label != p.split(os.path.sep)[-2]:
        print("{} - {}: {:.2f}%".format(p, label, preds[0][i] * 100))
        not_nice.append("{} - {}: {:.2f}%".format(p, label, preds[0][i] * 100))
    else:
        print("nice!: " + p)
print(not_nice)
print(1 - (len(not_nice)/len(muap_paths)))
print()
