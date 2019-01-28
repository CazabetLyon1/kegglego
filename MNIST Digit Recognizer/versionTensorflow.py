# Inspiré de : https://www.tensorflow.org/tutorials/keras/basic_classification
# Amélioration possible : Sauver/Charger le réseau de neurone pour eviter de le retrain à chaque fois

import random as rd # Lib pour le rand
import numpy as np # Lib de math
import matplotlib.pyplot as pt # Lib pour afficher (plot)
import pandas as pd # Lib pour lire les CSV
import tensorflow as tf # Lib pour les réseaux de neurones
from tensorflow import keras

# On ouvre le fichier et on le découpe en train/test
data = pd.read_csv("dataset/train.csv").values

featureTrain = data[0:40000, 1:]
featureTrain = featureTrain/255
labelTrain = data[0:40000, 0]

featureTest = data[40000:, 1:]
featureTest = featureTest/255
labelTest = data[40000:, 0]

# On déclare le réseau de neurone
net = keras.Sequential([
	keras.layers.Dense(784),
	keras.layers.Dense(128, activation = tf.nn.sigmoid),
	keras.layers.Dense(10, activation = tf.nn.softmax)
])

net.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# On entraine le réseau de neurone 5 fois d'affilé avec le jeu de train
net.fit(featureTrain, labelTrain, epochs = 5)

# On effectue des prédictions sur le jeu de test
result = net.predict(featureTest)

# On calcule l'accuracy de nos prédictions
count = 0
for i in range(0, 2000):
	count += 1 if np.argmax(result[i]) == labelTest[i] else 0
print("\n\033[1mAccuracy on 2000 sized test sample = ", count/2000, '\n')

# Permet l'affichage de prédiction random (le système de while est moche mais permet de bien gérer les inputs clavier)
char = ''
while True:
	while True:	
		char = input("\033[mDo some prédiction ? (y/n) : ")
		if char == 'y' or char == 'n':
			break
	if char == 'n':
		break
	rng = rd.randint(0, 2000)
	image = featureTest[rng]
	image.shape = (28, 28)
	pt.imshow(255 - image, cmap = 'gray')
	if np.argmax(result[rng]) == labelTest[rng]:
		print("\n\033[1;32mPredicted : ", np.argmax(result[rng]), " | Was : ", labelTest[rng])
	else:
		print("\n\033[1;31mPredicted : ", np.argmax(result[rng]), " | Was : ", labelTest[rng])
	pt.show()
