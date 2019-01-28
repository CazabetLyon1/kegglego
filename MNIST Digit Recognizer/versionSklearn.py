# Script basé sur : https://www.youtube.com/watch?v=aZsZrkIgan0
# Reste à faire : Sauver/Charger l'arbre de decision pour eviter de le retrain à chaque fois

import random as rd # Lib pour le rand
import numpy as np # Lib de math
import matplotlib.pyplot as pt # Lib pour afficher (plot)
import pandas as pd # Lib pour lire les CSV
from sklearn.tree import DecisionTreeClassifier # Lib avec les arbres de decision

# On ouvre le fichier et on le découpe en train/test
data = pd.read_csv("dataset/train.csv").values

featureTrain = data[0:40000, 1:]
labelTrain = data[0:40000, 0]

featureTest = data[40000:, 1:]
labelTest = data[40000:, 0]

# On déclare l'arbre et on l'entraine avec le jeu de train
tree = DecisionTreeClassifier()

tree.fit(featureTrain, labelTrain)

# On effectue des prédictions sur le jeu de test
result = tree.predict(featureTest)

# On calcule l'accuracy de nos test
count = 0
for i in range(0, 2000):
	count += 1 if result[i] == labelTest[i] else 0
print("Accuracy on 2000 sized test sample = ", (count/2000)*100, '\n')

while input("\033[mDo some prédiction ? (y/n) : ") == 'y':
	rng = rd.randint(0, 2000)
	image = featureTest[rng]
	image.shape = (28, 28)
	pt.imshow(255 - image, cmap = 'gray')
	if tree.predict([featureTest[rng]]) == [labelTest[rng]]:
		print("\n\033[1;32mPredicted : ", tree.predict([featureTest[rng]]), " | Was : ", [labelTest[rng]])
	else:
		print("\n\033[1;31mPredicted : ", tree.predict([featureTest[rng]]), " | Was : ", [labelTest[rng]])
	pt.show()
