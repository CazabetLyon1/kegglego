# MNIST Digit
Lien compétition : https://www.kaggle.com/c/digit-recognizer
Nous avons au début de notre projet réalisé de simples solutions pour un digit recognizer.
C'était plus dans le but de se familiariser au langage / différentes librairies.


# AI Open Images
Lien compétition : https://www.kaggle.com/c/google-ai-open-images-object-detection-track
Il s'agit-là de la compétition principale auquel nous avons participé.
*Voici un exemple de résultat obtenu avec notre algorithme :*
![resultat](https://image.noelshack.com/fichiers/2019/16/7/1555856346-resultat.png)

Pour cette compétition nous proposons 2 versions.
## Train
Lien kernel : https://www.kaggle.com/lefuret/train-ai-open-images-object-detection-track
Cette version comprend l'entraînement des réseaux, ils faut compter environ 30 minutes pour l'exécution complète.

Nous proposons en plus de notre kernel une version source **script** ou **notebook**
Toutefois il faut exécuter la version dans un kernel Kaggle ayant ces différents **dataset** :
  - https://www.kaggle.com/c/google-ai-open-images-object-detection-track
  - https://www.kaggle.com/bigquery/open-images

La connexion internet du kernel doit être activée tout comme l'utilisation de GPU.


## Demo
Lien kernel: https://www.kaggle.com/lefuret/demo-ai-open-images-object-detection-track
Cette version comprend uniquement la démo, les réseaux sont initialisés via des fichiers contenant des réseaux préalablement entrainés, il faut compter environ 1 minute pour l'exécution complète, puis moins de 3 secondes par images.

Nous proposons en plus de notre kernel une version source **script** ou **notebook**
Toutefois il faut exécuter la version dans un kernel Kaggle ayant ces différents **dataset** :
  - https://www.kaggle.com/lefuret/rpn-models

La connexion internet du kernel doit être activée tout comme l'utilisation de GPU.

Il serait possible de gérer les dépendances via des *pip install* mais ayant utilisé kaggle nous ne connaissons pas les dépendances exactes. Dans ce cas les fichiers du dataset doivent se trouver dans le dossier **../input**


