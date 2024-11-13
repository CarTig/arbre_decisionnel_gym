import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree

#*************AFFICHAGE ET INFORMATIONS GENERALES DE LA DATA**************************
df = pd.read_csv('gym_members_exercise_tracking.csv')
print(df.head(10))
print(df.shape)
print(df.describe())
print(df.isnull().sum()) #aucune donnée null
print(df.dtypes)

#************NETTOYAGE DES DONNEES****************
#enlever tous les NaN
df.dropna(inplace=True)

#*****************PRE TRAITEMENT DES DONNEES**********************************
#remplacement des données Gender de objet string a numérique
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})


#======================CLASSIFICATION ARBRE DE DECISION=======================
#choix de la colonne à prédire/ de l'information : on veut prédire le type de workout en fonction de différentes caractéristiques

# Encodage de la variable cible "Workout_Type" en numérique grace a .cat.codes
df["Workout_Type"] = df["Workout_Type"].astype("category").cat.codes

# Définir les caractéristiques (features) et la variable cible (target)
X = df.drop(columns=["Workout_Type"])  # features: toutes les colonnes sauf Workout_Type
y = df["Workout_Type"]  # cible : Workout_Type

# Diviser le dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#************ CRÉATION ET ENTRAÎNEMENT DE L'ARBRE DE DÉCISION ************
# Créer le modèle
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)  # max_depth limite la profondeur de l'arbre
clf.fit(X_train, y_train)

#************ VISUALISATION DE L'ARBRE DE DÉCISION ************
plt.figure(figsize=(30,20))
plot_tree(clf, feature_names=X.columns, class_names=["Yoga", "HIIT", "Cardio", "Strength"], filled=True)
plt.title("Arbre de décision pour la prédiction de Workout_Type")
plt.show()



#========================PREDICTION NIVEAU D'EXPERIENCE================================
# Encodage de la variable cible "Experience_level" en numérique grace a .cat.codes
df["Experience_Level"] = df["Experience_Level"].astype("category").cat.codes

# Définir les caractéristiques (features) et la variable cible (target)
X = df.drop(columns=["Experience_Level"])  # features: toutes les colonnes sauf Experience_Level
y = df["Experience_Level"]  # cible : Experience_Level

clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)

#visualisation de l'arbre de décision avec matplotlib
plt.figure(figsize=(30,20))
plot_tree(clf, feature_names=X.columns, class_names=["1","2","3"], filled=True)
plt.title("Arbre de décision pour la prédiction de Experience_Level")
plt.show()