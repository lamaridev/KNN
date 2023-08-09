import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Charger les données
dataset = pd.read_csv("C:/Users/hp/anaconda3/user+data.csv")

x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values



# Diviser les données en ensembles d'entraînement et de test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

knn_minkowski.fit(x_train_scaled, y_train)


y_pred_minkowski = knn_minkowski.predict(x_test_scaled)


confusion_mat = confusion_matrix(y_test, y_pred_minkowski)

print("Matrice de confusion :")
print(confusion_mat)


# Exemple à tester
new_example = np.array([[35, 57100]])

# Mise à l'échelle des caractéristiques de l'exemple
new_example_scaled = scaler.transform(new_example)

# Utiliser le modèle pour prédire la classe de l'exemple
predicted_class = knn_minkowski.predict(new_example_scaled)

print(predicted_class)