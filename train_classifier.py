import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
data_dict = pickle.load(open('./data_lettres.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Vérifier la forme des données
print("Vérification des formes des données :")
for i, item in enumerate(data):
    print(f"Élément {i} forme : {len(item)}")

# Trouver la longueur maximale
max_length = max(len(item) for item in data)

# Uniformiser la longueur des données en ajoutant des zéros (padding)
data_homogeneous = [item + [0] * (max_length - len(item)) for item in data]

# Convertir en tableau NumPy
data = np.array(data_homogeneous)
labels = np.asarray(labels)

# Vérifier la nouvelle forme des données
print("Nouvelle forme des données :", data.shape)

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialiser et entraîner le modèle
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Prédire les étiquettes pour l'ensemble de test
y_predict = model.predict(x_test)

# Calculer la précision
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Sauvegarder le modèle entraîné et max_length
with open('model_lettres.p', 'wb') as f:
    pickle.dump({'model': model, 'max_length': max_length}, f)
