from data_loader import load_normalized_data
from knn_from_scratch import KNN
from sklearn.model_selection import train_test_split

# 1. Chargement des données
X, Y, scaler = load_normalized_data(file_path="bienetre.csv")

# 2. Conversion en listes (full vanilla)
X = X.tolist()
Y = Y.tolist()

# 3. Split stratifié train/test (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# 4. modèle et entraînement
model = KNN()
model.fit(x_train, y_train)

# 5. Grid search pour trouver le meilleur k
results = model.grid_search(
    setting_model="n_neighbors",
    setting_values=[1, 3, 5, 7, 9],
    x_test=x_test,
    y_test=y_test
)
print("=== Grid Search ===")
print(results)
