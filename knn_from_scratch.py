class KNN(object):

	def __init__(self, n_neighbors=None):
		self.n_neighbors = n_neighbors
		self.x_train = None
		self.y_train = None
		
	def fit (self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x):
		distances = []
		for x_train_point in self.x_train:
			distance = 0

			for index in range(len(x)):
				distance = distance + (x[index] - x_train_point[index]) ** 2
			distance = distance ** 0.5
			distances.append(distance)

		sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
		n_nearest_neighbors = sorted_distances[:self.n_neighbors]

		y_label_nearest_neighbors = []
		for neighbor in n_nearest_neighbors:
			y_label_nearest_neighbors.append(self.y_train[neighbor[0]])

		target_distribution = {} 
		for y_label_nearest in y_label_nearest_neighbors:
			if y_label_nearest in target_distribution:
				target_distribution[y_label_nearest] = target_distribution[y_label_nearest] + 1
			else:
				target_distribution[y_label_nearest] = 1

		return max(target_distribution, key=lambda x: target_distribution[x])


	def evaluate(self, x_test, y_test):
		labels_target = list(set(y_test))
		y_pred = []
		for index in range(len(x_test)):
			y_pred.append(self.predict(x_test[index]))

		f1_scores = []
		for label in labels_target:
			TP = 0
			FP = 0
			FN = 0
			for index in range(len(y_pred)):
				if y_pred[index] == label and y_test[index] == label:
					TP = TP + 1
				elif y_pred[index] == label and y_test[index] != label:
					FP = FP + 1
				elif y_pred[index] != label and y_test[index] == label:
					FN = FN + 1

			precision = TP / (TP + FP)
			recall = TP / (TP + FN)
			f1 = (2 * precision * recall) / (precision + recall)
			f1_scores.append(f1)

		f1_macro = sum(f1_scores) / len(f1_scores)
        
		return {"f1_macro": f1_macro}


	def grid_search(self, k_values, x_test, y_test):
		best_k = None
		best_f1_score_macro = 0
		for value in k_values:
			self.n_neighbors = value
			metrics = self.evaluate(x_test, y_test)
			if metrics["f1_macro"] > best_f1_score_macro:
				best_f1_score_macro = metrics["f1_macro"]
				best_k = value
		return {
			"best_k": best_k,
			"best_f1_score_macro": best_f1_score_macro
		}



# légende pour comprendre : 
	# x = le nouveau oint à prédire
	# self.x_train = toutes les lignes d'entraînement (le dataset entier)
	# self.y_train = donnée target, ce que l'on cherchera à prédire
	# x_train_point = une seule ligne de x_train (à chaque tour de boucle)
	# index = le numéro de la colonne qu'on compare (salaire, age etc)
	# distance = la distance euclidienne entre x et x_train_point
	# self.n_neighbors = le nombre de voisins à considérer (notre k)
	# distances.append(distance) = la liste de toutes les distances calculées
	# neighbor = l'un des voisins les plus proche de ce que l'on souahite prédire
	# y_label_nearest_neighbors = liste des valeurs y des N voisins les plus proches de x
	# y_label_nearest = un item de y_label_nearest_neighbors
	# target_distribution = liste de la distribution de y pour y_label_nearest_neighbors
	# x_test et y_test = partie de la donnée que l'on a mis de côté et que l'on va utiliser pour tester les prédictione t leur fiabilité
	# y_pred_index = un item du résultat de la focntion predict (x_test[index] étant un élèment du data set de test)
	# x_true_index = la bonne réponse pour valider ou invalider y_pred_index (pareil il s'agit ici de la bonne réponse pour un élèment de x_test)
	# best_k = meilleur k (cf. n_neighbord) pour avoir le meilleur f1_score paramétré sur None car il sera mis à jour grâce à une boucle for
	# best_f1_score = meilleur score associé au best_k









		
	
		