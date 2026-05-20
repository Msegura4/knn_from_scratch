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










		
	
		