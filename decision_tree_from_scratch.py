from evaluate_and_opti_ml_model import EVALOPTI


class Node(object):

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree(EVALOPTI):

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, x_train, y_train):
        self.root = self._build_tree(x_train, y_train, depth=0)

    def _build_tree(self, x_train, y_train, depth):
        # 1. conditions pour arrêter la construction de l'arbre
        if depth == self.max_depth or len(set(y_train)) == 1:
            leaf_value = max(set(y_train), key=lambda item_y_train: y_train.count(item_y_train))
            return Node(value=leaf_value)

        # 2. trouver le meilleur split
        best_feature_index, best_threshold = self._best_split(x_train, y_train)

        # 3. créer le nœud et récurser
        pass

    def _best_split(self, x_train, y_train):
        best_gini = float("inf")
        best_feature_index = None
        best_threshold = None

        for feature_index in range(len(x_train[0])):
            feature_values = [x_train[index][feature_index] for index in range(len(x_train))]
            sorted_feature_values = sorted(set(feature_values))
            thresholds = [(sorted_feature_values[index] + sorted_feature_values[index + 1]) / 2
                          for index in range(len(sorted_feature_values) - 1)]

            for threshold in thresholds:
                y_left = []
                y_right = []

                for index in range(len(x_train)):
                    if x_train[index][feature_index] <= threshold:
                        y_left.append(y_train[index])
                    else:
                        y_right.append(y_train[index])

    def predict(self, x):
        pass