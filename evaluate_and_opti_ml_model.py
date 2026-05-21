class EVALOPTI(object):

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

	def grid_search(self, setting_model, setting_values, x_test, y_test):
		best_setting_model = None
		best_f1_score_macro = 0
		for value in setting_values:
			setattr(self, setting_model, value)
			metrics = self.evaluate(x_test, y_test)

			if metrics["f1_macro"] > best_f1_score_macro:
				best_f1_score_macro = metrics["f1_macro"]
				best_setting_model = value		
				
		return {
			"best_setting_model": best_setting_model,
			"best_f1_score_macro": best_f1_score_macro
		}
