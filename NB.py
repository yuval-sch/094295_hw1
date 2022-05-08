import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def main(path, test_path):
	df = pd.read_csv(path)
	y = df['label']
	X = df.drop('label', axis=1).reset_index(drop=True)
	X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
	categorical_pipeline = Pipeline(steps=[("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False))])
	numeric_pipeline = Pipeline(steps=[("scale", StandardScaler())])
	cat_cols = ['Gender']
	num_cols = list(X.columns.drop('Gender'))
	full_processor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ])
	NB_clf = BernoulliNB()
	X_processed = full_processor.fit_transform(X)
	param_grid = {
     	"alpha": [0, 0.02, 0.05, 0.08, 0.1],
     	"binarize": [0.5, 1, 2, 3],
	"fit_prior": [True, False]
	 }

	grid_cv = GridSearchCV(NB_clf, param_grid, n_jobs=-1, cv=5, scoring="f1")
	_ = grid_cv.fit(X_processed, y)
	print(grid_cv.best_score_)
	print(grid_cv.best_params_)
	final_nb = BernoulliNB(alpha=0, binarize=1, fit_prior=True)
	final_nb.fit(X_processed, y)
	y_pred = final_nb.predict(X_processed)
	print("Training results:")
	print(classification_report(y, y_pred))
	plot_confusion_matrix(final_nb, X_processed, y)
	plt.savefig("./train_conf.jpg")
	plt.close()
	test_df = pd.read_csv(test_path)
	test_y = test_df['label']
	test_X = test_df.drop('label', axis=1).reset_index(drop=True)
	test_X = test_X.loc[:, ~test_X.columns.str.contains('^Unnamed')]
	test_X_processed = full_processor.fit_transform(test_X)
	test_y_pred = final_nb.predict(test_X_processed)
	print("Test results:")
	print(classification_report(test_y, test_y_pred))
	plot_confusion_matrix(final_nb, test_X_processed, test_y)
	plt.savefig("./test_conf.jpg")
	plt.close()
	print(final_nb.feature_log_prob_)


if __name__ == '__main__':
	main(r"/home/student/data/train.csv", r"/home/student/data/test.csv")
