import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
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
	X = df.drop('label', axis=1)
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
	X_processed = full_processor.fit_transform(X)
	param_grid = {
    "max_depth": [8, 9, 10],
    "learning_rate": [0.09, 0.1, 0.2, 0.3],
    "gamma": [3, 5, 7],
    "reg_lambda": [1.5, 1.7, 2],
    "scale_pos_weight": [7, 8, 9],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
	}
	xgb_cl = xgb.XGBClassifier(objective="binary:logistic")
	grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="f1")
	_ = grid_cv.fit(X_processed, y)
	print(grid_cv.best_score_)
	print(grid_cv.best_params_)
	final_xgb = xgb.XGBClassifier(objective="binary:logistic", colsample_bytree=0.5, gamma=5, learning_rate=0.1, max_depth=9, reg_lambda=1.7, scale_pos_weight=8, subsample=0.8)
	final_xgb.fit(X_processed, y)
	y_pred = final_xgb.predict(X_processed)
	print("Training results:")
	print(classification_report(y, y_pred))
	plot_confusion_matrix(final_xgb, X_processed, y)
	plt.savefig("./train_conf_xgb.jpg")
	plt.close()
	test_df = pd.read_csv(test_path)
	test_y = test_df['label']
	test_X = test_df.drop('label', axis=1)
	test_X = test_X.loc[:, ~test_X.columns.str.contains('^Unnamed')]
	test_X_processed = full_processor.fit_transform(test_X)
	test_y_pred = final_xgb.predict(test_X_processed)
	print("Test results:")
	print(classification_report(test_y, test_y_pred))
	plot_confusion_matrix(final_xgb, test_X_processed, test_y)
	plt.savefig("./test_conf_xgb.jpg")
	plt.close()
	ax = xgb.plot_importance(final_xgb)
	ax.figure.savefig('./xgb_importance.jpg')
	ax = xgb.plot_importance(final_xgb, importance_type="cover")
	ax.figure.savefig('./xgb_importance_weighted.jpg')
	final_xgb.save_model("./xgboost.model")	


if __name__ == '__main__':
	main(r"/home/student/data/nan_train.csv", r"/home/student/data/nan_test.csv")
