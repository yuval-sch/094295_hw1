import pandas as pd
import xgboost as xgb
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def read_dir(path):
	patients = []
	os.chdir(path)
	for file in os.listdir():
		if file.endswith(".psv"):
			pid = file[8 : -4]
			file_path = path + '/' + file
			with open(file_path, 'r') as f:
				labels = []
				df = pd.read_csv(f, delimiter='|')
				df['id'] = [int(pid)] * len(df)
				sepsis = False
				sepsis_index = 0
				for i, row in enumerate(df.iterrows()):
					row = row[1]
					if row['SepsisLabel'] == 1:
						labels.append(1)
						sepsis = True
						sepsis_index = i
						break
				df['num_hours'] = [i + 1] * len(df)
				if sepsis:
					df['label'] = labels * len(df)
					df = df.iloc[: sepsis_index + 1, :]
				else:
					df['label'] = [0] * len(df)
				df.drop('SepsisLabel', axis=1, inplace=True)
				patients.append(df)
	return patients


def create_set(patients, keep_columns, min_max_columns, train=True):
	min_dict = {}
	for col in min_max_columns:
		min_dict[col] = "Min_" + col
	max_dict = {}
	for col in min_max_columns:
		max_dict[col] = "Max_" + col
	if train == True:
		j = 0
		while patients[j].mean()['HospAdmTime'] < -4000 or patients[j].mean()['HospAdmTime'] > 0:
			j += 1
		max_df = patients[j].max().to_frame().T[min_max_columns].rename(columns = max_dict)
		min_df = patients[j].min().to_frame().T[min_max_columns].rename(columns = min_dict)
		df = pd.concat([patients[j][keep_columns].mean().to_frame().T, max_df], axis=1)
		df = pd.concat([df, min_df], axis=1)
		for i, patient in enumerate(patients):
			if i <= j or patient.mean()['HospAdmTime'] < -4000 or patient.mean()['HospAdmTime'] > 0:
				continue
			curr_max = patient.max().to_frame().T[min_max_columns].rename(columns = max_dict)
			curr_min = patient.min().to_frame().T[min_max_columns].rename(columns = min_dict)
			curr_df = pd.concat([patient[keep_columns].mean().to_frame().T, curr_max], axis=1)
			curr_df = pd.concat([curr_df, curr_min], axis=1)
			df = pd.concat([df,curr_df], ignore_index=True)
	else:
		max_df = patients[0].max().to_frame().T[min_max_columns].rename(columns = max_dict)
		min_df = patients[0].min().to_frame().T[min_max_columns].rename(columns = min_dict)
		df = pd.concat([patients[0][keep_columns].mean().to_frame().T, max_df], axis=1)
		df = pd.concat([df, min_df], axis=1)
		for i, patient in enumerate(patients):
			if i == 0:
				continue
			curr_max = patient.max().to_frame().T[min_max_columns].rename(columns = max_dict)
			curr_min = patient.min().to_frame().T[min_max_columns].rename(columns = min_dict)
			curr_df = pd.concat([patient[keep_columns].mean().to_frame().T, curr_max], axis=1)
			curr_df = pd.concat([curr_df, curr_min], axis=1)
			df = pd.concat([df,curr_df], ignore_index=True)
	df1 = df.copy()
	for col in df1.columns:
		df1[col] = df1[col].fillna(df1[col].median())
	return df, df1

def predict(test_path):
	loaded_xgb = xgb.XGBClassifier()
	loaded_xgb.load_model("./xgboost.model")
	keep_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose', 'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'num_hours', 'label', 'id']
	min_max_colums = ['HR', 'O2Sat', 'Temp', 'MAP']
	test_patients = read_dir(test_path)
	nan_test, _ = create_set(test_patients, keep_columns, min_max_colums, train=False)
	nan_test = nan_test.drop('label', axis=1)
	X_test = nan_test.drop('id', axis=1)
	categorical_pipeline = Pipeline(steps=[("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False))])
	numeric_pipeline = Pipeline(steps=[("scale", StandardScaler())])
	cat_cols = ['Gender']
	num_cols = list(X_test.columns.drop('Gender'))
	full_processor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ])
	X_processed = full_processor.fit_transform(X_test)
	y_pred = loaded_xgb.predict(X_processed)
	df = pd.concat([nan_test["id"].astype('int'), pd.Series(y_pred)], axis=1)
	df.to_csv("./prediction.csv", header=False, index=False)


if __name__ == '__main__':
	predict(sys.argv[1])
