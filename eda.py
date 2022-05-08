import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.utils import shuffle

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

def statistics(all_patients):
	df = all_patients[0].mean().to_frame().T
	for i, patient in enumerate(all_patients):
		df = pd.concat([df, patient.mean().to_frame().T], ignore_index=True)
	total_summary = df.count() / len(df)
	print(total_summary)
	drop_columns = ['ICULOS']
	for item in total_summary.iteritems():
		if item[1] < 0.8:
			drop_columns.append(item[0])
	print(drop_columns)
	df.drop(drop_columns, axis=1, inplace=True)
	numerical_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose', 'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'HospAdmTime', 'num_hours']
	categorial_columns = ['Gender']
	scatt = pd.plotting.scatter_matrix(df, figsize=(25, 25))
	plt.savefig("../all_scatter.jpg") 
	plt.close()
	for col in numerical_columns:
		if col in ['Magnesium', 'HospAdmTime']:
			plt.hist(df[col], bins=7)
		else:
			plt.hist(df[col])
		plt.title(col + " Histogram")
		plt.savefig("../" + col + "_Histogram.jpg")
		plt.close()
		first_box = df[col][df.label == 1].dropna()
		second_box = df[col][df.label == 0].dropna()
		plt.boxplot([first_box, second_box])
		plt.title(col + " Boxplot")
		plt.savefig("../" + col + "_boxplot.jpg")
		plt.close()
	for col in categorial_columns:
		print("male with sepsis", df[col][df[col] == 1][df.label == 1].dropna().count())
		print("female with sepsis", df[col][df[col] == 0][df.label == 1].dropna().count())
		print("male without sepsis", df[col][df[col] == 1][df.label == 0].dropna().count())
		print("female without sepsis", df[col][df[col] == 0][df.label == 0].dropna().count())
		first_box = df[col][df.label == 1].dropna()
		second_box = df[col][df.label == 0].dropna()
		plt.boxplot([first_box, second_box])
		plt.title(col + " Boxplot")
		plt.savefig("../" + col + "_boxplot.jpg")
		plt.close()
	hypothesis_testing_t = ['HR', 'Temp', 'Resp']
	hypothesis_testing_u = ['BUN', 'HospAdmTime']
	for col in hypothesis_testing_t:
		first_group = df[col][df.label == 1].dropna()
		second_group = df[col][df.label == 0].dropna()
		print(col,":", ttest_ind(first_group, second_group))
	for col in hypothesis_testing_u:
		first_group = df[col][df.label == 1].dropna()
		second_group = df[col][df.label == 0].dropna()
		print(col,":", mannwhitneyu(first_group, second_group))


def create_set(patients, keep_columns, min_max_columns, train=True):
	min_dict = {}
	for col in min_max_columns:
		min_dict[col] = "Min_" + col
	max_dict = {}
	for col in min_max_columns:
		max_dict[col] = "Max_" + col
	if train == True:
		j = 0
		while patients[j].mean()['HospAdmTime'] < -4000:
			j += 1
		max_df = patients[j].max().to_frame().T[min_max_columns].rename(columns = max_dict)
		min_df = patients[j].min().to_frame().T[min_max_columns].rename(columns = min_dict)
		df = pd.concat([patients[j][keep_columns].mean().to_frame().T, max_df], axis=1)
		df = pd.concat([df, min_df], axis=1)
		for i, patient in enumerate(patients):
			if i <= j or patient.mean()['HospAdmTime'] < -4000:
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

if __name__ == '__main__':
	all_patients = read_dir(r"/home/student/data/train")
	statistics(all_patients)
	keep_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose', 'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'num_hours', 'label']
	min_max_colums = ['HR', 'O2Sat', 'Temp', 'MAP']
	nan_df, df = create_set(all_patients, keep_columns, min_max_colums, train=True)
	nan_df = shuffle(nan_df)
	df = shuffle(df)
	nan_df.to_csv("../nan_train.csv")
	df.to_csv("../train.csv")
	test_patients = read_dir(r"/home/student/data/test")
	nan_test, test = create_set(test_patients, keep_columns, min_max_colums, train=False)
	nan_test.to_csv("../nan_test.csv")
	test.to_csv("../test.csv")