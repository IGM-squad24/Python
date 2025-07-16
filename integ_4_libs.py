#---------------Integrative Scenario (All 4 LibrariesCombined)-------------------#

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv('student_scores.csv')

print("#------------------------Pandas-----------------------#\n")
print("Dataset Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

df['Average_Score'] = df[['Math_Score', 'Science_Score', 'English_Score']].mean(axis=1)

print("\nDataset with Average Score:")
print(df[['Student_ID', 'Name', 'Math_Score', 'Science_Score', 'English_Score', 'Average_Score']])


print("\n#------------------------Numpy-----------------------#\n")
scores = df[['Math_Score', 'Science_Score', 'English_Score']].to_numpy()

std_dev = np.std(scores, axis=0)
print("\nStandard Deviation of Scores:")
print(f"Math Score Std Dev: {std_dev[0]}")
print(f"Science Score Std Dev: {std_dev[1]}")
print(f"English Score Std Dev: {std_dev[2]}")


print("\n#------------------------Using Scikit-learn Preprocessing-----------------------#\n")
scaler = StandardScaler()

print("\n#------------------------Using Scikit-learn Preprocessing(StandardScaler)-----------------------#\n")
normalized_scores = scaler.fit_transform(df[['Math_Score', 'Science_Score', 'English_Score']])

df[['Math_Score_Normalized', 'Science_Score_Normalized', 'English_Score_Normalized']] = normalized_scores

print("\nDataset with Normalized Scores:")
print(df[['Student_ID', 'Name', 'Math_Score', 'Science_Score', 'English_Score', 'Math_Score_Normalized', 'Science_Score_Normalized', 'English_Score_Normalized']])

print("\n#------------------------Using Scikit-learn Preprocessing(OneHotEncode)-----------------------#\n")

encoder = OneHotEncoder(sparse_output=False)  
city_encoded = encoder.fit_transform(df[['City']])

city_encoded_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(['City']))

df_encoded = pd.concat([df, city_encoded_df], axis=1)

print("\nDataset with One-Hot Encoded Cities:")
print(df_encoded[['Student_ID', 'Name', 'City', 'Math_Score', 'Science_Score', 'English_Score'] + list(city_encoded_df.columns)])
