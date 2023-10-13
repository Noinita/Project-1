import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn as sk
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler



#Question 2.1
df = pd.read_csv("data\\Project 1 Data.csv")
df= df.dropna()
#Question 2.2

#Scatter Plot
attributes = ["X", "Y", "Z", "Step"]
pd.plotting.scatter_matrix(df[attributes], figsize=(12, 8))
plt.show()

#Question 2.3

# Calculating the Pearson correlation
correlation_coeff = df[attributes].corr()

# Creating a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_coeff, annot=True, cmap='inferno', fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()



#Question 2.4

train_y = df['Step']
df_X = df.drop(columns = ["Step"])


my_scaler = StandardScaler()
my_scaler.fit(df_X.iloc[:,:])
scaled_data = my_scaler.transform(df_X.iloc[:,:])
scaled_data_df = pd.DataFrame(scaled_data, columns=df_X.columns[:])
train_x = scaled_data_df
columns_list = train_x.columns.tolist()

# first classification model

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(train_x, train_y)
some_data = train_x.iloc[:10]
some_data.columns = train_x.columns
some_house_values = train_y.iloc[:10]

for i in range(10):
    some_predictions = model1.predict(some_data.iloc[i].values.reshape(1, -1))
    some_actual_values = some_house_values.iloc[i]
    print("Predictions:", some_predictions)
    print("Actual values:", some_actual_values)

model1_prediction = model1.predict(train_x)
from sklearn.metrics import mean_absolute_error
model1_train_mae = mean_absolute_error(model1_prediction, train_y)
model1_prediction = pd.DataFrame(model1_prediction).astype(int)
print("Model 1 training MAE is: ", round(model1_train_mae,2))

# second classification model 
from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(n_estimators=30, random_state=3)
model2.fit(train_x, train_y)
model2_prediction = model2.predict(train_x)
model2_train_mae = mean_absolute_error(model2_prediction, train_y)
model2_prediction = pd.DataFrame(model2_prediction).astype(int)
print("Model 2 training MAE is: ", round(model2_train_mae,2))

#third classification model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model3 = LogisticRegression(max_iter=1000, random_state=3)
model3.fit(train_x, train_y)
model3_prediction = model3.predict(train_x)
model3_train_mae = mean_absolute_error(model3_prediction, train_y)
model3_prediction = pd.DataFrame(model3_prediction).astype(int)
print("Model 3 training MAE is: ", round(model3_train_mae,2))



for i in range(10):
     some_predictions1 = model1.predict(some_data.iloc[i].values.reshape(1, -1))
     some_predictions2 = model2.predict(some_data.iloc[i].values.reshape(1, -1))
     some_predictions3 = model3.predict(some_data.iloc[i].values.reshape(1, -1))
     
     some_actual_values = some_house_values.iloc[i]
     print("Predictions Model 1:", some_predictions1)
     print("Predictions Model 2:", some_predictions2)
     print("Predictions Model 3:", some_predictions3)
     print("Actual values:", some_actual_values)
     
     
#cross validation
from sklearn.model_selection import cross_val_score
model1 = LinearRegression()
model1.fit(train_x, train_y)
model2 = RandomForestRegressor(n_estimators=30, random_state=3)
model2.fit(train_x, train_y)
model3 = LogisticRegression(max_iter=1000, random_state=3)
model3.fit(train_x, train_y)


# Perform k-fold cross-validation for Model 1
scores_model1 = cross_val_score(model1, train_x, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model1 = -scores_model1.mean()
print("Model 1 Mean Absolute Error (CV):", round(mae_model1, 2))

# Perform k-fold cross-validation for Model 2
scores_model2 = cross_val_score(model2, train_x, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model2 = -scores_model2.mean()
print("Model 2 Mean Absolute Error (CV):", round(mae_model2, 2))

# Perform k-fold cross-validation for Model 3
scores_model3 = cross_val_score(model3, train_x, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model3 = -scores_model2.mean()
print("Model 3 Mean Absolute Error (CV):", round(mae_model3, 2))



# #GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
     'fit_intercept': [True, False], 
   
    
    }

model1 = LinearRegression()
grid_search = GridSearchCV(model1, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_x, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters model 1:", best_params)
best_model1 = grid_search.best_estimator_



param_grid = {
     'n_estimators': [10, 30, 50],
     'max_depth': [None, 10, 20, 30],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4],
     'max_features': ['sqrt', 'log2']
 }
model2 = RandomForestRegressor(random_state=42)
model2.fit(train_x, train_y)
grid_search = GridSearchCV(model2, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_x, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters model 2:", best_params)
best_model2 = grid_search.best_estimator_

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  
    'penalty': ['l1', 'l2'],  
}


model3 = LogisticRegression(max_iter=1000, random_state=3)  # Use the same random_state as before
model3.fit(train_x, train_y)
grid_search = GridSearchCV(model3, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_x, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters for model 3:", best_params)



#Question 2.5
train_y = pd.DataFrame(train_y)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Comparisons for model 1
accmodel1 = accuracy_score(train_y.iloc[:, 0], model1_prediction.iloc[:, 0])
precmodel1 = precision_score(train_y.iloc[:, 0], model1_prediction.iloc[:, 0], average= 'weighted')
f1model1 = f1_score(train_y.iloc[:, 0], model1_prediction.iloc[:, 0], average= 'weighted')
print("The accuracy score for Model 1 is:",accmodel1 )
print("The precision score for Model 1 is:",precmodel1)
print("The F1 score for Model 1 is:",f1model1 ) 

#Comparisons for model 2
accmodel2 = accuracy_score(train_y.iloc[:, 0], model2_prediction.iloc[:, 0])
precmodel2 = precision_score(train_y.iloc[:, 0], model2_prediction.iloc[:, 0], average= 'weighted')
f1model2 = f1_score(train_y.iloc[:, 0], model2_prediction.iloc[:, 0], average= 'weighted')
print("The accuracy score for Model 2 is:",accmodel2 )
print("The precision score for Model 2 is:",precmodel2)
print("The F1 score for Model 2 is:",f1model2) 

#Comparisons for model 3
accmodel3 = accuracy_score(train_y.iloc[:, 0], model3_prediction.iloc[:, 0])
precmodel3 = precision_score(train_y.iloc[:, 0], model3_prediction.iloc[:, 0], average= 'weighted')
f1model3 = f1_score(train_y.iloc[:, 0], model3_prediction.iloc[:, 0], average= 'weighted')
print("The accuracy score for Model 3 is:",accmodel3)
print("The precision score for Model 3 is:",precmodel3)
print("The F1 score for Model 3 is:",f1model3) 


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

confusion_matrix_model1 = confusion_matrix(train_y, model1_prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_model1, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Model 1')
plt.show()

# Create confusion matrix for Model 2
confusion_matrix_model2 = confusion_matrix(train_y, model2_prediction)

# Create a heatmap for Model 2
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_model2, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Model 2')
plt.show()

# Creating confusion matrix for Model 3
confusion_matrix_model3 = confusion_matrix(train_y, model3_prediction)

# Creating heatmap for Model 3
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_model3, annot=True, fmt='d', cmap='YlOrRd')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model 3')
plt.show()


#Step 2.6
#Choosing model 2 as the desired model 
selected_model = model2
joblib.dump(selected_model, 'selected_model.joblib')
loaded_model = joblib.load('selected_model.joblib')
coordinates = [[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]]
new_data_scaled = my_scaler.transform(coordinates)
predictions = loaded_model.predict(new_data_scaled)
print('The Predictions Are', predictions)




