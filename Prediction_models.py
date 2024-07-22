#!/usr/bin/env python
# coding: utf-8

# <h3>IPL Score Prediction using Machine Learning</h3>

# This Machine learning model adapts a Regression Approach to Predict the score of first inning of an IPL Match.

# <h3>Import Libraries</h3>

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# <H3>Load the dataset</H3>

# In[2]:


ipl_df = pd.read_csv('ipl_data.csv')
ipl_df.shape


# <h3>Exploratory Data Analysis</h3>

# In[3]:


#first five column data 
ipl_df.head()


# In[4]:


ipl_df.describe()


# In[5]:


ipl_df.info()


# In[6]:


ipl_df.nunique()


# In[7]:


ipl_df.dtypes


# In[8]:


#wickets Distribution
sns.displot(ipl_df['wickets'],kde=False,bins=10)
plt.title("Wickets Distribution")

plt.show()


# In[9]:


#Runs Distribution
sns.displot(ipl_df['total'],kde=False,bins=10)
plt.title("Runs Distribution")

plt.show()


# <h3>Data Cleaning</h3>

# <h4>Removing Irrelevant Data columns</h4>

# In[10]:


#names of all columns
ipl_df.columns


# Here we can see that columns ['mid', 'date', 'venue','batsman', 'bowler','striker',
#        'non-striker'] wont provide any relevant information to train

# In[11]:


irrelevant = ['mid', 'date', 'venue','batsman', 'bowler','striker', 'non-striker']
print(f'Before removing irrelevant Columns:{ipl_df.shape}')
ipl_df = ipl_df.drop(irrelevant, axis=1)
print(f'After removing irrelevant Columns:{ipl_df.shape}')
ipl_df.head()


# <h4>Keeping only Consistent Teams</h4>

# In[12]:


#Define Consistent Teams
Const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 
               'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore', 
               'Delhi Daredevils', 'Sunrisers Hyderabad']


# In[13]:


print(f'Before Removing Inconsistent teams : {ipl_df.shape}')
ipl_df = ipl_df[(ipl_df['bat_team'].isin(Const_teams)) & (ipl_df['bowl_team'].isin(Const_teams))]
print(f'After Removing Inconsistent teams : {ipl_df.shape}')
print(f"Consistent Teams : \n{ipl_df['bat_team'].unique()}")
ipl_df.head()


# <h4>Remove First 5 overs of every match</h4>

# In[14]:


print(f'Before Removing Overs : {ipl_df.shape}')
ipl_df = ipl_df[ipl_df['overs'] >= 5.0]
print(f'After Removing Overs : {ipl_df.shape}')
ipl_df.head()


# <h3>Plotting a Correlation Matrix of current data</h3>

# In[15]:


from seaborn import heatmap
heatmap(data=ipl_df.select_dtypes(include=['float64', 'int64']).corr(), annot=True)

plt.show()


# <h3>Data Pre-Processing and Encoding</h3>

# <H4>Perfoming Label Encoding</H4>

# In[16]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
for col in ['bat_team', 'bowl_team']:
    ipl_df[col] = le.fit_transform(ipl_df[col])
ipl_df.head()


# In[17]:


from sklearn.compose import ColumnTransformer
ColumnTransformer = ColumnTransformer([('encoder',
                                        OneHotEncoder(),
                                        [0,1])],
                                      remainder='passthrough')


# In[18]:


ipl_df = np.array(ColumnTransformer.fit_transform(ipl_df))


# <h4>Saving the numpy array in a new dataframe with transformed columns</h4>

# In[19]:


cols = ['Batting_team_Chennai Super Kings', 'Batting_team_Delhi Daredevils', 'Batting_team_Kings XI Punjab', 
        'Batting_team_Kolkata Knight Riders', 'Batting_team_Mumbai Indians', 'Batting_team_Rajasthan Royals', 
        'Batting_team_Royal Challengers Bangalore', 'Batting_team_Sunrisers Hyderabad',
        'Bowling_team_Chennai Super Kings', 'Bowling_team_Delhi Daredevils', 'Bowling_team_Kings XI Punjab', 
        'Bowling_team_Kolkata Knight Riders', 'Bowling_team_Mumbai Indians', 'Bowling_team_Rajasthan Royals', 
        'Bowling_team_Royal Challengers Bangalore', 'Bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
        'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(ipl_df, columns=cols)


# In[20]:


#Encoded data
df.head()


# <h3>Model Building</h3>

# <h4>Prepare, train and test data</h4>

# In[21]:


features = df.drop(['total'], axis=1)
labels = df['total']


# In[22]:


from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)
print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")


# <H3>ML Algorithms</H3>

# In[23]:


models = dict()


# <h4>1.Decision Tree Regressor</h4>

# In[24]:


from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
#train model
tree.fit(train_features, train_labels)


# In[25]:


#Evaluate the models
train_score_tree = str(tree.score(train_features, train_labels)*100)
test_score_tree = str(tree.score(test_features, test_labels)*100)
print(f'Train Score : {train_score_tree[:5]}%\nTest Score : {test_score_tree[:5]}%')
models["tree"] = test_score_tree


# In[26]:


from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse 
print("-------Decision Tree Regressor Evaluation--------")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, tree.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, tree.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, tree.predict(test_features)))))


# <h3>Linear Regression</h3>

# In[27]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
#Train Model
linreg.fit(train_features, train_labels)


# In[28]:


#Evaluate Model
train_score_linreg = str(linreg.score(train_features, train_labels) * 100)
test_score_linreg = str(linreg.score(test_features, test_labels) * 100)
print(f'Train Score : {train_score_linreg[:5]}%\nTest Score : {test_score_linreg[:5]}%')
models["linreg"] = test_score_linreg


# In[29]:


print("-------Linear Regressor Evaluation--------")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, linreg.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, linreg.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, linreg.predict(test_features)))))


# <h3>Random Forest Regressor</h3>

# In[30]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
#Train model
forest.fit(train_features, train_labels)


# In[31]:


#Evaluate Model
train_score_forest = str(forest.score(train_features, train_labels) * 100)
test_score_forest = str(forest.score(test_features, test_labels) * 100)
print(f'Train Score : {train_score_forest[:5]}%\nTest Score : {test_score_forest[:5]}%')
models["forest"] = test_score_forest


# In[32]:


print("-------Random Forest Regression - Model Evaluation--------")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, forest.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, forest.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, forest.predict(test_features)))))


# <h3>Support Vector Machine</h3>

# In[33]:


from sklearn.svm import SVR
svm = SVR()
#Train Model
svm.fit(train_features, train_labels)


# In[34]:


#Evaluate Model
train_score_svm = str(svm.score(train_features, train_labels) * 100)
test_score_svm = str(svm.score(test_features, test_labels) * 100)
print(f'Train Score : {train_score_svm[:5]}%\nTest Score : {test_score_svm[:5]}%')
models["svm"] = test_score_svm


# In[35]:


print("-------Support Vector Regression - Model Evaluation--------")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, svm.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, svm.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, svm.predict(test_features)))))


# <h3>XGBoost</h3>

# In[36]:


from xgboost import XGBRegressor
xgb = XGBRegressor()
#Train Model
xgb.fit(train_features, train_labels)


# In[37]:


#Evaluate Model
train_score_xgb = str(xgb.score(train_features, train_labels) * 100)
test_score_xgb = str(xgb.score(test_features, test_labels) * 100)
print(f'Train Score : {train_score_xgb[:5]}%\nTest Score : {test_score_xgb[:5]}%')
models["xgb"] = test_score_xgb


# In[38]:


print("-------XGB Regression - Model Evaluation--------")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, xgb.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, xgb.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, xgb.predict(test_features)))))


# <h3>KNR</h3>

# In[39]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
#train model
knr.fit(train_features, train_labels)


# In[40]:


#Evaluate Model
train_score_knr = str(knr.score(train_features, train_labels) * 100)
test_score_knr = str(knr.score(test_features, test_labels) * 100)
print(f'Train Score : {train_score_knr[:5]}%\nTest Score : {test_score_knr[:5]}%')
models["knr"] = test_score_knr


# In[41]:


print("-------KNR - Model Evaluation--------")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, knr.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, knr.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, knr.predict(test_features)))))


# <h3>Best Model</h3>

# In[42]:


import matplotlib.pyplot as plt 
model_names = list(models.keys())
accuracy = list(map(float, models.values()))
#creating the bar plot
plt.bar(model_names, accuracy)
plt.show()


# from above, we can see that **Random Forest** has done a good job, closely followed by **Decision Tree** and **KNR**.
# so **Random Forest** will be the final model.

# <h3>Predictions</h3>

# In[43]:


def score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model = forest):
    prediction_array = []
    #Batting Team
    if batting_team == 'Chennai Super Kings':
        prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
    elif batting_team == 'Delhi Daredevils':
        prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
    elif batting_team == 'Kings XI Punjab':
        prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
    elif batting_team == 'Kolkata Knight Riders':
        prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
    elif batting_team == 'Mumbai Indians':
        prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
    elif batting_team == 'Rajasthan Royals':
        prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
    elif batting_team == 'Royal Challengers Bangalore':
        prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
    elif batting_team == 'Sunrisers Hyderabad':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
    #Bowling team
    if bowling_team == 'Chennai Super Kings':
        prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
    elif bowling_team == 'Delhi Daredevils':
        prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
    elif bowling_team == 'Kings XI Punjab':
        prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
    elif bowling_team == 'Kolkata Knight Riders':
        prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
    elif bowling_team == 'Mumbai Indians':
        prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
    elif bowling_team == 'Rajasthan Royals':
        prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
    elif bowling_team == 'Royal Challengers Bangalore':
        prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
    elif bowling_team == 'Sunrisers Hyderabad':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
    prediction_array = prediction_array + [runs, wickets, overs, runs_last_5, wickets_last_5]
    prediction_array = np.array([prediction_array])
    pred = model.predict(prediction_array)
    return int(round(pred[0]))  


# <h3>Test 1</h3>
# <ul><li>Batting Team: <b>Delhi Daredevils<b></li>
#     <li>Bowling Team: <b>Chennai Super Kings<b></li>
#     <li>Final Score: <b>147/9<b></li>
# </ul>

# In[50]:


batting_team = 'Delhi Daredevils'
bowling_team = 'Chennai Super Kings'
score = score_predict(batting_team, bowling_team, overs=10.2, runs=68, wickets=3, runs_last_5=29, wickets_last_5=1)
print(f'Predicted Score:{score} || Actual Score:147')


# <h3>Test 2</h3>
# <ul><li>Batting Team: <b>Mumbai Indians<b></li>
#     <li>Bowling Team: <b>Kings XI Punjab<b></li>
#     <li>Final Score: <b>176/7<b></li>
# </ul>

# In[63]:


batting_team = 'Mumbai Indians'
bowling_team = 'Kings XI Punjab'
score = score_predict(batting_team, bowling_team, overs=12.3, runs=113, wickets=2, runs_last_5=55, wickets_last_5=0)
print(f'Predicted Score:{score} || Actual Score:176')


# <h3>Test 3</h3>
# <ul><li>Batting Team: <b>Kings XI Punjab<b></li>
#     <li>Bowling Team: <b>Rajasthan Royals<b></li>
#     <li>Final Score: <b>185/4<b></li>
# </ul>

# In[66]:


batting_team = 'Kings XI Punjab'
bowling_team = 'Rajasthan Royals'
score = score_predict(batting_team, bowling_team, overs=14.0, runs=118, wickets=1, runs_last_5=45, wickets_last_5=0)
print(f'Predicted Score:{score} || Actual Score:185')


# <h3>Test 4</h3>
# <ul><li>Batting Team: <b>Kolkata knight Riders<b></li>
#     <li>Bowling Team: <b>Chennai Super Kings<b></li>
#     <li>Final Score: <b>172/5<b></li>
# </ul>

# In[65]:


batting_team = 'Kolkata Knight Riders'
bowling_team = 'Chennai Super Kings'
score = score_predict(batting_team, bowling_team, overs=18.0, runs=150, wickets=4, runs_last_5=57, wickets_last_5=1)
print(f'Predicted Score:{score} || Actual Score:172')


# <h3>Test 5</h3>
# <ul><li>Batting Team: <b>Delhi Daredevils<b></li>
#     <li>Bowling Team: <b>Mumbai Indians<b></li>
#     <li>Final Score: <b>110/7<b></li>
# </ul>

# In[70]:


batting_team = 'Delhi Daredevils'
bowling_team = 'Mumbai Indians'
score = score_predict(batting_team, bowling_team, overs=18.0, runs=96, wickets=8, runs_last_5=18, wickets_last_5=4)
print(f'Predicted Score:{score} || Actual Score:110')


# <h3>Test 6</h3>
# <ul><li>Batting Team: <b>Kings XI Punjab<b></li>
#     <li>Bowling Team: <b>Chennai Super Kings<b></li>
#     <li>Final Score: <b>153/9<b></li>
# </ul>

# In[72]:


batting_team = 'Kings XI Punjab'
bowling_team = 'Chennai Super Kings'
score = score_predict(batting_team, bowling_team, overs=18.0, runs=129, wickets=6, runs_last_5=34, wickets_last_5=2)
print(f'Predicted Score:{score} || Actual Score:153')


# <h3>Test 7</h3>
# <ul><li>Batting Team: <b>Sunrisers Hyderabad<b></li>
#     <li>Bowling Team: <b>Royal challengers Bangalore<b></li>
#     <li>Final Score: <b>146/10<b></li>
# </ul>

# In[73]:


batting_team = 'Sunrisers Hyderabad'
bowling_team = 'Royal Challengers Bangalore'
score = score_predict(batting_team, bowling_team, overs=10.5, runs=67, wickets=3, runs_last_5=29, wickets_last_5=1)
print(f'Predicted Score:{score} || Actual Score:146')


# <h3>Export Model</h3>

# In[77]:


import pickle
filename = "ml_model.pkl"
with open(filename, "wb") as file:
    pickle.dump(forest, file)

