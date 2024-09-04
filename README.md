**CMPT419-Project**

## In this project, we had done analysis and prediciton of the product category and prices of high end luxury brands like Gucci, Louis Vuitton, and Loro Piana - across Canada, the USA, and the UK

**Libraries used:**
* Anaconda: https://www.anaconda.com

**Anaconda has the following:**
* Pandas
* numpy
* sklearn
* matplotlib

### Note : If you don't have any of the above say sklearn use pip install scikit-learn to install it

**Python Configuration and Modules used:**
* Python 3.11.5
* import pandas as pd
* import numpy as np
* import matplotlib.pyplot as plt
* from scipy import stats
* from sklearn.model_selection import train_test_split, GridSearchCV
* from sklearn.linear_model import LinearRegression, Ridge
* from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
* from sklearn.neighbors import KNeighborsRegressor
* from sklearn.metrics import mean_squared_error, r2_score
* from sklearn.preprocessing import StandardScaler
* from sklearn.ensemble import RandomForestClassifier
* from sklearn.ensemble import GradientBoostingClassifier
* from sklearn.pipeline import make_pipeline
* from sklearn.neighbors import KNeighborsClassifier
* from sklearn.metrics import classification_report, accuracy_score

### Order of execution & Commands (and arguments):

### Data Cleaning and Exploration
* Run Data_Load.ipynb first
* Run Data_ETL_Gucci.ipynb
* Run Data_ETL_LV.ipynb
* Run Data_ETL_LP.ipynb 

### Data Stats
* Run Full_Data_Statistics.ipynb 

### Data Prediction
#### For Product Category: 
* Run Classification_model.ipynb

#### For Price Prediction: 
* Run Regression_Model.ipynb

### Data Influence
* Run data_exploration.ipynb
  
### Files produced/expected
### Data Cleaning
* Final_data.csv

### Data Influence
* output_complete_influence_scores.csv

