#!/usr/bin/env python
# coding: utf-8

# ### Importing required libraries

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load the Dataset

# In[50]:


df = pd.read_csv('CAR DETAILS.csv')


# ### Display top 5 rows

# In[51]:


df.head()


# In[52]:


df["brand"] = df["name"].agg(str.split).str[0]
print(df["brand"].value_counts())


# ### Checking the shape

# In[53]:


df.shape


# ### The dataset contains 4340 rows and 8 columns of customer information.

# In[54]:


df.columns


# ### checking the datatype

# In[55]:


df.dtypes


# ### Changing the datatype of date column

# In[56]:


df['year'] = pd.to_datetime(df['year'], format='%Y').dt.year

df.head()


# ### Handling Duplicates

# In[57]:


df.duplicated().sum()


# In[58]:


df = df.drop_duplicates()
df.shape


# ### EDA-

# ### Handling null values

# In[59]:


nv = df.isnull().sum()
nv = nv[nv>0]
nv


# ### No missing values are observed in the dataset

# In[60]:


# Generate summary statistics for numerical columns
print(df.describe())


# In[61]:


# Bar plot for the fuel column
plt.figure(figsize=(8, 6))
sns.countplot(x='fuel', data=df)
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.title('Distribution of Fuel Types')
plt.show()


# In[62]:


# Histogram for the 'selling_price' column
plt.figure(figsize=(8, 6))
plt.hist(df['selling_price'], bins=30, edgecolor='black')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.title('Histogram of Selling Price')
plt.show()


# In[63]:


# Box plot to identify potential outliers in 'selling_price'
plt.figure(figsize=(8, 6))
sns.boxplot(df['selling_price'])
plt.xlabel('Selling Price')
plt.title('Box Plot: Selling Price')
plt.show()


# In[64]:


sorted(df['selling_price'],reverse=True)


# In[65]:


df = df[~(df['selling_price']>=5500000) & (df['selling_price']<=8900000)]


# In[66]:


# Scatter plot for 'km_driven' vs. 'selling_price'
plt.figure(figsize=(8, 6))
plt.scatter(df['km_driven'], df['selling_price'], alpha=0.5)
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price')
plt.title('Scatter Plot: Kilometers Driven vs. Selling Price')
plt.grid(True)
plt.show()


# In[67]:


sorted(df['km_driven'],reverse=True)


# In[68]:


df = df[~(df['km_driven']>=400000) & (df['km_driven']<=806599)]


# In[69]:


# Box plot for 'fuel' vs. 'selling_price'
plt.figure(figsize=(8, 6))
sns.boxplot(x='fuel', y='selling_price', data=df)
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price')
plt.title('Box Plot: Fuel Type vs. Selling Price')
plt.show()


# In[70]:


print(df.info())
cars_train = df[["year", "km_driven", "fuel", "seller_type", "transmission", "brand", "owner"]]
print(df["brand"].head())


# In[71]:


from sklearn import preprocessing


# In[72]:


print(cars_train["seller_type"].unique())
le_seller = preprocessing.LabelEncoder()
le_seller.fit(cars_train["seller_type"])
cars_train["seller_type"] = le_seller.transform(cars_train["seller_type"])
print(cars_train["seller_type"].head())


# In[73]:


print(cars_train["transmission"].unique())
le_trans = preprocessing.LabelEncoder()
le_trans.fit(cars_train["transmission"])
cars_train["transmission"] = le_trans.transform(cars_train["transmission"])
print(cars_train["transmission"].head())


# In[74]:


print(cars_train["fuel"].unique())
le_fuel = preprocessing.LabelEncoder()
le_fuel.fit(cars_train["fuel"])
cars_train["fuel"] = le_fuel.transform(cars_train["fuel"])
print(cars_train["fuel"].head())


# In[75]:


print(cars_train["owner"].unique())
le_owner = preprocessing.LabelEncoder()
le_owner.fit(cars_train["owner"])
cars_train["owner"] = le_owner.transform(cars_train["owner"])
print(cars_train["owner"].head())


# In[76]:


le_brand = preprocessing.LabelEncoder()
le_brand.fit(cars_train["brand"])
cars_train["brand"] = le_brand.transform(cars_train["brand"])
print(cars_train["brand"].head())


# In[77]:


y = df["selling_price"]
print(y.shape)
x = cars_train
print(x.shape)


# In[78]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3,shuffle=True)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# ### Data Pre-Processing:

# Prepare the Data for Machine Learning Modeling

#  Apply Various Machine Learning Techniques

# In[79]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


# Create and train different models
model1 = LinearRegression()
model2 = RandomForestRegressor()
model3 = Ridge(alpha=0.4)
model4 = Lasso(alpha = 15.7)


# Fit the models on the training data
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


# Creating Functions to compute Regression Evaluation Metrics

# In[80]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def reg_eval_metrics(ytest,ypred):
  mae = mean_absolute_error(ytest,ypred)
  mse = mean_squared_error(ytest,ypred)
  rmse = np.sqrt(mse)
  r2 = r2_score(ytest,ypred)
  print('MAE',mae)
  print('MSE',mse)
  print('RMSE',rmse)
  print('R2 Score',r2)

def mscore(model):
  print('Training Score',model.score(x_train,y_train))  # Training Score
  print('Testing Score',model.score(x_test,y_test))     #Testing R2 score


# ### Predictions

# LinearRegression

# In[81]:


mscore(model1)


# In[82]:


ypred_model1 = model1.predict(x_test)


# In[83]:


reg_eval_metrics(y_test,ypred_model1)


# RandomForestRegressor

# In[84]:


mscore(model2)


# In[85]:


ypred_model2 = model2.predict(x_test)


# In[86]:


reg_eval_metrics(y_test,ypred_model2)


# Ridge

# In[87]:


mscore(model3)


# In[88]:


ypred_model3 = model3.predict(x_test)


# In[89]:


reg_eval_metrics(y_test,ypred_model3)


# Lasso

# In[90]:


mscore(model4)


# In[91]:


ypred_model4 = model4.predict(x_test)


# In[92]:


reg_eval_metrics(y_test,ypred_model4)


# In[93]:


import joblib

car_price_prediction = 'best_model.pkl'
joblib.dump(model2, car_price_prediction)


# In[94]:


# Load the saved model
car_price_prediction = 'best_model.pkl'
loaded_model = joblib.load(car_price_prediction)

# Randomly select 20 data points from the original dataset
new_dataset = df.sample(n=20, random_state=42)

le_seller = preprocessing.LabelEncoder()
le_seller.fit(new_dataset["seller_type"])
new_dataset["seller_type"] = le_seller.transform(new_dataset["seller_type"])

le_trans = preprocessing.LabelEncoder()
le_trans.fit(new_dataset["transmission"])
new_dataset["transmission"] = le_trans.transform(new_dataset["transmission"])

le_fuel = preprocessing.LabelEncoder()
le_fuel.fit(new_dataset["fuel"])
new_dataset["fuel"] = le_fuel.transform(new_dataset["fuel"])

le_owner = preprocessing.LabelEncoder()
le_owner.fit(new_dataset["owner"])
new_dataset["owner"] = le_owner.transform(new_dataset["owner"])


le_brand = preprocessing.LabelEncoder()
le_brand.fit(new_dataset["brand"])
new_dataset["brand"] = le_brand.transform(new_dataset["brand"])

X_new = new_dataset[["year", "km_driven", "fuel", "seller_type", "transmission", "brand", "owner"]]

# Apply the loaded model to the new dataset
predictions = loaded_model.predict(X_new)

# Print the predictions
print(predictions)



# In[95]:


import streamlit as st


# In[96]:


def main():
    st.title('Car Price Prediction App')
    st.write('Enter the car details to predict its price:')
    
    # Input fields for car details
    year = st.number_input('Year', min_value=1990, max_value=2023, step=1)
    km_driven = st.number_input('Kilometers Driven', min_value=0, step=1000)
    fuel = st.selectbox('Fuel', ['Petrol', 'Diesel', 'CNG', 'LPG'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    brand = st.text_input('Brand', '')
    owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
    
    # Predict button
    if st.button('Predict'):
        # Preprocess input data
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'brand': [brand],
            'owner': [owner]
        })
        
        le_seller = preprocessing.LabelEncoder()
        le_seller.fit(input_data["seller_type"])
        input_data["seller_type"] = le_seller.transform(input_data["seller_type"])

        le_trans = preprocessing.LabelEncoder()
        le_trans.fit(input_data["transmission"])
        input_data["transmission"] = le_trans.transform(input_data["transmission"])

        le_fuel = preprocessing.LabelEncoder()
        le_fuel.fit(input_data["fuel"])
        input_data["fuel"] = le_fuel.transform(input_data["fuel"])

        le_owner = preprocessing.LabelEncoder()
        le_owner.fit(input_data["owner"])
        input_data["owner"] = le_owner.transform(input_data["owner"])


        le_brand = preprocessing.LabelEncoder()
        le_brand.fit(input_data["brand"])
        input_data["brand"] = le_brand.transform(input_data["brand"])
        
        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)
        st.success(f'Predicted Car Price: ₹{prediction[0]:,.2f}')
    
if __name__ == '__main__':
    main()

