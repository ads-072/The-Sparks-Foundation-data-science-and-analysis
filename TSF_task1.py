#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression using Scikit learn

# In[4]:


##reading data from given path
path="http://bit.ly/w-data"
p=pd.read_csv(path)
print("Imported data successfully")
p.head(10)


# # plot data

# In[7]:


p.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # In this graph we can see positive linear relations between number of hours studied and percentage scored

# # Preparing data

# In[28]:


X = p.iloc[:, :-1].values  
y = p.iloc[:, 1].values 


# In[19]:


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# # Training the algorithm

# In[20]:


# Plotting the regression line
line=regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[21]:


print(X_test)
y_pred = regressor.predict(X_test)


# In[23]:


#comparing actual and predicted data
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # Evaluating the model

# In[24]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




