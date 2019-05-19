
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
#read csv data and display first 5 rows
customers=pd.read_csv('Ecommerce Customers.csv')
customers.head()
customers.describe()
customers.columns
customers.info()

#seaborn library to analyse data pictrically
sns.jointplot(data=customers,x='Time on App',y='Yearly Amount Spent')
sns.jointplot(data=customers,x='Time on App',y='Yearly Amount Spent',kind='hex')
sns.pairplot(customers)#from the above figure 'yearly amount spent' is highly corelated to 'length of membership' 


#so from the above deduced relationship construct lmplot() which is basically a regression plot
sns.lmplot(data=customers,x='Length of Membership',y='Yearly Amount Spent')

#creating training and testing data
customers.columns
y=customers['Yearly Amount Spent']
x=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

#importing ML libraries
from sklearn.model_selection import train_test_split
train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=101)
#importing Linear Regression Model
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
lm.coef_
predictions=lm.predict(x_test)

#using matplotlib to get pictorial view of regreesion model 
plt.scatter(y_test,predictions)
plt.xlabel('Y test(true values)')
plt.ylabel('Predicted values')

#checking for the models accuracy percentage-98.8 %accuracy
accuracy=lm.score( x_test,y_test)
print(accuracy*100,'%')

#calculation of metrics of predictions 
from sklearn import metrics
print("mean absolute error",metrics.mean_absolute_error(y_test,predictions))
print("mean squared error",metrics.mean_squared_error(y_test,predictions))
print("root mean squared error",np.sqrt(metrics.mean_squared_error(y_test,predictions)))

metrics.explained_variance_score(y_test,predictions)

#residuals-plotting histogram
sns.distplot((y_test-predictions),bins=50)

cdf=pd.DataFrame(lm.coef_,x.columns,columns=['coeff'])
cdf
#result from prediction is--->since coeff of "Time on app" is more company should focus on improving app better 

