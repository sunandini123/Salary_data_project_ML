import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


dataset=pd.read_csv(r"/Users/shashi/Desktop/Salary_Data.csv")


dataset

x=dataset.iloc[:,:-1].values 

y=dataset.iloc[:,-1].values 

# Introduces & splits data 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0 )



# creates linear model line y=mx+c , where m is slope which says about increase in salary  and c is intercept denotes the basic salary 

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)


y_pred=regressor.predict(X_test)

# comparison gives both the actual & predicted values 
comparison = pd.DataFrame({ 'Actual' : y_test , 'predicted' :  y_pred})

print(comparison)




import matplotlib.pyplot as plt

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



# VALIDATIONS 0 PREDICTing the future salary exactly 
# validations phase always work on future data (unseem data)

## MIN & MAX EXPERIENCE 


m_slope=regressor.coef_
print(m_slope)


c_intercept=regressor.intercept_
print(c_intercept)


# y 12 years experience predicted value 

y_12 = m_slope*12+c_intercept
print(y_12)


















































