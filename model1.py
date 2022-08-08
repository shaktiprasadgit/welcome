import pandas as pd
MyData = pd.read_csv(r"C:\Users\Ashish\Desktop\Velocity\DATA SCI BATCH\Main Notes\Flask\prediction_Expense\Income_Expense_Data.csv")

MyData.isnull().sum()


MyData["Income"].fillna((MyData["Income"].median()), inplace = True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(MyData)
scaled_data


MyData_scaled = pd.DataFrame(scaled_data)
MyData_scaled.columns = ["Age","Income","Expense"]

features = ["Income","Age"]
response = ["Expense"]
X=MyData_scaled[features]
y=MyData_scaled[response]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Importing neccesary packages
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

import pickle
pickle.dump(model, open('model1.pkl','wb'))