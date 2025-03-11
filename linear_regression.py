import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data: Study Hours vs Exam Score
data = {'Study Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Exam Score': [50, 55, 65, 70, 75, 78, 85, 88, 92, 95]}  
df = pd.DataFrame(data)

#print df
print(df)

# Scatter Plot
plt.scatter(df['Study Hours'], df['Exam Score'], color='blue', label='Actual Data')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.legend() 
plt.show()

# Splitting Data (80% Train, 20% Test)
x = df[['Study Hours']]  # Input feature  
y = df['Exam Score']  # Output

# When you use single square brackets, like df['Study Hours'], it returns a Pandas Series.
# A Series is essentially a one-dimensional array (column) with labels (index).

# When you use double square brackets, like df[['Study Hours']], it returns a Pandas DataFrame.
# A DataFrame is a two-dimensional table (like a spreadsheet) with both rows and columns

#print x which is study hours
print(x)
print(y)

#split your dataset into two parts
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print( "training X dataset") 
print(X_train)

print( "training Y dataset") 
print(y_train)

print( "testing X dataset")
print(X_test)

print( "testing Y dataset")
print(y_test)

# Training the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Coefficients
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (c): {model.intercept_}")

# Predict on Test Data
y_pred = model.predict(X_test)
print(f"predicted values : {y_pred}") 

# Compare Predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

# Visualizing Predictions
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Actual Test Data')
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.legend()
plt.show()

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Predict for a new student who studies 7.5 hours
new_study_hours = np.array([[7.5]])
predicted_score = model.predict(new_study_hours)
print(f"Predicted Score for 7.5 study hours: {predicted_score[0]:.2f}")
