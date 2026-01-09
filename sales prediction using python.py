# ================================
# SALES PREDICTION USING PYTHON
# ================================
# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Create sample dataset
data = {
    'Advertising_Spend': [1000, 1500, 2000, 2500, 3000, 3500, 4000],
    'Target_Segment': ['Youth', 'Adult', 'Youth', 'Senior', 'Adult', 'Youth', 'Senior'],
    'Platform': ['Online', 'TV', 'Online', 'TV', 'Online', 'Social Media', 'Social Media'],
    'Sales': [12000, 18000, 25000, 22000, 30000, 36000, 40000]
}
df = pd.DataFrame(data)

# 3. Data Cleaning
df.dropna(inplace=True)

# 4. Data Transformation (Encoding categorical data)
df_encoded = pd.get_dummies(df, drop_first=True)

# 5. Feature Selection
X = df_encoded.drop('Sales', axis=1)
y = df_encoded['Sales']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Predict Sales
y_pred = model.predict(X_test)

# 9. Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Model Evaluation ===")
print("Actual Sales:", y_test.values)
print("Predicted Sales:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 10. Advertising Impact Analysis
plt.figure(figsize=(8, 6))
plt.scatter(df['Advertising_Spend'], df['Sales'], color='blue', s=100)
plt.xlabel("Advertising Spend ($)")
plt.ylabel("Sales ($)")
plt.title("Advertising Spend vs Sales")
plt.grid(True, alpha=0.3)
plt.show()

# 11. Predict Future Sales
# CRITICAL FIX: Ensure column names match exactly what the model expects
print("\n=== Feature columns used by model ===")
print(X.columns.tolist())

# Create new data with EXACT column names from training
new_data = pd.DataFrame({
    'Advertising_Spend': [4500],
    'Target_Segment_Senior': [0],
    'Target_Segment_Youth': [1],
    'Platform_Social Media': [1],
    'Platform_TV': [0]
})

# Reorder columns to match training data
new_data = new_data[X.columns]

future_sales = model.predict(new_data)
print("\n=== Future Sales Prediction ===")
print(f"Predicted Future Sales: ${future_sales[0]:,.2f}")