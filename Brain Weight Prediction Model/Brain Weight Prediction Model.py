"""Brain Weight Prediction Model Using Head Size and Demographics"""

# Import Libraries and Modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Read DataSet
df = pd.read_csv("headbrain.csv")

# Define which columns are category-features and which are numeric-features
# Gender: 1 = Male, 2 = Female
# Age Range: 1 = 20-46 yrs, 2 = 46+ yrs
cat_features = ['Gender', 'Age Range']   # encoded as 1/2 in the dataset
num_features = ['Head Size(cm^3)']       # range ~2720 to 4747

# Split Data
X = df.iloc[:, 0:3]     # Gender, Age Range and Head Size
y = df.iloc[:, -1]      # Brain Weight

# Train_Test_Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode cols 0,1 as categories. Scale col 2 as number
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), cat_features),   # drop='first' avoids dummy trap
        ('num', StandardScaler(), num_features)
    ])

# Pipeline: preprocess -> then LinearRegression
pipe = make_pipeline(preprocessor, LinearRegression())

# Training Model
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Metrics Results
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")

# See feature names after encoding
print("\nFeatures after encoding:")
print(pipe.named_steps['columntransformer'].get_feature_names_out())
print("Coefficients:", pipe.named_steps['linearregression'].coef_)

# Compute residuals once, reused across all plots below
residuals = y_test - y_pred
print(f"\nMean residual: {residuals.mean():.2f}g")
print(f"Std of residuals: {residuals.std():.2f}g")

# 1. Actual vs Predicted scatter - shows model accuracy
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # perfect line
plt.xlabel('Actual Brain Weight (g)')
plt.ylabel('Predicted Brain Weight (g)')
plt.title('Actual vs Predicted Brain Weight')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Residuals plot - shows where model struggles
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Brain Weight (g)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.show()

# 3. Residual Histogram - shows if errors are "well-behaved"
plt.figure(figsize=(7, 5))
sns.histplot(residuals, bins=20, kde=True)   # kde = smooth curve
plt.axvline(x=0, color='r', linestyle='--', lw=2)   # zero-error line
plt.xlabel('Residuals = Actual - Predicted (g)')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.grid(True, alpha=0.3)
plt.show()