from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
df = pd.read_csv('Crop_recommendation.csv',
                 dtype={'N': 'int64', 'P': 'int64', 'K': 'int64',
                        'temperature': 'float64', 'humidity': 'float64',
                        'ph': 'float64', 'rainfall': 'float64'})

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2%}")
print(f"\nPrecision Score: {precision_score(y_test, y_pred, average='weighted'):.2%}")
print(f"\nRecall Score: {recall_score(y_test, y_pred, average='weighted'):.2%}")
print(f"\nF1 Score: {f1_score(y_test, y_pred, average='weighted'):.2%}")
print(f"\nLearned coefficients: [{', '.join([f'{coeff:.2f}' for coeff in model.coef_[0, :]])}]")
print(f"\nIntercept (for first class): {model.intercept_[0]:.2f}")


