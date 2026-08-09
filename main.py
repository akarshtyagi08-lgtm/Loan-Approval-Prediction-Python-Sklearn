# Import required Modules
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load Dataset as data
data = pd.read_csv("loan_approval_dataset.csv", skipinitialspace=True)

# Remove white spaces
data.columns = data.columns.str.strip()

# One-Hot Encoding using pandas
data = pd.get_dummies(data, drop_first=True)

# Input and Output
X = data.drop(columns=['loan_status_Rejected', 'loan_id'])
y = data['loan_status_Rejected']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initilize Model
model = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=42)

# Train Model
model.fit(X_train, y_train)

# Get Predictions for testing
y_pred = model.predict(X_test)

# Get full report
print(f"Confusion Metrix:\n\n{confusion_matrix(y_test, y_pred)}\n\nClassification Report:\n\n{classification_report(y_test, y_pred)}")