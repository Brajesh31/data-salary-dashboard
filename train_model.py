import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the dataset
df = pd.read_csv("ds_salaries.csv")

# Preprocessing: rename experience levels
df['experience_level'] = df['experience_level'].map({
    'EN': 'Entry',
    'MI': 'Mid',
    'SE': 'Senior',
    'EX': 'Executive'
})

# Feature columns and target
X = df[['job_title', 'experience_level', 'employment_type', 'company_location', 'remote_ratio']]
y = df['salary_in_usd']

# Column transformer
categorical = ['job_title', 'experience_level', 'employment_type', 'company_location']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical)],
    remainder='passthrough'
)

# Build pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"MAE: {mae:.2f}")

# Save the model
joblib.dump(model, "salary_model.pkl")
