import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv("cmu-sleep.csv")

# Preprocessing: Encode categorical variables
label_encoders = {}
for column in ['cohort', 'demo_race', 'demo_gender', 'demo_firstgen', 'term_units',
               'Zterm_units_ZofZ']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target variable
X = data[['TotalSleepTime', 'midpoint_sleep', 'daytime_sleep', 'term_units',
          'frac_nights_with_data']]
y = data['cum_gpa']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                    random_state=42)

# Hyperparameter tuning for Decision Tree
dt_params = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  # Updated values
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
dt_grid_search = GridSearchCV(DecisionTreeRegressor(), dt_params, cv=5,
                              scoring='neg_mean_squared_error', error_score='raise')
dt_grid_search.fit(X_train, y_train)
dt_best_model = dt_grid_search.best_estimator_
print(f"Best Decision Tree parameters: {dt_grid_search.best_params_}")

# Save the best Decision Tree model
joblib.dump(dt_best_model, 'decision_tree_model.joblib')

# Hyperparameter tuning for Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}
rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_params, cv=5,
                              scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train, y_train)
rf_best_model = rf_grid_search.best_estimator_
print(f"Best Random Forest parameters: {rf_grid_search.best_params_}")

# Save the best Random Forest model
joblib.dump(rf_best_model, 'random_forest_model.joblib')

# Train and evaluate models with scaled features
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': dt_best_model,
    'Random Forest': rf_best_model
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse}")
    print(f"{name} - R-squared: {r2}")

    # Save model
    joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.joblib')
