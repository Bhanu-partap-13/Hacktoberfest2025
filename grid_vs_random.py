import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,\
GridSearchCV
from scipy.stats import uniform

# Generate a toy dataset
X = np.random.rand(200, 10)
y = np.random.randint(2, size=200)
# Define the model and the hyperparameter
# search space
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': np.linspace(0.1, 1, 11),
    'bootstrap': [True, False]
}

# Use RandomizedSearchCV to sample
# from the search space and fit the model
random_search = RandomizedSearchCV(
  model,
  param_grid, 
  cv=5, 
  n_iter=10, 
  random_state=42)
random_search.fit(X, y)

# Use GridSearchCV to explore the entire search space and fit the model
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

# Print the best hyperparameters found by each method
print(f"Best hyperparameters found by RandomizedSearchCV: {random_search.best_params_}")
print(f"Best hyperparameters found by GridSearchCV: {grid_search.best_params_}")
