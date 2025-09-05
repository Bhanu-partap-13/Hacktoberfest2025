from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their possible values
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'beta': [0.01, 0.1, 1.0, 10.0]
}

# Create a model
model = SomeModel()

# Use grid search to find the optimal hyperparameters
grid_search = GridSearchCV(model, param_grid)
grid_search.fit(X, y)

# Print the optimal values for the hyperparameters
print(grid_search.best_params_)
