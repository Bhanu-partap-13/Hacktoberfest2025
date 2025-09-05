from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define the hyperparameters and their distributions
param_distributions = {
    'alpha': uniform(0.01, 10.0),
    'beta': uniform(0.01, 10.0)
}

# Create a model
model = SomeModel()

# Use randomized search to find the optimal hyperparameters
random_search = RandomizedSearchCV(model,
                                   param_distributions)
random_search.fit(X, y)

# Print the optimal values for the hyperparameters
print(random_search.best_params_)
