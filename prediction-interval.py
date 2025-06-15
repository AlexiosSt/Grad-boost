import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from sklearn.model_selection import train_test_split

# Generate synthetic data
def f(x):
    return x * np.sin(x)

rng = np.random.RandomState(42)
X = np.atleast_2d(rng.uniform(0, 10, 1000)).T
expected_y = f(X).ravel()

# Add noise to the targets
sigma=0.5+X.ravel()/10
noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
y = expected_y + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train a Gradient Boosting Regressor
all_models = {}
common_params = dict(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=2,
    min_samples_split=9,
    min_samples_leaf=9,
)

for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, **common_params)
    all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)

# Fit a usual squared error model
gbr_ls = GradientBoostingRegressor(loss='squared_error', **common_params)
all_models["squared error"] = gbr_ls.fit(X_train, y_train)

# Make predictions 
xx= np.atleast_2d(np.linspace(0, 10, 1000)).T

y_pred = all_models["squared error"].predict(xx)
y_pred_lower = all_models["q 0.05"].predict(xx)
y_pred_upper = all_models["q 0.95"].predict(xx)
y_pred_median = all_models["q 0.50"].predict(xx)

# Plot the results
fig = plt.figure(figsize=(10, 8))
plt.plot(xx, f(xx), 'black', label=r'$f(x)=x\, \sin(x)$', linewidth=3)
plt.plot(X_test, y_test, 'b.', markersize=10, label='Test data')
plt.plot(xx, y_pred_median, 'tab:orange', label='Median prediction', linewidth=3)
plt.plot(xx, y_pred, 'tab:green', label='Mean prediction', linewidth=3)
plt.fill_between(
    xx.ravel(),
    y_pred_lower,
    y_pred_upper,
    color='gray',
    alpha=0.4,
    label='90% prediction interval',
)
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Gradient Boosting Regression with Prediction Intervals')
plt.legend(loc="upper left")
#plt.xlim(0, 10)
plt.ylim(-10, 20)
plt.grid()
fig.savefig('gradient_boosting_prediction_intervals.png', dpi=300)