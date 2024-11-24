"""
=========================================================
Linear Regression Example
=========================================================
The example below uses only the first feature of the `diabetes` dataset,
in order to illustrate the data points within the two-dimensional plot.
The straight line can be seen in the plot, showing how linear regression
attempts to draw a straight line that will best minimize the
residual sum of squares between the observed responses in the dataset,
and the responses predicted by the linear approximation.

The coefficients, residual sum of squares and the coefficient of
determination are also calculated.

"""

# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# print(diabetes_X.shape) # --> Should give output (442, 10)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]
# print(diabetes_X.shape) # --> Should give output (442, 1)

"""
'diabetes_X[:, np.newaxis, 2]' uses slicing and 'np.newaxis' to select a specific feature and reshape the data.
It is actually done in two parts:

    1.  diabetes_X[:, 2] --> selects the third feature (since indexing is zero-based) from each sample in the dataset.
        If 'diabetes_X' originally has a shape of (n_samples, n_features) (here, the shape was '(442, 10)'),
        this slicing operation would result in a 1D array of shape (n_samples,) (here, '(442,)'), containing
        only the third feature for each sample.

    2.  'np.newaxis' adds a new axis to the array. When used in slicing, it increases the dimensionality of the array by one.

Combining these, 'diabetes_X[:, np.newaxis, 2]' reshapes the 1D array (n_samples,) (here, '(442,)')
to a 2D array of shape (n_samples, 1) (here, '(442, 1)'). This effectively converts the selected feature
from a column vector to a 2D array with one column.
"""

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# print(diabetes_X_train.shape)
# print(diabetes_X_test.shape)

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
# print(diabetes_y_train.shape)
# print(diabetes_y_test.shape)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, label="Test Data Points", color="skyblue")
plt.plot(
    diabetes_X_test, diabetes_y_pred, color="red", linewidth=1, label="Regression Line"
)
plt.xlabel("X_test")
plt.ylabel("Predicted y_test")
plt.xticks(())
plt.yticks(())
plt.title("Simple Linear Regression Model")
plt.legend()
plt.show()
