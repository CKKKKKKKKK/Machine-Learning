# Import 3rd party dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import normalize


## Model Implementation
# The Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Logistic Regression
class LogisticRegression():
    def __init__(self, learning_rate=.1, n_iterations=4000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def initialize_weights(self, n_features):
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        # Insert 0 as w_0
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        normal_X = normalize(X, norm = 'max')
        m_samples, n_features = normal_X.shape
        self.initialize_weights(n_features)
        # Insert a column of 1 as x_0
        normal_X = np.insert(normal_X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))
        for i in range(self.n_iterations):
            h_x = normal_X.dot(self.w)
            y_pred = sigmoid(h_x)
            w_grad = normal_X.T.dot(y_pred - y)
            self.w = self.w - self.learning_rate * w_grad
        return self.w

    def predict(self, X):
        normal_X = normalize(X, norm = 'max')
        normal_X = np.insert(normal_X, 0, 1, axis=1)
        h_x = normal_X.dot(self.w)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)

    def test(self, X, y):
        normal_X = normalize(X, norm = 'max')
        m_samples = normal_X.shape[0]
        normal_X = np.insert(normal_X, 0, 1, axis=1)
        h_x = normal_X.dot(self.w)
        y_pred = np.round(sigmoid(h_x))
        right_count = 0
        for i in range(m_samples):
            if y_pred[i] == y[i]:
                right_count += 1
        return right_count / m_samples



## Load training data
data = pd.read_csv("archive/train.csv")


## Data Preprocessing
# Convert labels into to two classes: low (0, 1) and high (2, 3)
data["price_classification"] = np.where(data["price_range"] <= 1, 0, 1)


## Data Splitting
# Split the original ‘train.csv’ into ‘train.csv’, ‘valid.csv’ and ‘test.csv’ with the ratio of 0.8 : 0.1 : 0.1
train_data = data.sample(frac = 0.8)
valid_data = data.drop(train_data.index).sample(frac = 0.5)
test_data = data.drop(train_data.index).drop(valid_data.index)
train_data.to_csv("train.csv")
valid_data.to_csv("valid.csv")
test_data.to_csv("test.csv")


## Train the Logistic Regression Model
# Decide what fields we want to process
output_param_name = 'price_classification'
price_range = 'price_range'

# Split training set into input and output
x_train = train_data.drop([output_param_name],axis=1,inplace=False).drop([price_range],axis=1,inplace=False).values
y_train = train_data[[output_param_name]].values

# Split test set into input and output
x_test = test_data.drop([output_param_name],axis=1,inplace=False).drop([price_range],axis=1,inplace=False).values
y_test = test_data[[output_param_name]].values

# Train model
logistic_regression = LogisticRegression()
theta = logistic_regression.fit(x_train, y_train)

# Print model parameters
theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})
print(theta_table)

# Test the Logistic Regression Model
accuracy_result = logistic_regression.test(x_test, y_test)
print('Accuracy: {:.2f}' .format(accuracy_result))


# # Decide what fields we want to process.
# input_param_name_1 = 'ram'
# input_param_name_2 = 'wifi'
# output_param_name = 'price_range'

# # Split training set input and output.
# x_train = train_data[[input_param_name_1, input_param_name_2]].values
# y_train = train_data[[output_param_name]].values

# # Split test set input and output.
# x_test = test_data[[input_param_name_1, input_param_name_2]].values
# y_test = test_data[[output_param_name]].values

# # Split test set input and output.
# x_valid = valid_data[[input_param_name_1, input_param_name_2]].values
# y_valid = valid_data[[output_param_name]].values

# # Configure the plot with training dataset.
# plot_training_trace = go.Scatter3d(
#     x=x_train[:, 0].flatten(),
#     y=x_train[:, 1].flatten(),
#     z=y_train.flatten(),
#     name='Training Set',
#     mode='markers',
#     marker={
#         'size': 10,
#         'opacity': 1,
#         'line': {
#             'color': 'rgb(255, 255, 255)',
#             'width': 1
#         },
#     }
# )

# # Configure the plot with test dataset.
# plot_test_trace = go.Scatter3d(
#     x=x_test[:, 0].flatten(),
#     y=x_test[:, 1].flatten(),
#     z=y_test.flatten(),
#     name='Test Set',
#     mode='markers',
#     marker={
#         'size': 10,
#         'opacity': 1,
#         'line': {
#             'color': 'rgb(255, 255, 255)',
#             'width': 1
#         },
#     }
# )

# # Configure the plot with test dataset.
# plot_valid_trace = go.Scatter3d(
#     x=x_valid[:, 0].flatten(),
#     y=x_valid[:, 1].flatten(),
#     z=y_valid.flatten(),
#     name='Valid Set',
#     mode='markers',
#     marker={
#         'size': 10,
#         'opacity': 1,
#         'line': {
#             'color': 'rgb(255, 255, 255)',
#             'width': 1
#         },
#     }
# )

# # Configure the layout.
# plot_layout = go.Layout(
#     title='Data Sets',
#     scene={
#         'xaxis': {'title': input_param_name_1},
#         'yaxis': {'title': input_param_name_2},
#         'zaxis': {'title': output_param_name}
#     },
#     margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
# )
# plot_data = [plot_training_trace, plot_test_trace, plot_valid_trace]
# plot_figure = go.Figure(data=plot_data, layout=plot_layout)

# # Render 3D scatter plot.
# plot_figure.show()



