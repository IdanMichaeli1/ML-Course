import numpy as np


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (y - np.mean(y)) / (np.max(y) - np.min(y))

    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    ones = np.ones(X.shape[0])
    return np.column_stack((ones, X))


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    error = np.dot(X, theta) - y
    return np.sum(error ** 2) / (2 * len(X))


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()
    J_history = []

    for _ in range(num_iters):

        J_history.append(compute_cost(X, y, theta))
        error = np.dot(X, theta) - y
        gradient = np.dot(X.T, error) / len(X)
        theta = theta - alpha * gradient

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    inverse_X_transpose_X = np.linalg.inv(np.matmul(X.T, X))
    pinv_theta = np.matmul(np.matmul(inverse_X_transpose_X, X.T), y)

    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration

    for t in range(num_iters):

        J_history.append(compute_cost(X, y, theta))
        if t > 0 and (J_history[t-1] - J_history[t] < 10**-8):
            break

        error = np.dot(X, theta) - y
        gradient = np.dot(X.T, error) / len(X)
        theta = theta - alpha * gradient

    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003,
              0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}
    theta = np.random.random(size=X_train.shape[1])

    for alpha in alphas:

        params, _ = efficient_gradient_descent(
            X_train, y_train, theta, alpha, iterations)

        alpha_dict[alpha] = compute_cost(X_val, y_val, params)

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []

    # Apply the bias trick to the input data
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)

    # Add the index of the column of the bias to the selected features
    selected_features = [0]

    # Initialize a training data set that will hold the chosen features
    X_train_selected_feat = np.ones((X_train.shape[0], 1))

    # Iterate 5 times to select the top 5 features
    for i in range(5):

        features_dict = {}
        theta = np.random.random(size=i + 2)

        # Iterate through each feature
        for feature in range(1, X_train.shape[1]):
            if feature not in selected_features:
                X_train_selected_feat = np.column_stack(
                    (X_train_selected_feat, X_train[:, feature]))
                selected_features.append(feature)

                params, _ = efficient_gradient_descent(
                    X_train_selected_feat, y_train, theta, best_alpha, iterations)

                features_dict[feature] = compute_cost(
                    X_val[:, selected_features], y_val, params)

                X_train_selected_feat = X_train_selected_feat[:, :-1]
                selected_features.pop()

        # Select the feature with the lowest cost
        min_value_key = min(features_dict, key=features_dict.get)

        # Add the selected feature to the selected features list permanently
        X_train_selected_feat = np.column_stack(
            (X_train_selected_feat, X_train[:, min_value_key]))
        selected_features.append(min_value_key)

    selected_features.remove(0)
    selected_features = [idx - 1 for idx in selected_features]

    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    for i, col in enumerate(df.columns):
        # Create a new column with the square of the original column
        df_poly[f'{col}^2'] = df[col] ** 2

        # Add interaction terms between columns
        for col2 in df.columns[i+1:]:
            df_poly[f'{col}*{col2}'] = df[col] * df[col2]

    return df_poly
