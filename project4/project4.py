import numpy as np


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)
        X = preprocess(X)
        X = apply_bias_trick(X)
        self.theta = np.random.random(size=X.shape[1])

        for t in range(self.n_iter):

            self.thetas.append(self.theta)
            self.Js.append(self.compute_cost(X, y))

            if t > 0 and np.abs(self.Js[t-1] - self.Js[t]) < self.eps:
                break

            self.theta = self.theta - self.eta * self.gradient(X, y)

    def sigmoid(self, X):
        power = -np.dot(X, self.theta)
        return 1 / (1 + np.exp(power))

    def gradient(self, X, y):
        return np.dot((self.sigmoid(X) - y), X)

    def compute_cost(self, X, y):
        h = self.sigmoid(X)
        sigma = np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return -sigma / X.shape[0]

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """

        X = preprocess(X)
        X = apply_bias_trick(X)
        return (self.sigmoid(X) > 0.5).astype(np.float64)


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
    X = np.column_stack((ones, X))
    return X


def preprocess(X):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    return (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
    # set random seed
    np.random.seed(random_state)

    shuffled_indices = np.random.permutation(len(X))

    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    X_folds = np.array_split(X_shuffled, folds)
    y_folds = np.array_split(y_shuffled, folds)

    accuracies = []
    for fold in range(folds):
        X_train = np.concatenate(X_folds[:fold] + X_folds[fold + 1:])
        y_train = np.concatenate(y_folds[:fold] + y_folds[fold + 1:])
        X_test = X_folds[fold]
        y_test = y_folds[fold]

        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)

    return np.mean(accuracies)


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    sqrt = sigma * np.sqrt(2 * np.pi)
    exp = np.exp(-np.square(data - mu) / (2 * np.square(sigma)))
    return 1 / sqrt * exp


# a vectorized version of the normal pdf function
vectorized_pdf = np.vectorize(norm_pdf)


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """

        self.weights = np.random.dirichlet(np.ones(self.k))
        self.mus = np.random.random(self.k)
        self.sigmas = np.random.random(self.k)

        self.responsibilities = []
        self.costs = []

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """

        likelihoods = self.weights * \
            vectorized_pdf(data, self.mus, self.sigmas)
        total_likelihoods = np.sum(likelihoods, axis=1, keepdims=True)
        self.responsibilities = likelihoods / total_likelihoods

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.weights = np.mean(self.responsibilities, axis=0)

        self.mus = np.sum(self.responsibilities * data,
                          axis=0) / (self.weights * data.shape[0])

        self.sigmas = np.sqrt(np.sum(self.responsibilities * np.square(
            (data - self.mus)), axis=0) / (self.weights * data.shape[0]))

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """

        self.init_params(data)
        for t in range(self.n_iter):

            self.costs.append(self.compute_cost(data))
            if t > 0 and np.abs(self.costs[t-1] - self.costs[t]) < self.eps:
                break

            self.expectation(data)
            self.maximization(data)

    def compute_cost(self, data):
        likelihoods = vectorized_pdf(data, self.mus, self.sigmas)
        return -np.sum(np.log(np.sum(self.weights * likelihoods, axis=1)))

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    return np.sum(weights * vectorized_pdf(data, mus, sigmas), axis=1)


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        X = preprocess(X)
        n_samples, n_features = X.shape
        self.classes, counts = np.unique(y, return_counts=True)
        self.priors = counts / n_samples
        self.gmm = []

        for label in self.classes:
            X_label = X[y == label]
            class_em_objects = []
            for feature in range(n_features):
                em_obj = EM(k=self.k, random_state=self.random_state)
                em_obj.fit(X_label[:, feature].reshape(-1, 1))
                class_em_objects.append(em_obj)
            self.gmm.append(class_em_objects)

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """

        X = preprocess(X)
        likelihoods = []

        for class_em_obj in self.gmm:
            class_probs = np.array([
                gmm_pdf(X[:, j].reshape(-1, 1), class_em_obj[j].weights,
                        class_em_obj[j].mus.flatten(), class_em_obj[j].sigmas.flatten())
                for j in range(X.shape[1])
            ])
            class_likelihood = np.prod(class_probs.squeeze().T, axis=1)
            likelihoods.append(class_likelihood)

        likelihoods = np.column_stack(likelihoods)
        posteriors = likelihoods * self.priors

        preds = np.where(
            posteriors[:, 0] > posteriors[:, 1], self.classes[0], self.classes[1])

        return preds


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    lor_predictions_x_train = lor.predict(x_train)
    lor_predictions_x_test = lor.predict(x_test)
    lor_train_acc = np.mean(lor_predictions_x_train == y_train)
    lor_test_acc = np.mean(lor_predictions_x_test == y_test)

    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    nb_predictions_x_train = naive_bayes.predict(x_train)
    nb_predictions_x_test = naive_bayes.predict(x_test)
    bayes_train_acc = np.mean(nb_predictions_x_train == y_train)
    bayes_test_acc = np.mean(nb_predictions_x_test == y_test)

    plot_decision_regions(x_train, y_train, lor, title="LOR")
    plot_decision_regions(x_train, y_train, naive_bayes, title="NB")

    plot_costs(lor.Js)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()


def plot_costs(costs):
    import matplotlib.pyplot as plt
    # Generate x-axis values (iteration numbers)
    iterations = range(1, len(costs) + 1)

    # Plot the costs against iterations
    plt.plot(iterations, costs)

    # Set labels for x-axis and y-axis
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost')

    # Add a title to the plot
    plt.title('Cost vs Iteration')

    # Display the plot
    plt.show()


def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    num_samples = 1000

    # dataset a
    mean_class1_a = [0, 0, 0]
    cov_class1_a = np.diag([50, 0.5, 50])

    mean_class2_a = [2, 2, 2]
    cov_class2_a = np.diag([0.5, 81, 8])

    # Generate samples for classes
    samples_class1 = multivariate_normal.rvs(
        mean=mean_class1_a, cov=cov_class1_a, size=num_samples)
    samples_class2 = multivariate_normal.rvs(
        mean=mean_class2_a, cov=cov_class2_a, size=num_samples)

    # Concatenate the samples and labels for the dataset
    dataset_a_features = np.concatenate(
        (samples_class1, samples_class2)).reshape(-1, 3)
    dataset_a_labels = np.concatenate(
        (np.zeros(num_samples), np.ones(num_samples)))

    # Shuffle the dataset and labels in the same order
    shuffle_indices = np.random.permutation(len(dataset_a_features))
    dataset_a_features = dataset_a_features[shuffle_indices]
    dataset_a_labels = dataset_a_labels[shuffle_indices]

   ####################################################################################

    # dataset b
    mean_class1_b = [3, 3, 3]
    cov_class1_b = np.diag([1, 1, 1])

    mean_class2_b = [2, 2, 3]
    cov_class2_b = np.diag([1, 1, 1])

    # Generate samples for classes
    samples_class1 = multivariate_normal.rvs(
        mean=mean_class1_b, cov=cov_class1_b, size=num_samples)
    samples_class2 = multivariate_normal.rvs(
        mean=mean_class2_b, cov=cov_class2_b, size=num_samples)
    samples_class2[:, 2] = samples_class2[:, 0] ** 2
    samples_class2[:, 1] = samples_class2[:, 2] ** 2

    # Concatenate the samples and labels for the dataset
    dataset_b_features = np.concatenate(
        (samples_class1, samples_class2)).reshape(-1, 3)
    dataset_b_labels = np.concatenate(
        (np.zeros(num_samples), np.ones(num_samples)))

    # Shuffle the dataset and labels in the same order
    shuffle_indices = np.random.permutation(len(dataset_b_features))
    dataset_b_features = dataset_b_features[shuffle_indices]
    dataset_b_labels = dataset_b_labels[shuffle_indices]

    plot_3d(dataset_a_features, dataset_a_labels, "a")
    plot_3d(dataset_b_features, dataset_b_labels, "b")

    return {'dataset_a_features': dataset_a_features,
            'dataset_a_labels': dataset_a_labels,
            'dataset_b_features': dataset_b_features,
            'dataset_b_labels': dataset_b_labels
            }


def plot_3d(features, labels, set):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot class 1 samples
    class1_indices = np.where(labels == 0)[0]
    ax.scatter(features[class1_indices, 0], features[class1_indices, 1],
               features[class1_indices, 2], c='r', label='Class 1')

    # Plot class 2 samples
    class2_indices = np.where(labels == 1)[0]
    ax.scatter(features[class2_indices, 0], features[class2_indices, 1],
               features[class2_indices, 2], c='b', label='Class 2')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title(f'Dataset {set}: 3D Visualization')
    ax.legend()
    plt.show()
