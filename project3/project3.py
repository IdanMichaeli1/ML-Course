import numpy as np


class conditional_independence():

    def __init__(self):

        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.6
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.25,
            (0, 1): 0.05,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.25,
            (0, 1): 0.05,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.125,
            (0, 0, 1): 0.005,
            (0, 1, 0): 0.125,
            (0, 1, 1): 0.045,
            (1, 0, 0): 0.125,
            (1, 0, 1): 0.045,
            (1, 1, 0): 0.125,
            (1, 1, 1): 0.405
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        X_values = np.array(list(X.values()))
        Y_values = np.array(list(Y.values()))
        X_Y_values = np.array(list(X_Y.values()))
        expected_X_Y = np.outer(X_values, Y_values).flatten()

        return not np.all(np.isclose(X_Y_values, expected_X_Y))

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        C_values = np.array(list(C.values()))
        X_C_values = np.array(list(X_C.values()))
        Y_C_values = np.array(list(Y_C.values()))
        X_Y_C_values = np.array(list(X_Y_C.values()))

        expected_X_Y_C = (X_C_values.reshape(-1, 1, 2) *
                          Y_C_values.reshape(1, -1, 2)) / C_values.reshape(1, 1, -1)
        expected_X_Y_C = expected_X_Y_C.flatten()

        return np.all(np.isclose(X_Y_C_values, expected_X_Y_C))


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """

    factorial = np.array([np.math.factorial(x)
                         for x in k]) if isinstance(k, np.ndarray) else k
    pmf = np.power(rate, k) * np.exp(-rate) / factorial
    return np.log(pmf)


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """

    likelihoods = np.zeros(len(rates))
    for i, r in enumerate(rates):
        likelihoods[i] = np.sum(poisson_log_pmf(samples, r))

    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates)
    return rates[np.where(likelihoods == np.max(likelihoods))][0]


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    return np.mean(samples)


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """

    sqrt = np.sqrt(2 * np.pi * np.square(std))
    exp = np.exp(-np.square(x - mean) / (2 * np.square(std)))
    return 1 / sqrt * exp


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        class_samples = dataset[dataset[:, -1] == class_value]
        self.num_total_samples = len(dataset)
        self.num_class_samples = len(class_samples)

        self.means = np.mean(class_samples[:, :-1], axis=0)
        self.stds = np.std(class_samples[:, :-1], axis=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.num_class_samples / self.num_total_samples

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return np.prod(normal_pdf(x, self.means, self.stds))

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        return 0 if self.ccd0.get_instance_posterior(x[:-1]) > self.ccd1.get_instance_posterior(x[:-1]) else 1


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.

    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    correct = 0
    for x in test_set:
        correct += map_classifier.predict(x) == x[-1]

    return correct / len(test_set)


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """

    left = np.power(2 * np.pi, -len(x) / 2) * \
        np.power(np.linalg.det(cov), -0.5)
    right = np.exp(-0.5 * np.matmul(np.matmul(np.transpose((x - mean)),
                   np.linalg.inv(cov)), (x - mean)))
    pdf = left * right

    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """

        class_samples = dataset[dataset[:, -1] == class_value]
        self.num_total_samples = len(dataset)
        self.num_class_samples = len(class_samples)

        self.means = np.mean(class_samples[:, :-1], axis=0)
        self.cov = np.cov(class_samples[:, :-1], rowvar=False)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.num_class_samples / self.num_total_samples

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.means, self.cov)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """

        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the prior probability of class 0 is higher and 1 otherwise.
        """
        return 0 if self.ccd0.get_prior() > self.ccd1.get_prior() else 1


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the likelihood probability of class 0 is higher and 1 otherwise.
        """
        return 0 if self.ccd0.get_instance_likelihood(x[:-1]) > self.ccd1.get_instance_likelihood(x[:-1]) else 1


# if a certain value only occurs in the test set, the probability for that value will be EPSILLON.
EPSILLON = 1e-6


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.class_samples = dataset[dataset[:, -1] == class_value]
        self.dataset = dataset

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        return len(self.class_samples) / len(self.dataset)

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """

        likelihood = 1
        n_i = len(self.class_samples)
        for j in range(self.dataset.shape[1] - 1):
            unique_vals = np.unique(self.dataset[:, j])
            v_j = (len(unique_vals))
            n_i_j = len(self.class_samples[self.class_samples[:, j] == x[j]])

            likelihood *= (n_i_j + 1) / \
                (n_i + v_j) if x[j] in unique_vals else EPSILLON

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        return 0 if self.ccd0.get_instance_posterior(x[:-1]) > self.ccd1.get_instance_posterior(x[:-1]) else 1

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        correct = 0
        for x in test_set:
            correct += self.predict(x) == x[-1]

        return correct / len(test_set)
