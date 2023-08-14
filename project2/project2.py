import numpy as np

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
             0.25: 1.32,
             0.1: 2.71,
             0.05: 3.84,
             0.0001: 100000},
             2: {0.5: 1.39,
             0.25: 2.77,
             0.1: 4.60,
             0.05: 5.99,
             0.0001: 100000},
             3: {0.5: 2.37,
             0.25: 4.11,
             0.1: 6.25,
             0.05: 7.82,
             0.0001: 100000},
             4: {0.5: 3.36,
             0.25: 5.38,
             0.1: 7.78,
             0.05: 9.49,
             0.0001: 100000},
             5: {0.5: 4.35,
             0.25: 6.63,
             0.1: 9.24,
             0.05: 11.07,
             0.0001: 100000},
             6: {0.5: 5.35,
             0.25: 7.84,
             0.1: 10.64,
             0.05: 12.59,
             0.0001: 100000},
             7: {0.5: 6.35,
             0.25: 9.04,
             0.1: 12.01,
             0.05: 14.07,
             0.0001: 100000},
             8: {0.5: 7.34,
             0.25: 10.22,
             0.1: 13.36,
             0.05: 15.51,
             0.0001: 100000},
             9: {0.5: 8.34,
             0.25: 11.39,
             0.1: 14.68,
             0.05: 16.92,
             0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    _, counts = np.unique(
        data[:, -1], return_counts=True)  # array with the counts of all unique labels
    label_probs = counts / data.shape[0]
    gini = 1 - np.sum(label_probs ** 2)

    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """

    _, counts = np.unique(data[:, -1], return_counts=True)
    label_probs = counts / data.shape[0]
    entropy = -np.sum(label_probs * np.log2(label_probs))

    return entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    # groups[feature_value] = data_subset
    groups = {feature_value: data[data[:, feature] == feature_value]
              for feature_value in np.unique(data[:, feature])}
    split_info = 0

    for feature_value in groups:
        p = groups[feature_value].shape[0] / data.shape[0]
        goodness += p * impurity_func(groups[feature_value])    # Info gain
        split_info -= p * np.log2(p)    # Split information

    goodness = impurity_func(data) - goodness

    # Gain ratio
    if gain_ratio:
        goodness = goodness / split_info

    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """

        unique_labels, counts = np.unique(self.data[:, -1], return_counts=True)
        return unique_labels[np.argmax(counts)]

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """

        if len(np.unique(self.data[:, -1])) == 1 or self.depth >= self.max_depth:
            self.terminal = True
            return

        best_groups = {}
        best_gain = -1
        for feature in range(self.data.shape[1] - 1):

            if len(np.unique(self.data[:, feature])) == 1:
                continue

            gain, groups = goodness_of_split(
                self.data, feature, impurity_func, self.gain_ratio)
            if gain > best_gain:
                best_gain = gain
                best_groups = groups
                self.feature = feature

        if self.chi != 1 and self.chi_test_value(best_groups) < chi_table[len(best_groups) - 1][self.chi]:
            self.terminal = True
            return

        for feature_value in best_groups:
            new_child = DecisionNode(best_groups[feature_value], depth=self.depth + 1,
                                     chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(new_child, feature_value)

    def chi_test_value(self, groups):
        """ Calculate the chi test value of a node using the chi table

        Args:
            groups (dictionary): pairs of feature_value as keys and the data corrsponding to this feature value as value

        Returns:
            float: chi test value
        """
        chi_value = 0
        unique_labels, counts = np.unique(self.data[:, -1], return_counts=True)
        p_y_zero = counts[0] / self.data.shape[0]
        p_y_one = counts[1] / self.data.shape[0]

        for feature_value in groups:
            Df = groups[feature_value].shape[0]
            Pf = np.sum(groups[feature_value][:, -1] == unique_labels[0])
            Nf = np.sum(groups[feature_value][:, -1] == unique_labels[1])
            E_zero = Df * p_y_zero
            E_one = Df * p_y_one

            chi_value += ((Pf - E_zero) ** 2 / E_zero) + \
                ((Nf - E_one) ** 2 / E_one)

        return chi_value


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """

    root = DecisionNode(data, chi=chi, max_depth=max_depth,
                        gain_ratio=gain_ratio)
    recursive_build_tree(root, impurity)

    return root


def recursive_build_tree(node, impurity_func):
    """ Build a decision tree recursively

    Args:
        node (DecisionNode): a node in the tree
        impurity_func (func): the chosen impurity measure.
    """

    node.split(impurity_func)
    if node.terminal:
        return

    for child in node.children:
        recursive_build_tree(child, impurity_func)


def predict(root, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: a row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """

    curr_node = root
    while not curr_node.terminal:

        if instance[curr_node.feature] not in curr_node.children_values:
            break

        index = curr_node.children_values.index(instance[curr_node.feature])
        curr_node = curr_node.children[index]

    return curr_node.pred


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """

    true_pred = 0
    for instance in dataset:
        if predict(node, instance) == instance[-1]:
            true_pred += 1

    return true_pred / dataset.shape[0] * 100


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(X_train, calc_entropy, True, max_depth=max_depth)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))

    return training, testing


def chi_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []

    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = build_tree(X_train, calc_entropy, True, chi)
        chi_training_acc.append(calc_accuracy(tree, X_train))
        chi_testing_acc.append(calc_accuracy(tree, X_test))
        depth.append(get_tree_specs(tree)[0])

    return chi_training_acc, chi_testing_acc, depth


def get_tree_specs(node, number_of_nodes=0, memo={}):
    """ Gets the depth and the number of nodes of this tree, rooted at this node

    Args:
        node (DecisionNode): a decision node

    Returns:
        int: the depth of the tree rooted at this node
        int: the number of nodes rooted at this node
    """

    if node.terminal:
        return node.depth, number_of_nodes + 1

    if node in memo:
        return memo[node]

    depth = max(get_tree_specs(child, memo=memo)[0] for child in node.children)
    nodes = sum(get_tree_specs(child, number_of_nodes, memo)
                [1] for child in node.children) + 1

    memo[node] = depth, nodes

    return depth, nodes


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of nodes in the tree.
    """
    return get_tree_specs(node)[1]
