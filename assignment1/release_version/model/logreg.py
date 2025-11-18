import numpy as np


class LogReg:
    def __init__(self, eta=0.01, num_iter=30, C=0.1):
        self.eta = eta          # learning rate
        self.num_iter = num_iter
        self.C = C              # L2 regularization strength
        self.W = None           # weight matrix (F x 2)

    def softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs:
        :return:
        """
        # TODO: adapt for your solution
        scores = np.asarray(inputs)
        # make values smaller to avoid overflow
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def train(self, X, Y):

        #################### STUDENT SOLUTION ###################
        X = np.asarray(X, dtype=np.float32)   # N x F
        Y = np.asarray(Y, dtype=np.float32)   # N x 2

        N, F = X.shape
        _, C = Y.shape  # C should be 2

        # initialize weights once: F x C
        if self.W is None:
            self.W = np.zeros((F, C), dtype=np.float32)

        for it in range(self.num_iter):
            # forward pass
            scores = X @ self.W          # N x C
            probs  = self.softmax(scores)

            # gradient of loss
            grad = X.T @ (probs - Y) / N   # F x C

            # add L2 term
            grad += (1.0 / self.C) * self.W

            # update weights
            self.W -= self.eta * grad

        return None
        #########################################################


    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        X = np.asarray(X, dtype=np.float32)

        # if X is a single feature vector, make it 2D: (1, F)
        if X.ndim == 1:
            X = X[None, :]

        scores = X @ self.W    # (N x F) @ (F x 2) = (N x 2)
        return self.softmax(scores) #activation function
        ############################################################


    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        X = np.asarray(X, dtype=np.float32)
        single = False

        # handle single example, reshape to (1, F)
        if X.ndim == 1:
            single = True
            X = X[None, :]

        probs = self.p(X)                     # N x 2
        class_idx = np.argmax(probs, axis=1)  # N , getting the most likely class

        idx2label = {0: "offensive", 1: "nonoffensive"}

        # list of strings
        labels = [idx2label[int(i)] for i in class_idx]

        if len(labels) == 1:
            return labels[0]
        else:
            return labels
        #############################################################


def buildw2i(vocab):
    """
    Create indexes for 'featurize()' function.

    Args:
        vocab: vocabulary constructed from the training set.

    Returns:
        Dictionaries with word as the key and index as its value.
    """
    # YOUR CODE HERE
    #################### STUDENT SOLUTION ######################
    # dictionary mapping each word to an index
    w2i = {}
    for i, word in enumerate(sorted(vocab)):
        w2i[word] = i
    return w2i
    ############################################################


def featurize(data, train_data=None):
    """
    Convert data into X and Y where X is the input and
    Y is the label.

    Args:
        data: Training or test data.
        train_data: Reference data to build vocabulary from.

    Returns:
        Matrix X and Y.
    """
    # YOUR CODE HERE
    ##################### STUDENT SOLUTION #######################
     # build vocabulary from train_data if not None
    if train_data is not None:
        corpus = train_data
    else:
        corpus = data

    # build vocab
    vocab = set()
    for tokens, label in corpus:
        vocab.update(tokens)

    # word-to-index mapping
    w2i = buildw2i(vocab)
    vocab_size = len(w2i)
    N = len(data)

    # create empty feature matrix X and Y
    X = np.zeros((N, vocab_size), dtype=np.float32)
    Y = np.zeros((N, 2), dtype=np.float32)

    # with bag-of-words filling up feature matrices X and Y
    for i, (tokens, label) in enumerate(data):
        # set X[i] to 1 where a word appears in the tweet
        for word in tokens:
            if word in w2i:
                X[i, w2i[word]] = 1.0

        # with one-hot encoding: [1,0] offensive, [0,1] nonoffensive
        if label == "offensive":
            Y[i, 0] = 1.0
            Y[i, 1] = 0.0
        else: 
            Y[i, 0] = 0.0
            Y[i, 1] = 1.0

    return X, Y
    ##############################################################

