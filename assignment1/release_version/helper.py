import matplotlib.pyplot as plt

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1


def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################
    k_values = [0.001, 0.1, 0.5, 1, 2, 5, 10]
    
    results = []
    accuracies = []
    f1_scores = []

    for k in k_values:
        # train naive bayes with current k value
        nb = NaiveBayes.train(train_data, k=k)
        #evaluate on test data
        acc = accuracy(nb, test_data)
        f1 = f_1(nb, test_data)
        results.append((k, acc, f1))
        accuracies.append(acc)
        f1_scores.append(f1)
        
    # create graph
    plt.figure(figsize=(10, 6))
    
    # plotting accuracy
    plt.plot(k_values, accuracies, marker='o', linewidth=1.5, markersize=5, label='Accuracy', color='green')
    
    # plotting f1 score
    plt.plot(k_values, f1_scores, marker='s', linewidth=1.5, markersize=5, label='f1 Score', color='orange')
    
    # titles and other custumizations
    plt.xlabel('Smoothing Parameter (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Influence of Smoothing Parameter (k) on Naive Bayes Performance', 
              fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim([0, 1])

    # save fig
    plt.tight_layout()
    plt.savefig('test_smooth.png', dpi=300, bbox_inches='tight')
    plt.close()
    ####################################################################



def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################
    '''Feature 1: Removal of Stop Words'''
    nb_feature1 = features1(train_data, k=1)
    acc_feature1 = accuracy(nb_feature1, test_data)
    f1_feature1 = f_1(nb_feature1, test_data)
    print(f"Feature 1: Removal of Stop Words \n Accuracy: {acc_feature1:.4f},  f1 Score: {f1_feature1:.4f}")
    

    '''Feature 2: Bigrams'''
    nb_feature2 = features2(train_data, k=1)
    acc_feature2 = accuracy(nb_feature2, test_data)
    f1_feature2 = f_1(nb_feature2, test_data)
    print(f"Feature 2: Bigrams \n Accuracy: {acc_feature2:.4f}, f1 Score: {f1_feature2:.4f}")
    
    #####################################################################



def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################
    X_train, Y_train = featurize(train_data)
    X_test, Y_test = featurize(test_data, train_data=train_data)

    logreg = LogReg(eta=0.01, num_iter=10)
    logreg.train(X_train, Y_train)

    #for evaluation a string format is required therefore its needed to restructure the test data
    test_examples = []
    for x_row, y_row in zip(X_test, Y_test):

        # Convert one-hot → string
        if y_row[0] == 1:
            true_label = "offensive"
        else:
            true_label = "nonoffensive"

        test_examples.append((x_row, true_label))

    acc = accuracy(logreg, test_examples)
    f1  = f_1(logreg, test_examples)

    print(f"Accuracy: {acc:.4f}")
    print(f"f1 Score: {f1:.4f}")
    #####################################################################
