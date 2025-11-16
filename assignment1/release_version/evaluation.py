def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    correct = 0
    total = 0

    for tweet, true_label in data:
        pred = classifier.predict(tweet)
        if pred == true_label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0
    ################################################################



def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    #collect all classes appearing in data
    classes = set()
    for tweet, label in data:
        classes.add(label)
    
    #list storing f1 score for each class
    f1_scores = []
    
    #going through the found classes and set them as the poitive class one by one
    for positive_class in classes:
        tp = 0
        fp = 0
        fn = 0
        for tweet, label in data:
            predicted_label = classifier.predict(tweet)
            
            if predicted_label == positive_class:
                if predicted_label == label:
                    tp += 1
                else:
                    fp += 1
            else:
                if label == positive_class:
                    fn += 1
        
        #calculate precision and recall for the current positive class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1_class = 2 * precision * recall / (precision + recall)
        else:
            f1_class = 0.0
        
        #keeping track of all f1 scores per class
        f1_scores.append(f1_class)
    
    # returning the macro f1 (average of all class f1 scores)
    macro_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0.0
    return macro_f1
    ################################################################
