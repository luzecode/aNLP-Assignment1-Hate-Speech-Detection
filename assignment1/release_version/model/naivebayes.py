import math


class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    def __init__(self, log_priors, log_likelihoods, vocab):
        """Initialises a new classifier.
        Args:
            log_priors: Dictionary with the log probability of each class
            log_likelihoods: Dictionary with the log probability of words given aclass
            vocab: Set of all unique words
        """
        self.log_priors = log_priors 
        self.log_likelihoods = log_likelihoods
        self.vocab = vocab
    ####################################################################


    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################
        # YOUR CODE HERE
        output = {}
        for class_label in self.log_priors:
            #get the log prior for each class label
            output[class_label] = self.log_priors[class_label]
            for word in x:
                #get and add the log likelihood for each word based on the current class
                if word in self.log_likelihoods[class_label]:
                    output[class_label] += self.log_likelihoods[class_label][word]
        

        # return the class that has the highest probability
        return max(output, key=output.get)
    
        ############################################################


    @classmethod
    def train(cls, data, k=1):
        """Train a new classifier on training data using maximum
        likelihood estimation and additive smoothing.

        Args:
            cls: The Python class representing the classifier.
            data: Training data.
            k: The smoothing constant.

        Returns:
            A trained classifier, an instance of `cls`.
        """
        ##################### STUDENT SOLUTION #####################
        # YOUR CODE HERE
        class_count = {}
        word_counts = {}

        #keepign track of amount of classes and total amount of words.
        for tokens, label in data:
            if label not in class_count:
                class_count[label] = 0
            class_count[label] += 1
            
            if label not in word_counts:
                word_counts[label] = {}
            
            for word in tokens:
                if word not in word_counts[label]:
                    word_counts[label][word] = 0
                word_counts[label][word] += 1
        
        #set up vocabulary
        vocab = set()
        for tokens, label in data:
            vocab.update(tokens)
        
        vocab_size = len(vocab)
        total_documents = len(data)

        #calculatign the log priors 
        log_priors = {}
        for class_label in class_count:
            log_priors[class_label] = math.log(class_count[class_label] / total_documents)
        
        #calculating the log likelihood
        log_likelihoods = {}
        for class_label in class_count:
            log_likelihoods[class_label] = {}
            total_words = sum(word_counts[class_label].values())

            for word in vocab:
                word_count = word_counts[class_label].get(word, 0)

                #additive smoothing formula
                likelihood = (word_count + k) / (total_words + k * vocab_size)
                log_likelihoods[class_label][word] = math.log(likelihood)
        
        return cls(log_priors, log_likelihoods, vocab)
        ############################################################



def features1(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    """removing stop words from a document to filter out "unimportant" words
    to have a cleaner result"""
    #manually set stop words to filter out
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was'}
    
    filtered_data = []
    for tokens, label in data:
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        if not filtered_tokens:
            filtered_tokens = ['<EMPTY>']
        filtered_data.append((filtered_tokens, label))
    
    return NaiveBayes.train(filtered_data, k)
    ##################################################################


def features2(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    """
    using bigrams to capture word order and context
    """
    bigram_data = []
    for tokens, label in data:
        enhanced_tokens = list(tokens)
        
        #add beginning and end of sentence marker
        tokens_with_markers = ['<START>'] + tokens + ['<END>']
        
        # add bigrams with sentence boundary markers
        for i in range(len(tokens_with_markers) - 1):
            bigram = tokens_with_markers[i] + "_" + tokens_with_markers[i + 1]
            enhanced_tokens.append(bigram)
        
        bigram_data.append((enhanced_tokens, label))
    
    return NaiveBayes.train(bigram_data, k)
    ##################################################################

