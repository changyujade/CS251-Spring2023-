'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
YOUR NAME HERE
CS 251/2: Data Analysis Visualization
Spring 2023
'''
import math
import numpy as np
from sklearn import metrics


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''

    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`.
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        '''

        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham

        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c

        self.class_priors = None

        self.class_likelihoods = None

        self.num_classes = num_classes

    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''
        return self.class_priors

    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''
        return self.class_likelihoods

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape
        label, num = np.unique(y, return_counts=True)

        self.class_priors = num / len(y)  # num_of_spam_or_ham/total_num_emails

        self.class_likelihoods = []
        for i in label:
            indices = np.where(y == i)
            row = data[indices]
            ncw = np.sum(row, axis=0)
            nc = np.sum(ncw)
            self.class_likelihoods.append((ncw + 1)/(nc + num_features))

        self.class_likelihoods = np.array(self.class_likelihoods)

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: this can also be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''

        num_test_samps, num_features = data.shape

        # 1 summation over likelihoods and addition to the prior

        likelihoods = self.get_likelihoods()
        priors = self.get_priors()
        posterior = np.log(priors).reshape(priors.shape[0],1)+np.log(likelihoods)@data.T

        # 2 calling argmax to find the index of the largest post
        return np.argmax(posterior, axis = 0)



    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        assert len(y) == len(y_pred)
        diffs = y == y_pred  # returns boolean
        # return number of trues and falses
        vals, counts = np.unique(diffs, return_counts=True)
        # count the correctly predicted values
        result = counts[np.where(vals == True)] / (np.sum(counts))

        return result[0]

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        num_class = np.unique(y)
        C = np.zeros((len(num_class),len(num_class)))
        for index in range(len(y)):
            actual = y[index]
            yhat = y_pred[index]
            C[actual,yhat] += 1

        return C
