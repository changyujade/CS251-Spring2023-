'''knn.py
K-Nearest Neighbors algorithm for classification
Ｊade Chang
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

        self.num_classes = num_classes

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data
        self.classes = y

    def distanceMetric(self, element, datapoint):
        return np.sqrt(np.dot((element-datapoint).T, element-datapoint))

    def predictOne(self,element,k):
        neighborsByDistance = np.argpartition([self.distanceMetric(element, datapoint) for datapoint in self.exemplars], k) # finds the index of the neighbours, sorted by distance
        # print([(x, self.distanceMetric(element, x)) for x in self.model[neighborsByDistance]])
        neighborLabels = [self.classes[neighborsByDistance[x]] for x in range(k)] # a list that stores the 'class' of the nearest training datapoints 
        # print(neighborLabels)
        vals, counts = np.unique(neighborLabels, return_counts=True) # number in each class
        # print(vals, counts)
        return vals[np.argwhere(counts == np.max(counts))][0,0] # the index of the class with the most numbers
        # returns the class of the testdata after voting

    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        return np.array([self.predictOne(element, k) for element in data])

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
        diffs = y == y_pred #returns boolean
        vals, counts = np.unique(diffs, return_counts=True) #return number of trues and falses
        result = counts[np.where(vals == True)] / (np.sum(counts)) #count the correctly predicted values
        
        return result[0]

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use a ColorBrewer color palette. List of possible ones here:
        https://github.com/CartoDB/CartoColor/wiki/CARTOColor-Scheme-Names
            - An example: cartocolors.qualitative.Safe_4.mpl_colors
            - The 4 stands for the number of colors in the palette. For simplicity, you can assume
            that we're hard coding this at 4 for 4 classes.
        - Each ColorBrewer palette is a Python list. Wrap this in a `ListedColormap` object so that
        matplotlib can parse it (already imported above).
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        cmap = ListedColormap(cartocolors.qualitative.Safe_4.mpl_colors)

        samp_vec = np.arange(self.exemplars.min(),self.exemplars.max(),
                             (self.exemplars.max()-self.exemplars.min())/n_sample_pts)
        x, y = np.meshgrid(samp_vec, samp_vec)
        
        x = x.reshape(n_sample_pts*n_sample_pts,1)
        y = y.reshape(n_sample_pts*n_sample_pts,1)
   
        fake_data = np.concatenate((x,y), axis = 1)
        print(fake_data.shape)

        y_pred = self.predict(fake_data,k)
        y_pred = y_pred.reshape(n_sample_pts, n_sample_pts)

        plt.pcolormesh(samp_vec,samp_vec,y_pred,cmap = cmap, shading = 'auto')
        plt.colorbar()
    
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
