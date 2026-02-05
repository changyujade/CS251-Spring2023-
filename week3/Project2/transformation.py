'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Jade Chang
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data
from scipy import stats as sc


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        self.orig_dataset = orig_dataset
        if data == None:
            self.data = orig_dataset
        else:
            super().__init__(data)

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''

        orig_dataset = self.orig_dataset

        filepath = orig_dataset.filepath
        selected_data = orig_dataset.select_data(headers)

        header2col = {}

        counter = 0
        for i in headers:
            header2col[i] = counter
            counter += 1
        # create a new Data object d3
        d3 = data.Data(filepath=None, headers=headers,
                       data=selected_data, header2col=header2col)

        self.data = d3

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        d2 = self.data.data.copy()
        Dh = np.hstack((d2, np.ones([d2.data.shape[0], 1])))
        return Dh

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''

        d2 = self.data.data.copy()
        T = np.eye(d2.data.shape[1]+1)
        T[:, -1] = np.array(magnitudes)
        return T

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''

        d2 = self.data.data.copy()
        S = np.eye(d2.data.shape[1]+1)
        # row, col = np.diag_indices_from(S)
        # S[row,col] = np.array(magnitudes)
        np.fill_diagonal(S, np.array(magnitudes))
        return S

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        T = self.translation_matrix(magnitudes)
        d2 = self.get_data_homogeneous()
        DhT = d2.T
        result = T@DhT
        result = result[:-1, :]
        result = result.T
        self.data.data = result
        return result

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''

        S = self.scale_matrix(magnitudes)
        d2 = self.get_data_homogeneous()
        DhT = d2.T
        result = S@DhT
        result = result[:-1, :]
        result = result.T
        self.data.data = result
        return result

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        result = (C@self.get_data_homogeneous().T).T[:, :-1]
        self.data.data = result

        return result

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        d2 = self.data.data.copy()
        mn = d2.min()
        mx = d2.max()
        d2 = self.get_data_homogeneous()

        shifts = np.empty(d2.shape[1])
        shifts.fill(-mn)
        T = self.translation_matrix(shifts)

        shifts = np.empty(d2.shape[1])
        scale = 1/(mx-mn)
        shifts.fill(scale)
        S = self.scale_matrix(scale)

        result = (S@T@d2.T).T[:, :-1]
        self.data.data = result

        return result

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        D = self.data.data.copy()
        Dh = self.get_data_homogeneous()

        mins = D.min(axis=0)
        maxes = D.max(axis=0)
        scale = 1/(maxes-mins)

        mins = np.append(-mins, 1)

        T = self.translation_matrix(mins)

        scale = np.append(scale, 1)
        S = self.scale_matrix(scale)

        result = (S@T@Dh.T).T[:, :D.shape[1]]
        # result = result[:D.shape[1],:]
        # result = result.T
        self.data.data = result

        return result

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        index = self.data.get_header_indices(header)
        radian = np.deg2rad(degrees)
        R = np.eye(4)

        if index[0] == 0:
            # rotate on x (first axis)
            R[1, 1] = np.cos(radian)
            R[1, 2] = -np.sin(radian)
            R[2, 1] = np.sin(radian)
            R[2, 2] = np.cos(radian)

        elif index[0] == 1:

            # rotate around y axis (second axis)
            R[0, 0] = np.cos(radian)
            R[0, 2] = np.sin(radian)
            R[2, 0] = -np.sin(radian)
            R[2, 2] = np.cos(radian)

        else:
            # rotate around z axis (third axis)
            R[0, 0] = np.cos(radian)
            R[0, 1] = -np.sin(radian)
            R[1, 0] = np.sin(radian)
            R[1, 1] = np.cos(radian)

        return R

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        R = self.rotation_matrix_3d(header, degrees)

        d2 = self.get_data_homogeneous()
        Dh = d2.T

        result = R@Dh

        result = result[:-1, :]
        result = result.T
        self.data.data = result
        return result

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        x = self.data.select_data(ind_var)
        y = self.data.select_data(dep_var)
        z = self.data.select_data(c_var)
        color_map = palettable.colorbrewer.sequential.Purples_9
        sc = plt.scatter(
            x, y, c=z, s=75, cmap=color_map.mpl_colormap, edgecolor='black')
        plt.colorbar(sc)
        plt.show()

    # Functions below are for extensions

    def z_score_scipy(self):
        """
        normalise on columns
        """
        D = self.data.data.copy()

        z_scores = sc.zscore(D)
        self.data.data = z_scores
        return z_scores

    def z_score(self):
        D = self.data.data.copy()
        mean = np.mean(D)
        std = np.std(D)
        new_D = (D - mean)/std
        self.data.data = new_D
        return new_D

    def z_score_separately(self):
        D = self.data.data.copy()
        mean = D.mean(axis=0)
        std = np.std(D, axis=0)
        new_D = (D - mean)/std
        self.data.data = new_D

        return new_D

    def normalize_broadcast(self):
        d = self.data.data.copy()
        mn = d.min()
        mx = d.max()
        d = d-mn
        d = d*(1/(mx-mn))
        self.data.data = d
        return d

    def normalize_separately_broadcast(self):
        d = self.data.data.copy()
        mn = d.min(axis=0)
        mx = d.max(axis=0)
        d = d-mn
        d = d*(1/(mx-mn))
        self.data.data = d
        return d

    def rotation_matrix_2d(self, degrees):
        '''Make an 2-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
    
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(3, 3). The 2D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        
        radian = np.deg2rad(degrees)
        R = np.eye(3)

        # rotate on x (first axis)
        R[0, 0] = np.cos(radian)
        R[0, 1] = -np.sin(radian)
        R[1, 0] = np.sin(radian)
        R[1, 1] = np.cos(radian)

        return R

    def rotate_2d(self, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        R = self.rotation_matrix_2d(degrees)

        d2 = self.get_data_homogeneous()
        Dh = d2.T

        result = R@Dh

        result = result[:-1, :]
        result = result.T
        self.data.data = result
        return result
