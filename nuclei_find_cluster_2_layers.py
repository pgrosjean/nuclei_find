# Nuclei Marker Script
# Parker Grosjean
# 9/26/2019


# Importing neccessary dependencies
import os

import czifile as czi
import numpy as np
import pandas as pd
import skimage
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import square
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import morphology

# Generating list of images for passing to constructor
im_list = []
for file in os.listdir():
    if file.endswith('.czi') or file.endswith('.jpg'):
        im_list.append(file)

# Generating list of csv files for nuclei clustering
csv_list = []
for file in os.listdir():
    if file.endswith('spot.csv'):
        csv_list.append(file)

# Sorting lists to ensure the images are associated with their respective csv files
csv_list = sorted(csv_list)
im_list = sorted(im_list)
file_list = zip(im_list, csv_list)


class czi_image:
    def __init__(self, file, csv_file, nuclei_channel=1):
        self.raw_im = np.squeeze(czi.imread(file))[nuclei_channel]
        self.labeled_spots_im = np.squeeze(czi.imread(file))[0]
        self.mask_1 = cv2.threshold(self.raw_im, skimage.filters.threshold_otsu(self.raw_im)+152, 1, cv2.THRESH_BINARY)[1]
        self.nuclei_image = skimage.exposure.equalize_hist(morphology.white_tophat(np.squeeze(czi.imread(file))[nuclei_channel], square(70)), mask=self.mask_1)
        self.corrected_ni = np.multiply(np.divide(self.nuclei_image, np.amax(self.nuclei_image)), 255)
        self.otsu = skimage.filters.threshold_otsu(self.corrected_ni)
        self.dtype = str(self.nuclei_image.dtype)
        self.nuclei_mask = cv2.threshold(self.corrected_ni, self.otsu, 1, cv2.THRESH_BINARY)[1]
        self.nuclei_mask_dense = cv2.threshold(self.corrected_ni, self.otsu+80, 1, cv2.THRESH_BINARY)[1]
        self.nuc_chan = nuclei_channel
        self.input_shape = self.nuclei_image.shape
        self.filename = file
        self.csv_file = csv_file
        self.spatial_df = pd.read_csv(csv_file)
        print('csv: \n', self.csv_file)
        print('image: \n', self.filename)

    def show(self):
        '''
        This function is used to show the input image and its corresponding
        image tranformations that are used in the algorithm.

        Args: None

        Returns: None (Only produces instances of matplotlib figures)
        '''
        print('self.nuclei_image')
        print('Max of self.nuclei_image:', np.amax(self.nuclei_image))
        print('Min of self.nuclei_image:', np.amin(self.nuclei_image))
        plt.imshow(self.nuclei_image)
        plt.show()
        print('self.corrected_ni')
        print('Max of self.corrected_ni:', np.amax(self.corrected_ni))
        print('Min of self.corrected_ni:', np.amin(self.corrected_ni))
        plt.imshow(self.corrected_ni)
        plt.show()
        print('Otsu Value:', self.otsu)
        print('self.nuclei_mask')
        plt.imshow(self.nuclei_mask)
        plt.show()
        # print('Distance Transform')
        # plt.imshow(ndimage.distance_transform_edt(self.nuclei_mask))
        # plt.show()
        print('Histogram', np.histogram(skimage.filters.gaussian(self.corrected_ni), bins=np.arange(256))[0])
        plt.hist(np.histogram(skimage.filters.gaussian(self.corrected_ni), bins=np.arange(255)))
        plt.show()

    def cell_markers(self):
        '''
        This fuction locates nuclei from the input nuclei images and classifies
        nuclei as either dense or sparse based upon a connected components analysis.
        The centroids of the sparse and dense nuclei are returned as lists of centroid
        indices.

        This function begins by initializing the nuclei mask generated by the constructor.
        A connected components transform is then used to classify nuclei in the mask
        as either sparse (less than 50 pixels) or dense (greater than or equalt to 50 pixels).
        The sparse pixels are then initialized into a new numpy array (binary mask). This dense mask
        is distance transformed (where the pixel values in the new array correspond to the
        distance of a signal pixel (1 or True) to a background signal (0 or False)). The peak_local_max
        algorithm is then used to find the peaks and initialize a new peak binary mask.
        Then the center_of_mass of each peak is found and used as the index for the correspondng nucleus.
        A binary mask of the dense pixels is then generated and multiplied by the original input
        nuclei image, which is then thresholded at an intensity of 80 above the otsu threshold
        used to generate the original nuclei mask. This leaves only higher intensity signal of the
        dense regions of tissue so the centroids of nuclei can be calculated using the distance transform.
        The resulting binary mask is then distance transformed. The peaks are then found and the center
        of masses returned the same as for the sparse nuclei.

        Args:
        None

        Returns:
        list_coms: list of the form [coms_sparse, coms_dense]
        where coms_sparse and coms_dense are lists of tuples where the tuples
        are of the form (row, column).
        '''
        print('Sparse Pass... \n')
        print('Distnace Transorming... \n')
        dist_transform = ndimage.distance_transform_edt(self.nuclei_mask)
        print('Finding Local Peaks... \n')
        max_local_peaks = peak_local_max(dist_transform, min_distance=1, indices=False)
        print('Labeling Nuclei... \n')
        # First Pass (Sparse Nuclei)
        mark_blob, cc_blob_num = ndimage.label(self.nuclei_mask, structure=
        [[1,1,1],
         [1,1,1],
         [1,1,1]])
        cc_areas = ndimage.sum(self.nuclei_image, mark_blob, range(cc_blob_num+1))
        # print(cc_areas)
        # Sparse Pass
        area_mask_sparse = (cc_areas > 50)
        mark_blob_sparse = mark_blob.copy()
        mark_blob_sparse[area_mask_sparse[mark_blob_sparse]] = 0
        clipped_marks = np.clip(mark_blob_sparse,0,1)
        mlp_sparse = np.multiply(clipped_marks, max_local_peaks)
        markers = ndimage.label(mlp_sparse, structure=
        [[1,1,1],
         [1,1,1],
         [1,1,1]])[0]
        coms_sparse = ndimage.center_of_mass(mlp_sparse, markers, np.arange(np.amax(markers)))
        print('Area per nuclei Sparse: ', np.sum(np.sum(clipped_marks))/len(coms_sparse))
        print('Otsu\'s Threshold:', self.otsu)
        # Second Pass (Dense Pass)

        # Defining neccessary dense objects
        mark_blob_dense = mark_blob.copy()
        area_mask_dense = (cc_areas <= 50)
        mark_blob_dense[area_mask_dense[mark_blob_dense]] = 0
        clipped_marks_dense = np.clip(mark_blob_dense,0,1)
        nm_dense = np.multiply(clipped_marks_dense, self.nuclei_mask_dense) # defining dense nuclei in new mask

        print('Dense Pass... \n')
        print('Distance Transforming... \n')
        dist_transform_d = ndimage.distance_transform_edt(nm_dense)
        print('Finding Local Peaks... \n')
        mlp_dense = peak_local_max(dist_transform_d, min_distance=1, indices=False)
        print('Labeling Nuclei... \n')
        markers_dense = ndimage.label(mlp_dense, structure=
        [[1,1,1],
         [1,1,1],
         [1,1,1]])[0]
        coms_dense = ndimage.center_of_mass(mlp_dense, markers_dense, np.arange(np.amax(markers_dense)))
        print('Area per nuclei Dense: ', np.sum(np.sum(clipped_marks_dense))/len(coms_sparse))
        list_coms = [coms_sparse, coms_dense]
        return list_coms

    def save_nuclei_mask(self, file_name='DAPI_mask.tif'):
        '''
        Saves the nuclei mask used for the nuclei locating algorithm as a tif.

        Args:
        file_name (str): name of output tiff image (default as DAPI_mask.tif)

        Returns:
        None (saves tif image to working directory)
        '''
        print('Saving Nuclei Mask... \n')
        skimage.io.imsave('../nuclei_find/' + file_name, self.nuclei_mask)

    def nearest_spot_clustering(self, coms_list):
        '''
        Clusters nuclei based on input indices in the csv file used by the constructor.

        This function begins by reading in the 'spots' or indices that are used to
        cluster nuclei from the csv. Then, the xdim_scale and ydim_scale values are
        used to transform the spots with indices starting in the bottom left corner
        to numpy array indices. A nearest neighbor classifier is then trained with
        the csv spot indices. The centroids of the nuclei from the cell_marker function
        are then iterated thrrough and clustered to the the clostest spot index.
        Heat maps are then generated via matplotlib where (1) each centroid has a
        heat corresponding to the number of cells clustered to its respective spot
        index, (2) digital spots are generated and have a heat associated with the
        number of nuclei assoiated with that spot, and (3) the digital spots generated
        are projected onto a binary mask of the nuclei. This also generates a csv file
        with the the number of dense and sparse nucelei per each spot index and saves
        it to the working directory.

        Args:
        coms_list (list): List output from the cell_markers function

        Returns:
        None (only generates matplotlib heat maps and saves csv)

        '''
        hm_cmap = 'inferno' # defining color scheme for heat map
        print('Clustering Nuclei to Spots... \n')
        coms_sparse, coms_dense = coms_list
        spatial_locations = self.spatial_df
        cols = spatial_locations['col']
        rows = spatial_locations['row']
        hashs = spatial_locations['hash']
        xdim_scale = spatial_locations['xdim_scale'].to_numpy()[0]
        ydim_scale = spatial_locations['ydim_scale'].to_numpy()[0]
        x_unscaled = spatial_locations['coords.x1'].to_numpy()
        y_unscaled = spatial_locations['coords.x2'].to_numpy()
        x = np.round(np.subtract(self.input_shape[1], np.multiply((self.input_shape[1]/xdim_scale), np.round(spatial_locations['coords.x1'].to_numpy()))))
        y = np.round(np.subtract(self.input_shape[0], np.multiply((self.input_shape[0]/ydim_scale), np.round(spatial_locations['coords.x2'].to_numpy()))))
        samples = zip(y,x,rows,cols,hashs,x_unscaled,y_unscaled)
        array_for_fit = []
        meta_data = []
        for sample in samples:
            if sample[0] <= self.input_shape[0] and sample[1] <= self.input_shape[1] and sample[0] >= 0 and sample[1] >= 0:
                array_for_fit.append(list(sample)[0:2])
                meta_data.append(list(sample)[2:])
        array_for_fit = np.array(array_for_fit).astype('int')
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(array_for_fit,np.arange(len(array_for_fit)))
        dict_sparse_count = {}
        dict_sparse = {}
        dict_dense = {}
        dict_dense_count = {}
        print('Clustering Sparse Nuclei... \n')
        for item in tqdm(coms_sparse[1:]):
            spot = neigh.kneighbors([item], return_distance=False)
            spot = spot[0][0]
            if spot not in list(dict_sparse_count.keys()):
                dict_sparse_count[spot] = 1
                dict_sparse[spot] = [item]
            else:
                dict_sparse_count[spot] += 1
                dict_sparse[spot].append(item)
        print('Clustering Dense Nuclei... \n')
        for item in tqdm(coms_dense[1:]):
            spot = neigh.kneighbors([item], return_distance=False)
            spot = spot[0][0]
            if spot not in list(dict_dense_count.keys()):
                dict_dense_count[spot] = 1
                dict_dense[spot] = [item]
            else:
                dict_dense_count[spot] += 1
                dict_dense[spot].append(item)
        points_list = np.array(list(set(dict_sparse.keys()) | set(dict_dense.keys())))
        out_dict = {'row': [meta_data[point][0] for point in points_list]}
        out_dict['col'] = [meta_data[point][1] for point in points_list]
        out_dict['coords.x1'] = [meta_data[point][3] for point in points_list]
        out_dict['coords.x2'] = [meta_data[point][4] for point in points_list]
        out_dict['hash'] = [meta_data[point][2] for point in points_list]
        out_dict['Dense Nuclei Count'] = np.array([dict_dense_count.get(key, 0) for key in points_list])
        out_dict['Sparse Nuclei Count'] = np.array([dict_sparse_count.get(key, 0) for key in points_list])
        out_dict['Total Nuclei Count'] = np.add(out_dict['Dense Nuclei Count'], out_dict['Sparse Nuclei Count'])
        output_df = pd.DataFrame.from_dict(out_dict)
        output_df.to_csv('C:/Users/parke/Documents/code/nuclei_find/csv_files/Nuclei_Spot_Quantification_' + str(self.csv_file))

        color_dict = {}
        for spot_index in points_list:
            color_dict[spot_index] = np.random.rand(3,)
        #### TURN ON FOR INTERACTIVE PLOT ####
        # fig = plt.figure()
        # plt.imshow(self.nuclei_image, cmap='gray')
        # # plt.imshow(self.labeled_spots_im, cmap='gray')
        # print('Plotting Dense Nuclei... \n')
        # for spott in tqdm(dict_dense.keys()):
        #     x = [index[1] for index in dict_dense[spott]]
        #     y = [index[0] for index in dict_dense[spott]]
        #     plt.scatter(x, y, c=color_dict[spott], marker='.')
        # print('Plotting Sparse Nuclei... \n')
        # for spott in tqdm(dict_sparse.keys()):
        #     x = [index[1] for index in dict_sparse[spott]]
        #     y = [index[0] for index in dict_sparse[spott]]
        #     plt.scatter(x, y, c=color_dict[spott], marker='*')
        # plt.show()


        # creating the final image to show clustering
        print('Making Figure Image...')
        figure_image_ar = np.array(self.nuclei_mask).copy()
        color_dict_2 = {}
        for spot_index in points_list:
            # color_dict_2[spot_index] = np.random.randint(0,np.amax(points_list))
            color_dict_2[spot_index] = (dict_dense_count.get(spot_index, 0) + dict_sparse_count.get(spot_index, 0))
            # color_dict_2[spot_index] = (dict_dense_count.get(spot_index, 0)) # Dense
            # color_dict_2[spot_index] = (dict_sparse_count.get(spot_index, 0)) # Sparse

        exp_factor = 2 # factor used to expand the nuclei in the heat map
        for spott in tqdm(dict_dense.keys()):
            x = np.array([index[1] for index in dict_dense[spott]]).astype('int')
            y = np.array([index[0] for index in dict_dense[spott]]).astype('int')
            for index in zip(x,y):
                x,y = index
                if x < self.input_shape[1]-exp_factor and y < self.input_shape[0]-exp_factor and x > exp_factor and y > exp_factor:
                    figure_image_ar[y-exp_factor:y+exp_factor,x-exp_factor:x+exp_factor] = color_dict_2[spott]
                    # figure_image_ar[y-exp_factor:y+exp_factor,x-exp_factor:x+exp_factor] = 20
                else:
                    figure_image_ar[y,x] = color_dict_2[spott]

        for spott in tqdm(dict_sparse.keys()):
            x = np.array([index[1] for index in dict_sparse[spott]]).astype('int')
            y = np.array([index[0] for index in dict_sparse[spott]]).astype('int')
            for index in zip(x,y):
                x,y = index
                if x < self.input_shape[1]-exp_factor and y < self.input_shape[0]-exp_factor and x > exp_factor and y > exp_factor:
                    figure_image_ar[y-exp_factor:y+exp_factor,x-exp_factor:x+exp_factor] = color_dict_2[spott]
                    # figure_image_ar[y-exp_factor:y+exp_factor,x-exp_factor:x+exp_factor] = 5
                else:
                    figure_image_ar[y,x] = color_dict_2[spott]

        ax = plt.subplot(111)
        plt.axis('off')
        image = ax.imshow(figure_image_ar, cmap=hm_cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=cax)
        plt.show()

        kernel_r = 50
        figure_image_ar_2 = np.array(self.nuclei_mask).copy()
        figure_image_ar_3 = np.array(self.nuclei_mask).copy()
        fig_im = np.lib.pad(figure_image_ar_2,kernel_r*2,self.padwithzeros)
        fig_im_2 = np.lib.pad(figure_image_ar_3,kernel_r*2,self.padwithzeros)
        print('max color_dict_2', np.amax(np.array(list(color_dict_2.values()))))
        print('max fig_im', np.amax(fig_im))

        for spot_numb in points_list:
            y,x = array_for_fit[spot_numb]
            y += kernel_r*2
            x += kernel_r*2
            weighted_kernel = np.multiply(morphology.disk(kernel_r), color_dict_2[spot_numb])
            if fig_im[y-kernel_r:y+kernel_r+1,x-kernel_r:x+kernel_r+1].shape == (kernel_r*2+1,kernel_r*2+1):
                fig_im[y-kernel_r:y+kernel_r+1,x-kernel_r:x+kernel_r+1] = np.add(weighted_kernel, fig_im[y-kernel_r:y+kernel_r+1, x-kernel_r:x+kernel_r+1])
                fig_im_2[y-kernel_r:y+kernel_r+1,x-kernel_r:x+kernel_r+1] = np.multiply(weighted_kernel, fig_im_2[y-kernel_r:y+kernel_r+1, x-kernel_r:x+kernel_r+1])
            else:
                print('Padding format error... \n', (x,y))
                fig_im[y-kernel_r:y+kernel_r+2,x-kernel_r:x+kernel_r+2] = np.multiply(weighted_kernel, fig_im[y-kernel_r:y+kernel_r+2, x-kernel_r:x+kernel_r+2])

        ax = plt.subplot(111)
        image = ax.imshow(np.lib.pad(figure_image_ar_2,kernel_r*2,self.padwithzeros), cmap='nipy_spectral')
        image2 = ax.imshow(fig_im, alpha=0.7, cmap=hm_cmap)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image2, cax=cax)
        plt.show()
        # plt.imshow(np.lib.pad(figure_image_ar_2,kernel_r*2,self.padwithzeros), cmap='nipy_spectral')
        # plt.imshow(fig_im, alpha=0.7, cmap='nipy_spectral')
        # plt.show()

        ax = plt.subplot(111)
        image = ax.imshow(np.lib.pad(figure_image_ar_3,kernel_r*2,self.padwithzeros), cmap='nipy_spectral')
        image2 = ax.imshow(fig_im_2, alpha=0.7, cmap=hm_cmap)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image2, cax=cax)
        plt.show()
        # plt.imshow(np.lib.pad(figure_image_ar_3,kernel_r*2,self.padwithzeros), cmap='nipy_spectral')
        # plt.imshow(fig_im_2, alpha=0.7, cmap='nipy_spectral')
        # plt.show()
        # return [dict_dense, dict_dense, points_list]

    # def density(self, coms_list):
    #     print('Importing ROIs...')
    #     dense_image = np.zeros([self.input_shape[0],self.input_shape[1]])
    #     for spott in
    #     sparse_image = np.zeros([self.input_shape[0],self.input_shape[1]])
    #     ROI_masks = [file for file in os.listdir() if file.endswith('roi.png')]
    #     # normalizing mask to czi image size
    #     ROI_arrays = {}
    #     for file in tqdm(ROI_masks):
    #         ROI_arrays[file.split('_')[0]] = (skimage.io.imread(file))
    #
    #     for file in



    def padwithzeros(self, vector, pad_width, iaxis, kwargs):
        '''
        This function pads an image with zeros around the borders. Used by np.lib.pad
        to generate padded image.

        Args:
        vector (numpy array): Array instance used to pad array
        pad_width (tuple): Tuple with length corresponding to the dimensions of
        the array being padded describing the amount to be padded by for example
        (2,3) would add +2 and +3 to the first and second dimensions of a 2D
        array.

        Returns:
        Vector: vector used by np.lib.pad to pad the array of interest.
        '''
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector

    def for_csv(self, coms, csv_name='nucelei_markers'):
        '''
        This function generates a dictionary that is used to create a pandas
        dataframe with the cell counts and nuclei type (sparse or dense) counts.

        Args:
        coms (list of lists): list of lists of the form
        [[list of center of mass sparse], [list of center of mass dense]]

        Returns:
        out_dict: Dictionary to use for generating pandas dataframe for csv generation
        of the form {'Image Name': str, 'Cell Number': int, 'X Pixel': int,
        'Y Pixel': int, 'Nuclei Type:': str, 'Dense': int, 'Sparse': int}

        '''
        image_name = []
        cell_number = []
        nuc_loc_x = []
        nuc_loc_y = []
        nuc_type = []
        sparse_nuc = 0
        dense_nuc = 0

        print('Formatting Output... \n')
        n_type = 0
        c = 0
        for type in tqdm(coms):
            for cell in type:
                cell_number.append(c)
                nuc_loc_x.append(np.round(cell[1]))
                nuc_loc_y.append(np.round(cell[0]))
                image_name.append(self.filename)
                if n_type == 0:
                    nuc_type.append('Sparse')
                    sparse_nuc += 1
                else:
                    nuc_type.append('Dense')
                    dense_nuc += 1
                c += 1
            n_type += 1
        out_dict = {'Image Name': image_name, 'Cell Number': cell_number,
        'X Pixel': nuc_loc_x, 'Y Pixel': nuc_loc_y, 'Nuclei Type:': nuc_type,
        'Dense': dense_nuc, 'Sparse': sparse_nuc}
        return out_dict

    def show_nuclei(self, cell_dict):
        '''
        This function creates a matplotlib interactive figure of the dense and
        sparse nuclei masks and the nuclei image overlaid with the markers of
        the nuclei centroids. The figure also includes a nuclei image with no
        markers for reference when navigating the interactive plot.

        Args:
        cell_dict (dictionary): dictionary returned by the for_csv function. This
        dictionary is of the form {'Image Name': str, 'Cell Number': int,
        'X Pixel': int, 'Y Pixel': int, 'Nuclei Type:': str,
        'Dense': int, 'Sparse': int}

        Returns:
        None (generates interactive matplotlib figure instance)

        '''

        print('Generating Validation Image... \n')
        if len(cell_dict['Cell Number']) > 1:
            x_sparse = [x for x in cell_dict['X Pixel'][0:(cell_dict['Sparse']-1)]]
            y_sparse = [y for y in cell_dict['Y Pixel'][0:(cell_dict['Sparse']-1)]]
            data_sparse = (x_sparse, y_sparse)
            x_dense = [x for x in cell_dict['X Pixel'][cell_dict['Sparse']:-1]]
            y_dense = [y for y in cell_dict['Y Pixel'][cell_dict['Sparse']:-1]]
            # data_dense = (x_dense, y_dense)
            # data = (data_sparse, data_dense)
            # colors = ('r', 'b')
            # groups = ('sparse', 'dense')
            # for data, color, group in zip(data, colors, groups):
            # x, y = data

            fig = plt.figure()
            ax1, ax2, ax3, ax4 = fig.subplots(4,1, sharex=True, sharey=True)
            ax1.imshow(self.nuclei_mask, cmap='gray')
            ax1.scatter(x_sparse, y_sparse, c='r', marker='.')
            ax1.scatter(x_dense, y_dense, c='b', marker='.')
            ax1.set_title('Nuclei Mask Sparse')
            ax2.imshow(self.corrected_ni, cmap='gray')
            ax2.scatter(x_sparse, y_sparse, c='r', marker='.')
            ax2.scatter(x_dense, y_dense, c='b', marker='.')
            ax2.set_title('Nuclei Image')
            ax3.imshow(self.nuclei_mask_dense, cmap='gray')
            ax3.scatter(x_sparse, y_sparse, c='r', marker='.')
            ax3.scatter(x_dense, y_dense, c='b', marker='.')
            ax3.set_title('Nuclei Mask Dense')
            ax4.imshow(self.corrected_ni, cmap='gray')
            ax4.set_title('Nuclei Image Alone')
            plt.show()
        else:
            print('No Nuclei Detected.')

def find_nuclei(file_list, csv_filename='nuclei_locations'):
    print('Generating Class Instances... \n')
    czis = []
    for image, csv in tqdm(file_list):
        czis.append(czi_image(image, csv))
    dfs = []
    print('Analyzing Images... \n')
    for im_czi in tqdm(czis):
        cells = im_czi.cell_markers()
        cell_dict = im_czi.for_csv(cells)
        im_czi.show_nuclei(cell_dict)
        dfs.append(pd.DataFrame.from_dict(cell_dict))
    c = 1;
    for df_ in dfs:
        df_new = df_.drop(columns=['Sparse','Dense'])
        df_new.to_csv(csv_filename + '_' + str(c) + '.csv')
        c += 1

def cluster_nuclei(file_list, csv_filename='nuclei_locations'):
    print('Generating Class Instances... \n')
    czis = []
    for image, csv in tqdm(file_list):
        czis.append(czi_image(image, csv))
    dfs = []
    print('Analyzing Images... \n')
    for im_czi in tqdm(czis):
        im_czi.save_nuclei_mask()
        cells = im_czi.cell_markers()
        im_czi.nearest_spot_clustering(cells)
        cell_dict = im_czi.for_csv(cells)
        im_czi.show_nuclei(cell_dict)
        dfs.append(pd.DataFrame.from_dict(cell_dict))
    c = 1
    for df_ in dfs:
        df_new = df_.drop(columns=['Sparse','Dense'])
        df_new.to_csv(csv_filename + '_' + str(c) + '.csv')
        c += 1
    # df = dfs[0]
    # for df2 in dfs[1:]:
    #     df.concat[[df,df2]]
    # df.to_csv(csv_filename + '.csv')

# def test(im_list):
#     print('This is a test...')
#     im_czi = czi_image(im_list[0])
#     print('otsu:', im_czi.otsu)
#     cells = im_czi.cell_markers()
#     cell_dict = im_czi.for_csv(cells)
#     im_czi.show_nuclei(cell_dict)

# test(im_list)
# find_nuclei(file_list)
cluster_nuclei(file_list)
