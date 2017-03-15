import glob, random, math, pickle
import cv2
import pandas as pd
import numpy as np
from os import path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

class DataSet(object):

    def __init__(self):
        self.time_series = False
        self.image_identifiers = []
        self.images = np.empty(0)
        self.image_labels = np.empty(0)

    def _get_images(self, max_size=-1):
        images = []
        for identifier in self.image_identifiers:
            images.append(self._get_image(identifier))
        self.images = np.array(images)

    def get_images(self):
        return self.images

    def get_image_labels(self):
        return self.image_labels

    def _get_image(self, image_identifier):
        pass

    def get_sets(self, test_size = 0.2):
        
        if not self.time_series:
            # Use train_test_split if the data is not time series data
            rand_state = np.random.randint(0, 100)
            img_train, img_test, label_train, label_test = train_test_split(self.images, self.image_labels, test_size=test_size, random_state=rand_state)
        else:
            # Divide the data into 1/test_size groups to minimize the probability
            # that a time series will span the train set-test set split
            n_groups = math.ceil(1./test_size)
            group_size = math.ceil(len(self.images)/n_groups)
            groups = np.repeat(np.arange(n_groups), group_size)[:len(self.images)]
            logo = LeaveOneGroupOut()
            for train_idx, test_idx in logo.split(self.images, y=self.image_labels, groups=groups):
                img_train, img_test, label_train, label_test = self.images[train_idx], self.images[test_idx], self.image_labels[train_idx], self.image_labels[test_idx]

        return img_train, img_test, label_train, label_test

class ImageFolderDataSet(DataSet):

    def __init__(self, image_folder_path, label, time_series=False):
        super(ImageFolderDataSet, self).__init__()
        self.time_series = time_series
        self.image_identifiers = glob.glob(image_folder_path)
        self.image_labels = np.zeros(len(self.image_identifiers)) + label
        self._get_images()

    def _get_image(self, image_identifier):
        image = cv2.imread(image_identifier)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class ImageFileDataSet(DataSet):
    headers = ["xmin","xmax","ymin","ymax","Frame","Label","Preview URL"]
    hot_labels = ["Car", "Truck"]

    def __init__(self, image_file_path, image_file_name = "labels.csv", image_dims = (64,64), max_size=-1, time_series=True):
        super(ImageFileDataSet, self).__init__()
        self.time_series = time_series
        self.image_folder_path = image_file_path
        self.image_dims = image_dims
        image_file = pd.read_csv(image_file_path + image_file_name)
        # Convert the labels to integers
        image_file.Label = image_file.Label.apply(lambda l: 1 if l in self.hot_labels else 0, np.int64)
        # Drop any records with an empty selection area
        self.image_file = image_file.query("ymax > xmax and ymin > xmin")
        # Set the identifiers and labels
        self.image_identifiers = self.image_file.index
        self.image_labels = self.image_file.Label.values
        if max_size >= 0:
            indexes = random.sample(range(len(self.image_identifiers)), max_size)
            indexes.sort()
            self.image_identifiers = self.image_identifiers[indexes]
            self.image_labels = self.image_labels[indexes]
        self._get_images()

    def _get_image(self, image_identifier):
        record = self.image_file.loc[image_identifier]
        image_path = path.join(self.image_folder_path, record.Frame)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[record.xmax:record.ymax,record.xmin:record.ymin,:]
        return cv2.resize(image, self.image_dims, interpolation = cv2.INTER_AREA)

class DataSetShuffler(object):

    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def shuffle(self, data_sets):
        img_train, img_test, label_train, label_test = data_sets[0].get_sets(self.test_size)
        
        for data_set in data_sets[1:]:
            x_train, x_test, y_train, y_test = data_set.get_sets(self.test_size)
            img_train = np.concatenate((img_train, x_train), axis=0)
            img_test = np.concatenate((img_test, x_test), axis=0)
            label_train = np.concatenate((label_train, y_train), axis=0)
            label_test = np.concatenate((label_test, y_test), axis=0)

        rand_state = np.random.randint(0, 100)
        img_train, label_train = shuffle(img_train, label_train, random_state=rand_state)
        rand_state = np.random.randint(0, 100)
        img_test, label_test = shuffle(img_test, label_test, random_state=rand_state)

        return img_train, img_test, label_train, label_test
        

extraNonVehicles = ImageFolderDataSet("non-vehicles/Extras/*", 0)
gtiNonVehicles = ImageFolderDataSet("non-vehicles/GTI/*", 0, time_series=True)
gtiFarVehicles = ImageFolderDataSet("vehicles/GTI_Far/*", 1, time_series=True)
gtiLeftVehicles = ImageFolderDataSet("vehicles/GTI_Left/*", 1, time_series=True)
gtiMiddleCloseVehicles = ImageFolderDataSet("vehicles/GTI_MiddleClose/*", 1, time_series=True)
gtiRightVehicles = ImageFolderDataSet("vehicles/GTI_Right/*", 1, time_series=True)
kittiVehicles = ImageFolderDataSet("vehicles/KITTI_extracted/*", 1)
objectDetection = ImageFileDataSet("object-detection-crowdai/", max_size=14200)

shuffler = DataSetShuffler()
data_sets = [
                 extraNonVehicles,
                 gtiNonVehicles,
                 gtiFarVehicles,
                 gtiLeftVehicles,
                 gtiMiddleCloseVehicles,
                 gtiRightVehicles,
                 kittiVehicles,
                 objectDetection
            ]
img_train, img_test, label_train, label_test = shuffler.shuffle(data_sets)

def draw_image(axis, image, label):
    if label == 0:
        axis.set_title('Not Vehicle', fontsize=10)
    else:
        axis.set_title('Vehicle', fontsize=10)
    axis.imshow(image)

indexes = random.sample(range(len(img_train)), 9)

f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(24,9))
f.tight_layout()
draw_image(ax1, img_train[indexes[0]], label_train[indexes[0]])
draw_image(ax2, img_train[indexes[1]], label_train[indexes[1]])
draw_image(ax3, img_train[indexes[2]], label_train[indexes[2]])
draw_image(ax4, img_train[indexes[3]], label_train[indexes[3]])
draw_image(ax5, img_train[indexes[4]], label_train[indexes[4]])
draw_image(ax6, img_train[indexes[5]], label_train[indexes[5]])
draw_image(ax7, img_train[indexes[6]], label_train[indexes[6]])
draw_image(ax8, img_train[indexes[7]], label_train[indexes[7]])
draw_image(ax9, img_train[indexes[8]], label_train[indexes[8]])
f.savefig('output_images/preprocessing.png')
plt.close(f)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call image output if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return hog_image
    # Otherwise call with features output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# apply color conversion if other than 'RGB'
def convert_image(image, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    return feature_image
    
def extract_features(images, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in images:
        feature_image = convert_image(image, cspace)     

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

cspace='RGB'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0

f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(24,9))
f.tight_layout()
draw_image(ax1, get_hog_features(convert_image(img_train[indexes[0]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[0]])
draw_image(ax2, get_hog_features(convert_image(img_train[indexes[1]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[1]])
draw_image(ax3, get_hog_features(convert_image(img_train[indexes[2]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[2]])
draw_image(ax4, get_hog_features(convert_image(img_train[indexes[3]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[3]])
draw_image(ax5, get_hog_features(convert_image(img_train[indexes[4]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[4]])
draw_image(ax6, get_hog_features(convert_image(img_train[indexes[5]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[5]])
draw_image(ax7, get_hog_features(convert_image(img_train[indexes[6]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[6]])
draw_image(ax8, get_hog_features(convert_image(img_train[indexes[7]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[7]])
draw_image(ax9, get_hog_features(convert_image(img_train[indexes[8]], cspace)[:,:,hog_channel], orient, pix_per_cell, cell_per_block, True), label_train[indexes[8]])
f.savefig('output_images/hog.png')
plt.close(f)

# Extract the HOG features
features_train = extract_features(img_train, cspace=cspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

features_test = extract_features(img_test, cspace=cspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

X = np.vstack((features_train, features_test)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

n_training_samples = len(features_train)
scaled_X_train, scaled_X_test = scaled_X[:n_training_samples], scaled_X[n_training_samples:]

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Number of Training Samples:', n_training_samples)
print('Number of Training Vehicles:', int(label_train.sum(0)))
print('Feature vector length:', len(scaled_X_train[0]))

# Use a linear SVC 
svc = LinearSVC()

svc.fit(scaled_X_train, label_train)

print('Test Accuracy of SVC = ', round(svc.score(scaled_X_test, label_test), 4))

with open('vehicle_fit.p', 'wb') as handle:
    pickle.dump(svc, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('vehicle_fit.p', 'rb') as handle:
#    svc = pickle.load(handle)
