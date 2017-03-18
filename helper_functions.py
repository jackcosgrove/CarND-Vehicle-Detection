import cv2
import numpy as np
from skimage.feature import hog

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def canny(img, size=(32, 32), low_threshold=1, high_threshold=10):
    gray = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges.ravel()

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

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

def single_image_features(image, cspace='RGB', spatial_size=(32,32),
                        nbins=32, bins_range=(0,256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        canny_low_threshold=1, canny_high_threshold=10):
    # Create a list to append feature vectors to
    feature_image = convert_image(image, cspace)     

    # Apply bin_spatial() to get spatial color features
    spatial = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() to get color histogram features
    color = color_hist(feature_image, nbins=nbins, bins_range=bins_range)
    # Apply canny() to detect edges
    canny_feat = canny(feature_image, size=spatial_size,
                       low_threshold=canny_low_threshold,
                       high_threshold=canny_high_threshold)
        
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(features)        
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)

    # Return list of feature vectors
    return np.concatenate((spatial, color, canny_feat, hog_features))

def extract_features(images, cspace='RGB', spatial_size=(32,32),
                     nbins=32, bins_range=(0,256), orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     canny_low_threshold=1, canny_high_threshold=10):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in images:
        features.append(single_image_features(image, cspace=cspace,
                        spatial_size=spatial_size, nbins=nbins,
                        bins_range=bins_range, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel,
                        canny_low_threshold=canny_low_threshold,
                        canny_high_threshold=canny_high_threshold))
        
    # Return list of feature vectors
    return features

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# A function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    x_start, x_stop = x_start_stop
    y_start, y_stop = y_start_stop
    if x_start is None:
        x_start = 0
    if x_stop is None:
        x_stop = img.shape[1]
    if y_start is None:
        y_start = 0
    if y_stop is None:
        y_stop = img.shape[0]
    
    # Compute the span of the region to be searched 
    x_span = x_stop - x_start
    y_span = y_stop - y_start
    
    # Compute the number of pixels per step in x/y
    x_step = int(xy_window[0] * (1-xy_overlap[0]))
    y_step = int(xy_window[1] * (1-xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    x_nwin = int((x_span-nx_buffer) / x_step)
    y_nwin = int((y_span-ny_buffer) / y_step)
    
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    x_pos = x_start
    y_pos = y_start
    for y_win in range(y_nwin):
        for x_win in range(x_nwin):
            window_list.append(((x_pos, y_pos), (x_pos + xy_window[0], y_pos + xy_window[1])))
            x_pos += x_step
        x_pos = x_start
        y_pos += y_step
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32,32), nbins=32, bins_range=(0,256),
                    orient=9, pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, canny_low_threshold=1,
                    canny_high_threshold=10):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using extract_features()
        features = single_image_features(test_img, cspace=color_space,
                            spatial_size=spatial_size, nbins=nbins, bins_range=bins_range,
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel,
                            canny_low_threshold=canny_low_threshold,
                            canny_high_threshold=canny_high_threshold)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
