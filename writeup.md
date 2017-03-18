##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/preprocessing.png
[image2]: ./output_images/hog.jpg
[image3]: ./output_images/sample_boxes_1000.png
[image4]: ./output_images/sample_heatmap_1000.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I first created a class `ImageFolderDataSet` (line 58 in `train.py`) to extract and label the training and test set images. Since some of the image sets were time series data, I could not simply shuffle all images after concatenating them into a single set. Instead if the `time_series` parameter was set to true, I first divided the entire set into `n` groups, where `n=1/test_size` where `test_size` is the fraction of the set reserved for the test set in decimal form. So a `test_size=0.2` would result in 5 groups. I then used a `sklearn.model_selection.LeaveOneGroupOut` splitter class to randomly reserve one of these groups as the test set. This meant there were at most a few sequential images spanning the training and test sets if the training/test boundaries occurred within image sequences. Had I shuffled all images before splitting, there may have been many more sequential images in both sets. Visual inspection of the GTI time series data showed that image sequences were 3-5 images long.

I also created a class `ImageFileDataSet` to read in images from the Udacity data set. For this reason, I had to split each `DataSet` into training and test sets separately, and then concatenate and shuffle all of these split sets into a single training and test set using the `DataSetShuffler` class. This was not ideal as it loaded all images into memory at once, rather than shuffle references to images and load images during the classifier fit process. However I did not hit the limit of my hardware.

Unfortunately the inclusion of the Udacity data set yielded worse test set accuracy (around 91%) as well as worse performance on the test video. There were too many false positives. I think this is due to the fact that the labeled data in the Udacity data set often has non-square aspect ratios.

After assembling the training and test sets, I extracted features from both. Feautre extraction occurred at in the function `extract_features()` at line 90 of the file `helper_functions.py`. This function was invoked at lines 187 and 193 of `train.py`. 

Feature extraction used four sets of features: binned spatial feautures, color channel histgrams, binned edge detections through the Canny filter, and histogram of gradient (HOG) feautres. I found that the inclusion of the binned spatial feaures and the color channel histograms improved performance, while the Canny edge detction had little discernible effect. I left the Canny edge detection in the classifier it did not seem to hurt performance either.

The code for HOG feauture extraction is contained in the function `get_hog_features()` beginning at line 28 of the file `helper_functions.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are nine labeled examples of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here are the HOG images from the nine examples above using the hue channel of the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tuned my classifier with different HOG parameters on a subset of the training data. Initially I was using the RGB colorspace and red channel as it yielded the highest test set accuracy, but that would change later on to using the HSV colorspace and hue channel. Curiously, the HSV colorspace yielded a worse test set accuracy than the RGB colorspace (about 95% vs. 97%) but in the final video the HSV-trained classifier was better able to identify both vehicles.

I tried various combinations of parameters and given the training image size of `(64, 64)` pixels, a `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` parameters worked well. I found the accuracy of the classifier was insensitive to the `orientations` parameter, so I left its value at `9`. I also found that the hue channel yielded the best test set accuracy.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the default parameters, after normalizing the training and test set features using a single scaler. This normalization occurred in lines 201-207 of `train.py`. The training of the SVM occurred in lines 215-218 of `train.py`. The training set included 14210 samples, with a test set split size of `0.2`. Within this training set, there were 7036 labeled vehicles, so the training set was balanced. The feature vector length was 2884, and the test set accuracy was 0.9546.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

