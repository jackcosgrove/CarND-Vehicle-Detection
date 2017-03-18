import pickle
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from helper_functions import *
from hyperparameters import *

class HeatQueue(object):

    def __init__(self, queue_depth=20):
        self.queue_depth = queue_depth
        self.queue = []

    def append_heat(self, heat):
        self.queue.append(heat)
        if len(self.queue) > self.queue_depth:
            self.queue.pop(0)

    def get_heat_map(self):
        maps = np.array(self.queue)
        return np.sum(maps, axis=0)

global svc
global X_scaler
global img_number
global heat_queue

img_number = 0
heat_queue = HeatQueue(queue_depth=heat_queue_depth)

with open('vehicle_fit.p', 'rb') as handle:
    data = pickle.load(handle)

svc = data['svc']
X_scaler = data['scaler']

def process_image(image):
    global svc
    global X_scaler
    global img_number
    global heat_queue

    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=cspace, 
                        spatial_size=spatial_size, nbins=nbins, bins_range=bins_range,
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        canny_low_threshold=canny_low_threshold,
                        canny_high_threshold=canny_high_threshold)

    window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)

    if img_number % 20 == 0:
        draw_image = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output_images/sample_boxes_' + str(img_number) + '.png', draw_image)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Store the heat in the queue
    heat_queue.append_heat(heat)

    # Visualize the heatmap when displaying    
    heatmap = heat_queue.get_heat_map()

    # Apply threshold to remove false positives
    heatmap = apply_threshold(heatmap, heat_threshold)

    if img_number % 20== 0:
        draw_heatmap = np.clip(heatmap, 0, 255)
        cv2.imwrite('output_images/sample_heatmap_' + str(img_number) + '.png', draw_heatmap)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    final_img = draw_labeled_bboxes(np.copy(image), labels)

    img_number += 1
    
    return final_img

#output = 'test_video_output.mp4'
#clip1 = VideoFileClip("test_video.mp4")
output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)
