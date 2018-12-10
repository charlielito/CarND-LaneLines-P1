# **Finding Lane Lines on the Road** 


**Carlos Andres Alvarez, Udacity Self Driving Nanodegree**

[//]: # (Image References)

[image1]: ./test_images/solidYellowCurve2.jpg "Original"
[image2]: ./test_images_output/pre_solidYellowCurve2.jpg "Pre"
[image3]: ./test_images_output/full_solidYellowCurve2.jpg "Pos"


### Report

### 1. Pipeline description


The pipeline consisted of 10 steps: 
1) The images are converted to grayscale
2) A Gaussian blur is applied to the image, with kernel 9x9
3) Use Canny transform to exctract only edges of the image (future lines), with `low_treshold = 50` and `high_threshold = 150`
4) From that image get a region of interest defined with a four side polygone, where lanes normally are. The bottom coordinates of the polygone are just the corners of the image. The other 2 coordinates are defined as follows: the y-coordinate is the 61% of the height of the image, and the x-coordinates are the 45% and 55% of width of the image respectively.
5) Use that region of interest to make a mask and perform a bit-wise-and operation with the canny transform image to keep only the information inside that region.
6) Get all possible lines of that filtered image with the Hough transform, with these parameters: `rho = 2`, `theta = np.pi/180 `, `threshold = 20`, `min_line_length = 10`, `max_line_gap = 10`
7) Separate the right and left lines by their slope and position in the image. Slopes greater than `0.4` and belonging to the second half of the image correspond to right lane lines, and slopes less than `-0.4` and first half of the image belongs to the left lane lines.
8) Finally fit with two linear regressions the two group of lines. This returns the equations of the two lane lines.
9) With the equations, just get 2 points that satisfy each equations. Here we have the `y` coordinates where we want to start and end the lines, i.e the bottom of the image and the starting point of the polygon mask. With that, `x` can be calculated as: `x = (y-b)/m`, where `m` is the slope and `b` the intercept of the linear fit. This returns the final lines that will be displayed on the original image.
10) Use the final lines to greate a black image with only those lines, and perform a `weighted addition` with the original color image.

The function `draw_lines()` was not modified to keep the code more modular. So that function only does what its name says, draw specified lines. A special function called `get_left_right_lines()` does the trick. It implements the 7th and 8th steps of the pipeline. A function called `fit_lines()` uses this information to calculate the 9th step.

Following the main steps of the pipeline can be visualized: Original image -> from 1th to 7th step -> and final steps and returned image.

Original image
![alt text][image1]
From 1th to 7th step
![alt text][image2]
Final steps and returned image
![alt text][image3]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be that the pipeline could fail for images with differet contrasts or images with shadows as in the challenge video, because the parameters for the pipeline were fitted  for the normal videos.

Another shortcoming could be the estimation of the linear fit, because of some recognized lines that not belong to the actual lane marks, that estimation could be affected.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use better parameters to make the pipeline contrast invariant, or use color segmentation or other technique related.

Another potential improvement could be also to take into account big differences of the lines estimation between frames, so this would avoid misleading estimations or no "stable" lines marks. 

Other improvement could be to take previous lines estimation if in the current frame the pipeline fails to detect any lines.
