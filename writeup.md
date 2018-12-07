# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Report

### 1. Pipeline description


The pipeline consisted of 10 steps: 
1) The images are converted to grayscale
2) A Gaussian blur is applied to the image, with kernel 7x7
3) Use Canny transform to exctract only edges of the image (future lines), with `low_treshold = 50` and `high_threshold = 150`
4) From that image get a region of interest defined with a four side polygone, where lanes normally are. The bottom coordinates of the polygone are just the corners of the image. The other 2 coordinates are defined as follows: the y-coordinate is the 61% of the height of the image, and the x-coordinates are the 45% and 55% of width of the image respectively.
5) Use that region of interest to make a mask and perform a bit-wise-and operation with the canny transform image to keep only the information inside that region.
6) Get all possible lines of that filtered image with the Hough transform, with these parameters: `rho = 2`, `theta = np.pi/180 `, `threshold = 20`, `min_line_length = 10`, `max_line_gap = 10`
7) Separate the right and left lines by their slope and position in the image. Slopes greater than `0.4` and belonging to the second half of the image correspond to right lane lines, and slopes less than `-0.4` and first half of the image belongs to the left lane lines.
8) Finally fit with two linear regressions the two group of lines. This returns the equations of the two lane lines.
9) With the equations, just get 2 points that satisfy each equations. Here we have the `y` coordinates where we want to start and end the lines, i.e the bottom of the image and the starting point of the polygon mask. With that, `x` can be calculated as: `x = (y-b)/m`, where `m` is the slope and `b` the intercept of the linear fit. This returns the final lines that will be displayed on the original image.
10) Use the final lines to greate a black image with only those lines, and perform a `weighted addition` with the original color image.

The function `draw_lines()` was not modified to keep the code more modular. So that function only does what its name says, draw specified lines. A special function called `get_left_right_lines()` does the trick. It implements the 7th and 8th steps of the pipeline. A function called `fit_lines()` uses this information to calculate the 9th step.


![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be that the lines detection part with the Canny transform fails

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to 

Another potential improvement could be to 
