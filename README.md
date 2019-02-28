# **Traffic Sign Recognition** 

## Self-Driving Car Nanodegree Term 1 Project 2
### Scott Henderson

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram]: ./writeup_images/histogram.png "Histogram of Sign Types"

[sample]: ./writeup_images/sample.png "Traffic Sign (original)"
[sample_grayscale]: ./writeup_images/sample_grayscale.png "Traffic Sign (grayscale)"
[sample_normalized]: ./writeup_images/sample_normalized.png "Traffic Sign (normalized)"
[sample_intensity]: ./writeup_images/sample_intensity.png "Traffic Sign (intensity)"
[sample_scaled]: ./writeup_images/sample_scaled.png "Traffic Sign (scaled)"
[sample_rotated]: ./writeup_images/sample_rotated.png "Traffic Sign (rotated)"
[sample_translated]: ./writeup_images/sample_translated.png "Traffic Sign (translated)"

[test_image01]: ./test_images/image01.jpg "Test Image 1"
[test_image02]: ./test_images/image02.jpg "Test Image 2"
[test_image03]: ./test_images/image03.jpg "Test Image 3"
[test_image04]: ./test_images/image04.jpg "Test Image 4"
[test_image05]: ./test_images/image05.jpg "Test Image 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/BScottHenderson/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

My initial implementation was to simply load the data from the saved pickle files, convert images from RGB to grayscale, normalize the images and the implement a LeNet-5 neural network using random data selected from a normal distribution to initialize weights and biases.

With this starting point I began to make modifications. Each one slightly improved the resulting test accuracy though I did not track this for each change.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used simple Python and numpy methods to examine the size and shape of the signs data set (see the Jupyter Notebook for details):

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. This is a histogram showing the frequency of each sign class in the training data set:

![alt text][histogram]

Sample image:

![alt text][sample]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The original data set consisted of color (RGB), uniform sized images of traffic signs. Each image was converted from RGB to grayscale and then normalized. The normalized, grayscale images were then used as a base for adding augmented images to the training and validation sets. To save time I just used simple image manipulation methods found at this site: https://docs.opencv.org/3.4.0/da/d6e/tutorial_py_geometric_transformations.html  Since the histogram indicates that some traffic sign classes have relatively few samples, I settled on a method that would ensure that each traffic sign type has a minimum sample size. To make up the difference I used the existing images for each traffic sign type to create new augmented images that were then added to both the training and validation data sets.

Sample image:

![alt text][sample]

Since I intended to use the LeNet-5 neural network architecture it was necessary to convert the images from RGB to grayscale:

![alt text][sample_grayscale]

Next the image was normalized to provide a similar data distribution for each pixel:

![alt text][sample_normalized]

The next step was to add augmented images. In each case I applied the augmentation method to the grayscale, normalized image and added a new image (and label) to the training and validation data sets. The image augmentation pipeline consists of the following four steps:

1. A random image intensity modifcation. Note that the image has already been converted to grayscale and normalized at this point.

![alt text][sample_intensity]

2. A random scaling operation. My initial attempt was to use the OpenCV resize() function and then crop and/or pad the resulting image to maintain the consistent 32x32 image size required by LeNet-5. This did not prove useful so I used a random perspective warping operation instead.

![alt text][sample_scaled]

3. A random rotation.

![alt text][sample_rotated]

4. A trandom translation.

![alt text][sample_translated]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For this project I just used the standard LeNet-5 achitecture with the addition of dropout operations after each fully connected layer.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16     				|
| Fully connected		| input 400, output 120        					|
| Dropout               |                                               |
| Fully connected		| input 120, output 84        					|
| Dropout               |                                               |
| Softmax				|              									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For training the model I used the TensorFlow AdamOptimizer. I did not have time to test other optimization methods.

There are six hyperparameters that I used for training:

1. Initialization: Mu and sigma to control the normal distribution used for initializing weights and biases. I made a couple of attempts to modify these values but the results were not good and I settled on using mu = 0 and sigma = 0.1 as this seemed to work as well as anything else.
2. Training:
- dropout keep probability: I set this value to 0.5 intially and this is the value I ended up with. At one point I tried a couple of higher values but this did not improve the model accuracy.
- learning rate: I left this value set to 0.001. I tried a few experiments with slightly higher or lower values but these did not improve model training.
- epochs: I used an initial value of 5 but quickly increased the value to 10. In the end the value used - 25 - was a tradeoff between model accuracy and training time
- batch size: I tried several values between 64 and up to 1024 for this value and settled on 128 as a value that would not require quite as much memory but still provided adequent training performance

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95.4%
* validation set accuracy of - I do not have separate training and validation accuracy values.
* test set accuracy of 94.3%

I chose to use the LeNet-5 architecture becuase I was familiar with it having just completed the CNN lessons. This architecture worked well on the mnist data set so it seemed like it would be a good starting place for traffic sign classification. In the end I did not have time to explore other architecture options or even to attempt to modify the basic LetNet-5 architecture. I spent the majority of my time on this project working on adding augmented images to the training and validation sets. The various details of this effort proved to affect the final accuracy of the model to a huge degree. In the end I was barely able to achieve the minimum 0.93 accuracy value required for this project.

I think more work can be done in hyperparameter tuning. For the training hyperparameters I simply tried few more or less random values in an attempt to improve model accuracy. A more systematic - and even programmatic - approach is called for.

Also there are several hyperparameters I added related to image augmentation. These values were also set just by trial and error and I think a lot more time could be spent deriving more useful values. One related area is the number of augmented images to add to each data set. The way I chose to handle this was to ensure that each traffic sign type has a minimum number of samples in the data set. Changing this miniumum number had a large impact on the model accuracy. So, again, more time could be spent investigating these minimum values.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test_image01] ![alt text][test_image02] ![alt text][test_image03]![alt text][test_image04] ![alt text][test_image05]

The yield sign and the road work sign - images 1 and 2 - may be difficult to classify due to other objects in the image, especially in the case of the road work sign.

The two speed limit signs - image 3 and 5 - might be difficult to classify because the sign is small and off center in each case. Also there are other objects in the image, specifically a road in each case.

The stop sign - image 4 - might be difficult to classify because the perspective is slightly off. The pov seems to be looking up and to one side of the sign.

All of these potential issues are reasons why it is useful to add augmented images to the training data set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											|
| Road Work    			| Road Work										|
| 70 km/h	      		| Priority Road					 				|
| 100 km/h	      		| No Entry						 				|
| Stop Sign      		| Stop		   									| 


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is not really good performance and is an indicator that the model does not work well on real world data. More work is clearly needed but that is a project for another time. It is interesting to note that both of the failures were for the speed limit signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The first image is a yield sign.  The model correctly predicted this as a yield sign with a probability of 1.0.
The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Yield		   									| 
| 0.0     				| Road work										|
| 0.0					| Ahead only									|
| 0.0	      			| Speed limit (50km/h)			 				|
| 0.0				    | Double curve      							|

The second image is a Road work sign.  The model correctly predicted this as a Road work sign with a probability of 1.0.
The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Road work	   									| 
| 0.0     				| Double curve									|
| 0.0					| Bicycles crossing								|
| 0.0	      			| Beware of ice/snow			 				|
| 0.0				    | Road narrows on the right						|

The third image is a Speed limit (70km/h) sign.  The model incorrectly predicted this as a Priority road sign with a probability of 0.98. This is quite a high probability given that the classification is clearly wrong. We do see a speed limit sign - albeit with the wrong speed limit number - as the third highest softmax probability but this still rounds to 0.0.
The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98        			| Priority road									| 
| 0.01    				| Roundabout mandatory							|
| 0.0					| Speed limit (30km/h)							|
| 0.0	      			| Traffic signals				 				|
| 0.0				    | No vehicles	      							|

The fourth image is a Stop sign.  The model correctly predicted this as a Stop sign with a probability of 0.99.
The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Stop		   									| 
| 0.01     				| Speed limit (70km/h)							|
| 0.0					| Speed limit (50km/h)							|
| 0.0	      			| No entry						 				|
| 0.0				    | Traffic signals      							|

The fifth image is a Speed limit (100km/h) sign.  The model incorrectly predicted this as a No entry sign with a probability of 1.0. As noted above for the other speed limit sign, this is a very high probablity for an incorrect prediction. Again it is interesting to note that the model seems to have a problem with speed limit signs. At least in this very small additional test data set.
The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| No entry	   									| 
| 0.0     				| Go straight or right							|
| 0.0					| Stop											|
| 0.0	      			| Bumpy road					 				|
| 0.0				    | Traffic signals      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

This seems like an interesting exercise. However, this project is already more than a week late so it will have to wait for another time.
