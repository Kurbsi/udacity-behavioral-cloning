# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[image8]: ./examples/center.jpg "Normal Image"
[image9]: ./examples/left.jpg "Left Image"
[image10]: ./examples/right.jpg "Right Image"
[image11]: ./examples/recover.jpg "Recover Image"
[image12]: ./examples/regular.jpg "Regular Image"
[image13]: ./examples/flipped.jpg "Flipped Image"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My approach on creating a valid and working model was two-folded. I started with the preporcessing of the images and used a model which only consists of one Flatten layer and an additional Dense layer with one output. The input shape is a 160x320x3 image and the outpout represents the steering angle. Training data for the first tests only consists of the center image and the steering angle.

With this approach I gather a basic training set and made sure the general pipeline is working. The pipeline in general was working, but of course the car did not stay on the track for a second.

The first approach to improve the result was to add also the left and right camera images provided by the simulator. The simulator overs three images. A center image, which where the windshield is. This camera is also fed in the autonomous mode to the model to calculate the appropriate steering angle. In addition there are also left and right camera images. These are slightly offset to the center image. Using these camera can greatly improve the result when the car is slightly of the center. To compensate the camera offset for the training set, a correction offset to the steering angle needs to be added. For left and right a correction factor of +0.2 and -0.2 respectively was added.

![alt text][image8]
![alt text][image9]
![alt text][image10]

In the next step I tried to increase the available data for training with the already exisiting data. Especially for the first, easy track the the data is very biased towards steering to the left, since the track is a closed counter-clockwise running loop. To compensate the bias all images are additionally flipped and added to the training set. The appropriate steering angle was inverted and also added to the training set.

![alt text][image12]
![alt text][image13]

One additional step in preprocessing was to mean center and normalize the images. Since this also has to be applied when using the model, this was done already as part of Keras model. The Keras framework offers a Lambda layer where every pixel was divided by 255.0 and mean centered by substracting 0.5. Anther preprocessing step done already in the Keras model was cropping the top and the bottom of the image by 50 px and 20 px respectively. For this purpose the Cropping2D layer was used.

With this training set even with only one Dense layer the car could be kept on the street for a short period, although the car was very slow and weaved left and right pretty heavily.

My final model is based on the model from Nvidia shown in the lecture. As said the first two "layers" of the model are the Lambda layer and the Cropping2D layer. After that the model consists of 5 Convolution2D layers with increasing filters of 24, 36, 48 and 64 for both of the last two layers. The first three Convolution2D layers have a stride of 2x2 and the last two layers have strides of 1x1. All layers have a relu activation layer, which conveniently can already be passed in the Keras Conv2D layer.

After that the data is flatten and passed through four regular Dense (fully connected) layer. The output shape decreases from 100 to 50 to 10 to 1, to finally receive the steering angle.

One addition which was added to the original model was three Dropout layers to compensate overfitting. The first dropout layer as added after the first Conv2D layer with a probability of 0.2, the second and third Dropout layer where added after the flattening the data and after the first Dense layer, each with a probility of 0.5 of dropping input units.

Following a overview of the model:

| Layer  		| Description  				|
|---|---|
|Input			| 160x320x3 RGB image 		|
|Lambda			| mean center and normalize |
|Cropping2D		| output 90x320x3 			|
|Conv2D			| 24 filter, 5x5 kernel size,  2x2 strides, relu activation, output 43x158x24	|
|Dropout		| drop 0.2 of input units	|
|Conv2D			| 36 filter, 5x5 kernel size,  2x2 strides, relu activation, output 20x77x36	|
|Conv2D			| 48 filter, 5x5 kernel size,  2x2 strides, relu activation, output 8x37x48		|
|Conv2D			| 64 filter, 3x3 kernel size,  1x1 strides, relu activation, output 6x35x64		|
|Conv2D			| 64 filter, 3x3 kernel size,  1x1 strides, relu activation, output 4x33x64		|
|Flatten 		| output 8448 |
|Dropout		| drop 0.5 of input units |
|Dense			| output 100 |
|Dropout		| drop 0.5 of input units |
|Dense			| output 50 |
|Dense			| output 10 |
|Dense			| output 1 	|

As far as the loss went, I used mean squared error, as the problem of steering angle is akin to a regression problem. Also the Adam optimizer was use, therefore no need of fine tune the learning rate.

#### 2. Attempts to reduce overfitting in the model

As mentioned dropout layers where added to the original model, although in the end these layers did not greatly improve the rather high validation loss compared to the training loss. Still the model performed on both tracks well enough. This can be seen in the videos [1](easy.mp4) [2](hard.mp4). 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I started with two counter-clockwise laps and on clockwise lape on the easy track. I then also gather short snippets of me recovering onto the road. This was already enough to drive the easy lap autonomously. To also master the harder track I also recorded one lap counter-clockwise and clockwise on this track. This enabled the car already to drive most of the hard track, but had some trouble in harder section like in the beginning where to lanes are visible. The car had the tendency to steer on the opposite lane. This could be mitigated by recording a couple of short sequences from this part of the track. 

