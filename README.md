# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Image"
[image2]: ./examples/left.jpg "Left Image"
[image3]: ./examples/right.jpg "Right Image"
[image4]: ./examples/centerFlipped.jpg "Center Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 - containing a recording of the vehicle driving autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The images are cropped to remove sky and car body in the images. (model.py line 59)

The model consists of a convolution neural network with 4 layers of 5x5 filter sizes and depth of 5. (model.py lines 61-67)

RELU activation function is used to introduce nonlinearity. (model.py lines 61-67)

Each convolution layer is pooled using max pooling. (model.py lines 61-67)

After the convolution layers there is a dropout layer to reduce over fitting. (model.py line 69)

The dropout layer is followed by 3 fully connected layers of size 600, 300 and 1. (model.py lines 70-73)


#### 2. Attempts to reduce overfitting in the model

As stated, the model contains a dropout layer to reduce over fitting. (model.py line 69)

The model was fit with a 20% validation set. This was randomly selected from the full training set in the call to fit(). (model.py line 76)


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

I almost exclusivly used center lane driving to gather my data. I used images from the left and right camera to train on recovery hence no recovery laps were undertaken. To remove left bias of the training set I created additional mirrored versions of the center image. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the LeNet model. Being familiar with the structure I thought this would be an easy first step to facilitate a benchmark.

This first model has high error on both training and validation datasets. When run in the simulator the car tended to drive straight at corners.

For my second model I used an architecture similar to the VGG model. This tended to overfit as evident by the small error on training dataset and high error on the validation dataset.

To combat the overfitting I added a dropout layer between the convolution layers and fully connected layers.

I also added some additional data of me driving around the tighter corners as this was a trouble spot for previous models.

With my final model the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 58-73) consisted of a convolution neural network with the following layers and layer sizes.

* Convolution (5 filters: 5x5)
* Relu activation
* Max pooling (2x2)
* Convolution (5 filters: 5x5)
* Relu activation
* Max pooling (2x2)
* Convolution (5 filters: 5x5)
* Relu activation
* Max pooling (2x2)
* Convolution (5 filters: 5x5)
* Relu activation
* Max pooling (2x2)
* Dropout (p=0.5)
* Dense (600)
* Dense (300)
* Dense (1)


#### 3. Creation of the Training Set & Training Process

After practicing with the simulator, I undertook 2 laps of center lane driving. Below is an example of the images captured.

![alt text][image1]

To avoid the need of recording recovery laps, I used the images from the left and right camera with an added correction factor to the steering angle. Below are the corresponding left and right images for the above image.

![alt text][image2]
![alt text][image3]


To augment the data I also flipped the center images and angles. I believed this would reduce the left turn bias seen in the earlier models I fit. Below is the corresponding flipped image. 

![alt text][image4]

In addition to the two standard laps I also undertook additional recorded cornering segments.

The final dataset included a total 26,316 images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 10 and 12 as evidenced by the stagnation in reduction of validation error.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
