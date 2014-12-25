##Overview

This work was done during my internship in Tencent company during the summer of 2014.

Given an input street view image, this program can automatically find the speed-limit signs in it.

For example the two images blow are the input and output images.

![input image](/Images/input.png?raw=true "input image")
![output image](/Images/output.png?raw=true "output image")

##Techniques
The pipeline contains 2 mayjor parts. First it detects the candidate windows that may contain speed-limit signs, this part is done by combining a shape detector and a cascade classifier.
Next we recognize the category for each candidate window using convolutional neural networks(CNN), the most popular image classification algorithm at that time (maybe I should use "ever" here).

###Detection
One significant feature of the speed-limit signs is that they all contain red circles and numbers. So the first version of our detection algorithm is to find the red circles in the image.
To do this, we binarize the image using a color filter. We manually set the threshold in HSV space. Then we find the regions which contain at least 2 hierarchical contour levels.
The function cv::findContours in the OpenCV library can directed do such work for us.

After finish this, most speed-limit signs can be found except for those which are occluded by other objects and contain no whole circle. 
To tackle this defect, we resort to the cascade classifier which concatenates several weak classifer using boost techniques.
Cascade classifier works quite well in tasks such as face detection.
We use the color and shape filter to find some (around 800) true patches of speed-limit sign in the training image set(3000+ panorama images shot in Beijing).
These samples are used as positive samples.
We randomly pick some images containing no signs as negative samples and train a initial version of cascade classifier.
We then apply this classifier to the training images and find the signs that are occluded.
The detected signs are used as a new version of positive samples and the false positive patches are added to the negative samples.
After several iterations, the classifier works well.

To train the cascade classifier, the positive and negative samples are converted to grayscale images and their histogram are normalized.
OpenCV provides a convenient way for us to train it. Here we use HOG as the feature descriptor.

In the detection stage, for one speed-limit sign tens of windows might be detected. 
OpenCV automaticlly clusters the windows based on their position. It just compute the average (x, y, width, height) for each group of windows.
We slightly modify this scheme by use the window whose size is the median in the group.

###Recognition
Our aim is to judge what value the sign writes where a multi-class classifier is needed. 
The classes are 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 and non-speed-limit.

We choose CNN for its excellent performance in recent image classification competetions.
The network architecture is almost the same as [1] states except for that the number of neurons in the last layer is 13.

We label the patches detected in the training set by the color & shape filter and the cascade classifier.
To make the size of each category in the training set balanced, affine transforms might be applied to generate multiple copies from one instance of sign. 

In training, each category contains about 1000 samples.
We use cuda-convnet [2] as the CNN framework. 
(I didn't know about any other framework at that time. It's hard to customize cuda-convnet. 2 months later I tried caffe. It is really a huge contribution for the community of neural networks)

##Notice
Most of the code and all data for training is on a work station in Tencent. I don't have a copy.

If you want to use the code or data, please contact my mentor Hai-Feng Deng at Tencent company.

##Reference

[1] Cireşan, D., Meier, U., Masci, J., & Schmidhuber, J. (2012). Multi-column deep neural network for traffic sign classification. Neural Networks, 32, 333-338.

[2] https://code.google.com/p/cuda-convnet/