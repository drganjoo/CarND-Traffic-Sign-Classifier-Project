# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


Link to my [project code](https://github.com/drganjoo/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

Dataset given has been broken down into training, validation and testing sub sets. 

* Total dataset: 51839
* Number of training examples = 34799 (67.13% of total)
* Number of validation examples = 4410 (8.51% of total)
* Number of testing examples = 12630 (24.36% of total)
* Image data shape = (32, 32, 3)
* Number of classes = 43

Signs:

0. Speed limit (20km/h)
1. Speed limit (30km/h)
2. Speed limit (50km/h)
3. Speed limit (60km/h)
4. Speed limit (70km/h)
5. Speed limit (80km/h)
6. End of speed limit (80km/h)
7. Speed limit (100km/h)
8. Speed limit (120km/h)
9. No passing
10. No passing for vehicles over 3.5 metric tons
11. Right-of-way at the next intersection
12. Priority road
13. Yield
14. Stop
15. No vehicles
16. Vehicles over 3.5 metric tons prohibited
17. No entry
18. General caution
19. Dangerous curve to the left
20. Dangerous curve to the right
21. Double curve
22. Bumpy road
23. Slippery road
24. Road narrows on the right
25. Road work
26. Traffic signals
27. Pedestrians
28. Children crossing
29. Bicycles crossing
30. Beware of ice/snow
31. Wild animals crossing
32. End of all speed and passing limits
33. Turn right ahead
34. Turn left ahead
35. Ahead only
36. Go straight or right
37. Go straight or left
38. Keep right
39. Keep left
40. Roundabout mandatory
41. End of no passing
42. End of no passing by vehicles over 3.5 metric tons

#### 2. Include an exploratory visualization of the dataset.

The following bar chart shows data distribution in the training set:

![Distribution of Data](/writeup/distribution.png?raw=true "Distribution")

**Top 6** occurring categories in the training set:

| Category #| Name                                         |Count |
|-----------|----------------------------------------------|------|
| 2         | Speed limit (50km/h)                         | 2010 |
| 1         | Speed limit (30km/h)                         | 1980 |
| 13        | Yield                                        | 1920 |
| 12        | Priority road                                | 1890 |
| 38        | Keep right                                   | 1860 |
| 10        | No passing for vehicles over 3.5 metric tons | 1800 |

**Least 6** occurring categories in the training set:

| Category #     | Name                                | Count |
|------|-------------------------------------|-----|
| 0    | Speed limit (20km/h)                | 180 |
| 37   | Go straight or left                 | 180 |
| 19   | Dangerous curve to the left         | 180 |
| 32   | End of all speed and passing limits | 210 |
| 27   | Pedestrians                         | 210 |
| 41   | End of no passing                   | 210 |


### Some sample images

![50](/writeup/50.png "50")
![30](/writeup/30.png "30")
![end-of-no-passing](/writeup/end-of-no-passing.png "end-of-no-passing")
![gostraight-left](/writeup/gostraight-left.png "go straight or left")
![keepright](/writeup/keepright.png "keep right")
![nopass](/writeup/nopass.png "no pass")
![priority](/writeup/priority.png "priority")
![yield](/writeup/yield.png "yeild")

### Design and Test a Model Architecture

#### Preprocessing

1) **YCbCr** color space has been used. This color space provides a more natural way of representing luminosity and image colors.

[More details on YCbCr](https://en.wikipedia.org/wiki/YCbCr)

To convert to YCbCr, cv2 has not been used as that did not work on all images at the same time. Instead the following formula has been used:

```
def get_ycbcr(data):
    ycbcr = np.array([[[0.299],  [0.5]      , [-0.168736]],
                  [[0.587],  [-0.418688], [-0.331264]],
                  [[0.114],  [-0.081312], [0.5]]      
                 ]).squeeze()
    data_ycbcr = np.dot(data, ycbcr) + np.array([0,128,128])
    return data_ycbcr
```

2) Normalization: In order to reduce the effect of brightness on image classification, all of the images have been normalized before training. The Y channel is normalized separate from the other 2 color channels of CbCr.

Mean is computed across all of the training set's Y component and CbCr
```
        mean = [np.mean(data_ycbcr[:,:,:,0]), np.mean(data_ycbcr[:,:,:,(1,2)])]
```

Similarly, standard deviation is computed across all of the training set's Cb & Cr components:

```
        sigma = [np.std(data_ycbcr[:,:,:,0]), np.std(data_ycbcr[:,:,:,(1,2)])]
```

In order to normalize validation and testing sets, mean and standard deviation **are not** recomputed but the same ones that were computed from the training set is used.

```
def normalize_ycbcr(data, mean=None, sigma=None):
    data_ycbcr = get_ycbcr(data)
    
    if mean == None or sigma == None:
        mean = [np.mean(data_ycbcr[:,:,:,0]), np.mean(data_ycbcr[:,:,:,(1,2)])]
        sigma = [np.std(data_ycbcr[:,:,:,0]), np.std(data_ycbcr[:,:,:,(1,2)])]
        
    data_ycbcr[:,:,:,0] -= mean[0]
    data_ycbcr[:,:,:,0] /= sigma[0] + 1e-7
    data_ycbcr[:,:,:,1:2] -= mean[1]
    data_ycbcr[:,:,:,1:2] /= sigma[1] + 1e-7
    
    #print('After norm[0,0,0,0]:', data_ycbcr[0,0,0])
    
    return data_ycbcr, mean, sigma
```

Various executions of the model with different parameters, different color spaces and different architecutres, all resulted in overfitting of data. Hence, more data was generated from the given ones.

Also, since the distribution of data was not uniform (minimum 180 images of 'speed limit 20' versus maximum 2010 images of 'speed limit 50'), it was decided to generate images in such a way to make this distribution more unifrom.

Following functions were used for data generation:

1) Scale Image to 36x36, then increase brightness, crop it back to 32x32
2) Scale Image to 36x36, then decrease brightness, crop it back to 32x32
3) Rotate Image randomly between 10 to 20 degrees
4) Rotate Image randomly between -10 to -20 degrees
5) Reduce image to 28x28
6) Add random noise to the image
7) Shift image 2 pixels to the left
8) Shift image 2 pixels to the right

Some examples:

![Rotate Left](/writeup/9960.jpg)
![Rotate Right](/writeup/9963.jpg)
![Shift](/writeup/9772.jpg)

The difference between the original data set and the augmented data set is the following:

* Number of original training examples = 34799
* Extra generated: (49981, 32, 32, 3)
* Combined training set: (84780, 32, 32, 3)

![Distribution Combined](/writeup/distribution_comb.png "combined distribution")

#### 2. Final Model Architecture

Final model consisted of the following layers:

| Layer         		|     Description	        					| 
|---------------------|---------------------------------------------| 
| Input         		| 32x32x3 YCbCr Normalized Image    			| 
| Convolution 5x5x3x6  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| 					                            |
| Max Pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5x6x16  | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU		            |           									|
| Max Pooling	        | 2x2 stride, outputs 5x5x16                    |
| Flatten   	        | Input: 5x5x16, Output: 400                    |
| Fully connected N1    | Input: 400, Output: 120                       |
| RELU		            |           									|
| Fully connected N2    | Input: 120, Output:  84                       |
| RELU		            |           									|
| DROPOUT               | Keep Probability: 0.5                         |
| Fully connected N3    | Input: 84, Output:  43                        |


#### 3. Model Training Parameters

To train the model, the following parameters were used:

| Component         	| Description    		                      | 
|-----------------------|---------------------------------------------| 
| EPOCH     | 250                        |
| Batch Size     | 128                        |
| Learning Rate     | 0.001                        |
| Initial Weights   | Xavier Initializer                        |
| Dropout Keep Probability   | 0.5                        |
| Optimizer   | Adam Optimizer                        |
| Cost Function   | Softmax Cross Entropy                        |
| Evaluation Mechanism   | Best weights found in any EPOCH rather than last|


#### 4. Approach taken for finding the solution

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.957
* test set accuracy of 0.947

An iterative approach was chosen to find the solution:

##### Original LeNet

As a first step the original LeNet was used as that gives a good starting point. Only a maximum of about 0.90 accuracy on validation set was acheived. But training set accuracy was about 0.99 implying that it was overfitting.

##### Weight changes to mu and sigma

Keeping the original LeNet played around with different mu, sigma, Epochs and batch sizes.

It was interesting to find out that batch graident algorithms do not work that well if the batch size is very high. In this particular case I tried using batch size from 128 to almost equal to all of the input size. As I moved higher the validation accuracy kept going down instead of moving up

##### Changed to YCbCr instead of RGB

Since YCbCr has a specific channel for luminosity, I tried using this instead of plain RGB as it provides an easier way to average out the brightness. This however, did not achieve the desired result and it remains to be seen how much of an effect does YCbCr have on the result.

##### Weights changes to Xavier 

Instead of using normally distributed weights, Xavier weight initializers have been used:

```
weights = { 
    'wc1': tf.get_variable("wc1", shape=[5,5,3,6], initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable("wc2", shape=[5,5,6,16], initializer=tf.contrib.layers.xavier_initializer()),
    'wn1': tf.get_variable("wn1", shape=[400,120], initializer=tf.contrib.layers.xavier_initializer()),
    'wn2': tf.get_variable("wn2", shape=[120,84], initializer=tf.contrib.layers.xavier_initializer()),
    'wn3': tf.get_variable("wn3", shape=[84,n_classes], initializer=tf.contrib.layers.xavier_initializer()),
}
```

##### More data was generated

The model was still overfitting, hence more data was generated using the following functions:

1) Scale Image to 36x36, then increase brightness, crop it back to 32x32
2) Scale Image to 36x36, then decrease brightness, crop it back to 32x32
3) Rotate Image randomly between 10 to 20 degrees
4) Rotate Image randomly between -10 to -20 degrees
5) Reduce image to 28x28
6) Add random noise to the image
7) Shift image 2 pixels to the left
8) Shift image 2 pixels to the right

#### Dropout Layer(s) Introduced

Dropout layers were introduced in all intermediate layers including convnet layers. But this started underfitting badly.

#### Dropout Layer(s) In Fully Connected Only

Dropout layers were then kept in fully connected layers only. This reduced underfitting that resulted from dropout in convnet layers and then achieved the desired 0.93 result in validation set.

Poor performance on the loss function was noticed when only one dropout is used:

!["One Dropout Accuracy"](/writeup/one-dropout-accuracy.png "1 Accuracy")
!["One Dropout Loss"](/writeup/one-dropout-loss.png "1 Loss")

With two dropouts the accuracy and loss were much better:

!["Two Dropout Accuracy"](/writeup/two-dropout.png "2 Accuracy")
!["Two Dropout Loss"](/writeup/two-dropout-loss.png "2 Loss")


#### Final Result on Validation Set

* training set accuracy of 0.992
* validation set accuracy of 0.957
* test set accuracy of 0.94

Since it is achieving > 90% on test set, it seems to be performing quite reasonable.


### Why Do I believe LeNet is 'OK' for traffic sign classification

Well to be honest, I don't think LeNet is the best solution for this. Recent papers clearly indicate better performance from GoogLeNet or Resnet on ImageNet so chances are that they would be better than LeNet on traffic sign classification as well.

However, LeNet is sufficient enough for this assignment as it is a much smaller network (computationally) comapred to recent other methods and provides quite good performance as well.

## Test a Model on New Images

#### 1. Real life german traffic sign images:

Instead of choosing from the web a fellow Udacity student provided real life images. Five traffic signs were  cropped out of these images:

!["30 Zone"](/writeup/t1.png "30")
!["Priority"](/writeup/t2.png "Priority")
!["Pedestrian"](/writeup/t3.png "Pedestrian")
!["Yield"](/writeup/t4.png "Yield")
!["Workers"](/writeup/t5.png "Workers")

The first image (30 Zone) might be difficult to classify because it has the word "Zone" written on the traffic sign where as the training set does not have this written:

!["30 Zone"](/writeup/t1-big.jpg "30 Zone Big")
!["30 Training"](/writeup/2769.jpg "30 Training")

The third image (pedistrian with woman and a child) will be near to impossible to detect successfully since nothing close to this was included in the training set:

!["Pedestrian"](/writeup/t3.png "Pedestrian")

It was included since it was a real world image found on the streets and I wanted to check what the system would classify this to be.

#### 2. Model's performance on real world images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 Zone      		    | End of all speed and passing limits | 
| Priority     			| **Priority Road** 										|
| Pedestrian			| Keep Right|
| Yield	      		| **Yield**					 				|
| Road Work			| **Road Work**      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

This does not match the accuracy we had on the test set BUT it might be due to the fact that the first sign (30 Zone) was not exactly the same kind of sign as was present in the training and test sets. And the fact that there was a new sign for pedestrians that did not match anything close to the pedistrian sign we had in training.

#### 3. Model Prediction Certainity 

The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.

For the first image, the model is absolutely wrong on the prediction.

!["30 Zone"](/writeup/t1.png "t1")

| Sign   | Probability  |
|---|---|
|End of all speed and passing limits|0.54|
|Keep right|0.43|
|Go straight or right|0.03|
|Priority road|0.00|
|Turn left ahead|0.00|

!["30 Bar"](/writeup/30-bar.png "30-bar")

For the Second image, the model correctly predicts it as a priority road:

!["Priority"](/writeup/t2.png "t2")

|Sign|Probability|
|----|-----------|
|Priority road|1.00|
|Speed limit (20km/h)|0.00|
|Speed limit (30km/h)|0.00|
|Speed limit (50km/h)|0.00|
|Speed limit (60km/h)|0.00|

!["T2 Bar"](/writeup/t2-bar.png "T2-Bar")

For the third image, the model incorrectly predicts it as keep right, which was expected since this image was never part of the training set however it was found as a real life sign on German roads:

!["Pedestrian"](/writeup/t3.png "t3")

|Sign|Probability|
|----|-----------|
|Keep right|0.80|
|Turn left ahead|0.17|
|Priority road|0.02|
|Go straight or right|0.01|
|Ahead only|0.00|

!["T3 Bar"](/writeup/t3-bar.png "T3-Bar")

For the fourth image, the model correctly predicts it as a priority Yeild sign and it does that with a 1.0 probability:

!["Yield"](/writeup/t4.png "t4")

|Sign|Probability|
|----|-----------|
|Yield|1.00|
|Speed limit (20km/h)|0.00|
|Speed limit (30km/h)|0.00|
|Speed limit (50km/h)|0.00|
|Speed limit (60km/h)|0.00|

!["T4 Bar"](/writeup/t4-bar.png "T4-Bar")

For the fifth image, the model correctly predicts it as a Road Work sign and it does that with a 1.0 probability:


!["Road Work"](/writeup/t5.png "t5")

|Sign|Probability|
|----|-----------|
|Road work|1.00|
|Bumpy road|0.00|
|Road narrows on the right|0.00|
|Bicycles crossing|0.00|
|Speed limit (20km/h)|0.00|

!["T5 Bar"](/writeup/t5-bar.png "T5-Bar")

## Network Visualization

Following is the first layer of conv net for "30 Speed Limit" real world image:

!["visual-t1"](/writeup/visual-t1.png)

Following is the first layer of conv net for "Priority Road" real world image:

!["visual-t2"](/writeup/visual-t2.png)

Following is the first layer of conv net for "Pedestrian" real world image:

!["visual-t3"](/writeup/visual-t3.png)

Following is the first layer of conv net for "Yield" real world image:

!["visual-t4"](/writeup/visual-t4.png)

Following is the first layer of conv net for "Road Worker" real world image:

!["visual-t5"](/writeup/visual-t5.png)



## Short Commings
