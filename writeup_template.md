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

|Category|Name|Random Image|
| -------------------- | ------------------------------ |-------|
|0|Speed limit (20km/h)|!["0"](/writeup/signs/0-10108.jpg)|
|1|Speed limit (30km/h)|!["1"](/writeup/signs/1-3704.jpg)|
|2|Speed limit (50km/h)|!["2"](/writeup/signs/2-33414.jpg)|
|3|Speed limit (60km/h)|!["3"](/writeup/signs/3-5506.jpg)|
|4|Speed limit (70km/h)|!["4"](/writeup/signs/4-7856.jpg)|
|5|Speed limit (80km/h)|!["5"](/writeup/signs/5-13324.jpg)|
|6|End of speed limit (80km/h)|!["6"](/writeup/signs/6-21616.jpg)|
|7|Speed limit (100km/h)|!["7"](/writeup/signs/7-23730.jpg)|
|8|Speed limit (120km/h)|!["8"](/writeup/signs/8-16554.jpg)|
|9|No passing|!["9"](/writeup/signs/9-11859.jpg)|
|10|No passing for vehicles over 3.5 metric tons|!["10"](/writeup/signs/10-17918.jpg)|
|11|Right-of-way at the next intersection|!["11"](/writeup/signs/11-8806.jpg)|
|12|Priority road|!["12"](/writeup/signs/12-28311.jpg)|
|13|Yield|!["13"](/writeup/signs/13-23634.jpg)|
|14|Stop|!["14"](/writeup/signs/14-29512.jpg)|
|15|No vehicles|!["15"](/writeup/signs/15-30352.jpg)|
|16|Vehicles over 3.5 metric tons prohibited|!["16"](/writeup/signs/16-5183.jpg)|
|17|No entry|!["17"](/writeup/signs/17-30487.jpg)|
|18|General caution|!["18"](/writeup/signs/18-20982.jpg)|
|19|Dangerous curve to the left|!["19"](/writeup/signs/19-6766.jpg)|
|20|Dangerous curve to the right|!["20"](/writeup/signs/20-25999.jpg)|
|21|Double curve|!["21"](/writeup/signs/21-25934.jpg)|
|22|Bumpy road|!["22"](/writeup/signs/22-4586.jpg)|
|23|Slippery road|!["23"](/writeup/signs/23-1789.jpg)|
|24|Road narrows on the right|!["24"](/writeup/signs/24-10969.jpg)|
|25|Road work|!["25"](/writeup/signs/25-34699.jpg)|
|26|Traffic signals|!["26"](/writeup/signs/26-1460.jpg)|
|27|Pedestrians|!["27"](/writeup/signs/27-10489.jpg)|
|28|Children crossing|!["28"](/writeup/signs/28-26854.jpg)|
|29|Bicycles crossing|!["29"](/writeup/signs/29-10735.jpg)|
|30|Beware of ice/snow|!["30"](/writeup/signs/30-25289.jpg)|
|31|Wild animals crossing|!["31"](/writeup/signs/31-828.jpg)|
|32|End of all speed and passing limits|!["32"](/writeup/signs/32-10260.jpg)|
|33|Turn right ahead|!["33"](/writeup/signs/33-26398.jpg)|
|34|Turn left ahead|!["34"](/writeup/signs/34-20264.jpg)|
|35|Ahead only|!["35"](/writeup/signs/35-19451.jpg)|
|36|Go straight or right|!["36"](/writeup/signs/36-1147.jpg)|
|37|Go straight or left|!["37"](/writeup/signs/37-4892.jpg)|
|38|Keep right|!["38"](/writeup/signs/38-15053.jpg)|
|39|Keep left|!["39"](/writeup/signs/39-25490.jpg)|
|40|Roundabout mandatory|!["40"](/writeup/signs/40-4330.jpg)|
|41|End of no passing|!["41"](/writeup/signs/41-178.jpg)|
|42|End of no passing by vehicles over 3.5 metric tons|!["42"](/writeup/signs/42-9811.jpg)|

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
![yield](/writeup/yield.png "yield")

### Design and Test a Model Architecture

#### Preprocessing

1) Earlier, **YCbCr** color space was used thinking that this particular color space provides a more natural way of representing luminosity and image colors.

[More details on YCbCr](https://en.wikipedia.org/wiki/YCbCr)

2) But later on RGB color space was used directly and that proved to be much better than YCbCr.

2) Normalization: In order to reduce the effect of brightness on image classification, all of the images have been normalized before training. 

```
def normalize_rgb(data, mean = None, sigma = None):
    print('Normalizing in RGB')
    if mean == None or sigma == None:
        print("Computing mean")
        mean = np.mean(data)
        sigma = np.std(data)
        
    print('Before normalizing data[0,0,0,0]:', data[0,0,0])
        
    data = data.astype(np.float32) - mean
    data /= (sigma + 1e-7)
    
    print('After normalizing data[0,0,0,0]:', data[0,0,0])
    
    return data, mean, sigma
```

In order to normalize validation and testing sets, mean and standard deviation **are not** recomputed but the same ones that were computed from the training set is used.

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
![Shift](/writeup/9972.jpg)

The difference between the original data set and the augmented data set is the following:

* Number of original training examples = 34799
* Extra generated: (49981, 32, 32, 3)
* Combined training set: (84780, 32, 32, 3)

![Distribution Combined](/writeup/distribution_comb.png "combined distribution")

#### 2. Final Model Architecture

Final model consisted of the following layers:

| Layer         		|     Description	        					| 
|---------------------|---------------------------------------------| 
| Input         		| 32x32x3 RGB Normalized Image    			| 
| Convolution 5x5x3x6  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| 					                            |
| Max Pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5x6x16  | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU		            |           									|
| Max Pooling	        | 2x2 stride, outputs 5x5x16                    |
| Flatten   	        | Input: 5x5x16, Output: 400                    |
| Fully connected N1    | Input: 400, Output: 120                       |
| RELU		            |           									|
| DROPOUT               | Keep Probability: 0.5                         |
| Fully connected N2    | Input: 120, Output:  84                       |
| RELU		            |           									|
| DROPOUT               | Keep Probability: 0.5                         |
| Fully connected N3    | Input: 84, Output:  43                        |


#### 3. Model Training Parameters

To train the model, the following parameters were used:

| Component         	| Description    		                      | 
|-----------------------|---------------------------------------------| 
| EPOCH     | 400                        |
| Batch Size     | 128                        |
| Learning Rate     | 0.0008                        |
| Initial Weights   | Xavier Initializer                        |
| Dropout Keep Probability   | 0.5                        |
| Optimizer   | Adam Optimizer                        |
| Cost Function   | Softmax Cross Entropy                        |
| Evaluation Mechanism   | Best weights found in any EPOCH rather than last|


#### 4. Approach taken for finding the solution

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.982
* test set accuracy of 0.968

An iterative approach was chosen to find the solution:

##### Original LeNet

As a first step the original LeNet was used as that gives a good starting point. Only a maximum of about 0.90 accuracy on validation set was acheived. But training set accuracy was about 0.99 implying that it was overfitting.

##### Weight changes to mu and sigma

Keeping the original LeNet, played around with different mu, sigma, Epochs and batch sizes but the validation accuracy did not go up.

It was interesting to find out that batch gradient algorithms do not work that well if the batch size is very high. In this particular case I tried using batch size from 128 to almost equal to all of the input size. As I moved higher the validation accuracy kept going down instead of moving up

##### Changed to YCbCr instead of RGB

Since YCbCr has a specific channel for luminosity, I tried using this instead of plain RGB as it provides an easier way to average out the brightness. I thought the network would have a better effect with the luminosity in a different channel and I would be normalizing the Y channel separately from the CbCr color channels. But this did not prove to be a good idea since it is clear from the following that some layers of C1 were not doing anything at all:

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

#### Changed back to RGB

Changed back to RGB and the same C1 layer has the following outlook now:


Following is the first layer of conv net for "30 Speed Limit" real world image:

!["visual-t1"](/writeup/visual-t1-rgb.png)

Following is the first layer of conv net for "Priority Road" real world image:

!["visual-t2"](/writeup/visual-t2-rgb.png)

Following is the first layer of conv net for "Pedestrian" real world image:

!["visual-t3"](/writeup/visual-t3-rgb.png)

Following is the first layer of conv net for "Yield" real world image:

!["visual-t4"](/writeup/visual-t4-rgb.png)

Following is the first layer of conv net for "Road Worker" real world image:

!["visual-t5"](/writeup/visual-t5-rgb.png)


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

#### Dropout in the second Network Layer Only

Dropout layers were then kept in fully connected layers only. This reduced underfitting that resulted from dropout in convnet layers and then achieved the desired 0.93 result in validation set.

Poor performance on the loss function was noticed when only one dropout is used:

!["One Dropout Accuracy"](/writeup/one-dropout-accuracy.png "1 Accuracy")
!["One Dropout Loss"](/writeup/one-dropout-loss.png "1 Loss")

#### Dropout in the first & second Network Layers

With two dropouts the accuracy and loss were much better:

!["Two Dropout Accuracy"](/writeup/two-dropout.png "2 Accuracy")
!["Two Dropout Loss"](/writeup/two-dropout-loss.png "2 Loss")


#### Final Result on Validation Set

* training set accuracy of 1.000
* validation set accuracy of 0.982
* test set accuracy of 0.968

Since it is achieving > 93% on test set, it seems to be performing quite reasonable.


### Why Do I believe LeNet is 'OK' for traffic sign classification BUT not the best

Well to be honest, I don't think LeNet is the best solution for this. Recent papers clearly indicate better performance from GoogLeNet or Resnet on ImageNet so chances are that they would be better than LeNet on traffic sign classification as well.

However, LeNet is sufficient enough for this assignment as it is a much smaller network (computationally) comapred to recent other methods and provides quite good performance as well.

## Test a Model on New Images

#### 1. Real life german traffic sign images:

Instead of choosing from the web a fellow Udacity student, Sonja Krause-Harder, provided real life images. Five traffic signs were  cropped out of these images:

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
| 30 Zone      		    | Speed limit (50km/h) | 
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

|Sign|Probability|
|----|-----------|
|Speed limit (50km/h)|1.00|
|Speed limit (30km/h)|0.00|
|Wild animals crossing|0.00|
|Speed limit (100km/h)|0.00|
|Speed limit (80km/h)|0.00|

!["30 Bar"](/writeup/30-bar.png "30-bar")

For the Second image, the model correctly predicts it as a priority road:

!["Priority"](/writeup/t2.png "t2")

|Sign|Probability|
|----|-----------|
|Priority road|1.00|
|Traffic signals|0.00|
|End of no passing by vehicles over 3.5 metric tons|0.00|
|Speed limit (20km/h)|0.00|
|Speed limit (30km/h)|0.00|

!["T2 Bar"](/writeup/t2-bar.png "T2-Bar")

For the third image, the model incorrectly predicts it as keep right, which was expected since this image was never part of the training set however it was found as a real life sign on German roads:

!["Pedestrian"](/writeup/t3.png "t3")

|Sign|Probability|
|----|-----------|
|Turn left ahead|0.79|
|Go straight or right|0.11|
|Ahead only|0.10|
|Roundabout mandatory|0.00|
|End of no passing|0.00|

!["T3 Bar"](/writeup/t3-bar.png "T3-Bar")

For the fourth image, the model correctly predicts it as a Yield sign and it does that with a 1.0 probability:

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
|Speed limit (20km/h)|0.00|
|Speed limit (30km/h)|0.00|
|Speed limit (50km/h)|0.00|


!["T5 Bar"](/writeup/t5-bar.png "T5-Bar")

## Network Visualization

Following is the first layer of conv net for "30 Speed Limit" real world image:

!["visual-t1"](/writeup/visual-t1-rgb.png)

Following is the first layer of conv net for "Priority Road" real world image:

!["visual-t2"](/writeup/visual-t2-rgb.png)

Following is the first layer of conv net for "Pedestrian" real world image:

!["visual-t3"](/writeup/visual-t3-rgb.png)

Following is the first layer of conv net for "Yield" real world image:

!["visual-t4"](/writeup/visual-t4-rgb.png)

Following is the first layer of conv net for "Road Worker" real world image:

!["visual-t5"](/writeup/visual-t5-rgb.png)


## Short Commings

* The real world image of "30 Zone", which was a little different from the training set "30 Zone" images, should at least have the correct category in the top 5 but it failed to even come close. More investigation needs to be done to see where the bug is in the convolutional layers or maybe it is the way the image was shortened to 32x32

* The network comes close to merging but it never fully merges and there is some variation even towards the end of 400th EPOCH
