# Comparing-Classifiers-like-LDA-SVM-and-DTC

## Abstract 

In this assignment, we will be having a deeper dive into the SVD matrixes and performing PCA decomposition. Then, we will be taking a look at a modern classifier, Linear Discriminate Analysis. Also, we will be taking a look at older classifiers like Decision Tree Classifiers and Support Vector Machines. We will be using the digit dataset, MNIST_784 which holds writings of the digits 0 to 9 with different hand styles. We will be using the classifiers to identify each drawing of images with the labels 0 to 9. 

## Section 1: Introduction

The MNIST_786 dataset is made up of 70,000 images of digits 0 to 9. Each image is condensed into a 28 by 28 pixel image which is converted into a column of pixels that is 784 in length. The goal of this assignment is to find a way to identify clusters of images and figure out which digit they belong to. The classifiers we will be using are Linear Discriminate Analysis (LDA), Decision Tree Classifiers (DTC), and Support Vector Machines (SVM). The following narrative will explain the performance and accuracy of each classifier. 

## Section 2: Theoretical Background

The dataset is converted so that the image matrix, X.data, is used in SVD. The three outputs of the SVD are U, s, and V. The s stands for the sigma portion of the SVD. The interpretation of each individual matrix based on our code is that the U is the eigenvector of the AAT and the left single vectors of matrix A which is the main image matrix. In this case, the U matrix shape is a (70000, 784) which makes sense since the U shape should match the A matrix. The  V matrix is the right single vector of matrix A and is the eigenvector of the ATA matrix. The shape of V should be a square matrix and the resulting matrix we got was (784, 784). The s or sigma matrix is the singular value of the SVD. In this case, we should be getting 784 singular values which the code says it does. 

Also, to understand the classifiers, we should take a look at the older ones, SVM and DTC. The SVM process is by creating a plane across the data set. It would then find the best plane that can separate the data set based on the type of data set. In this case, we want to separate two digits so a hyperplane would be created until we find the best scoring plane. 

The DTC creates a line across a graph for each individual feature. It would then take the highest accuracy and separate the data set into two branches. Then, the classifier would take the two newly separated data sets and creates a boundary between the two data with the next highest accurate feature. The process continues until we run out of features. The result would be a tree that separates the data sets into two if we needed two labels.

Finally, LDA is a common classifier used in machine learning and produces a linear line in PCA space. The dataset is in PCA space and the LDA would fit a linear line that can separate the dataset as best as possible. We will be using LDA and comparing it to the older classifiers.

## Section 3: Algorithm, Implementation, and Development

To convert the dataset for SVD, I would grab the data of X which is just X = mnist.data and I would also need to grab the labels which are just Y = mnist.target. Then, I would use the X matrix for the SVD code using np.linalg.svd(). The output is U, s, and V where s is the sigma or singular value. Afterward, the assignment ask us to graph the singular values and I used the stem plot to visualize the singular values. We will discuss the best number of modes to reconstruct the images. Also, we discussed the interpretation of the U, s, and V matrix in the theoretical discussion. 

Then, the assignment asks us to project a 3D plot of three V modes. I would need to do a dot product of the X matrix and a transpose version of the V matrix. The output should be a matrix of V-mode projections. I would then use the scatter plot to plot the 3D graph by using the 2nd column of the V projection matrix as the x-axis, the 3rd column of the V projection matrix as the y-axis, and the 5th column of the V projection matrix as the z-axis. Then, the Y labels are used to color the data based on the labels so there will be 10 colors. The result will be a clump of data points with 10 colors in 3D space. 

Next, I build a function that outputs the training and testing values of X and Y based on the digits we are looking for. The first function, classifier2, separates two digits in the MNIST data set, and the second function, classifier3, separate three digits. The following code is posted in Figure 1.  We can see that I filtered the digits with the digits we need to separate and split the data into training and testing data. The number of test sizes is set to 0.3 which means that 30% of the data is used for testing. In the end, the function returns the X_training, y_training, X_testing, and y_testing. 

```python
# returns training and testing data for two digits separation
# the input parameters requires two digits
# the targeting data is turn into int and
# the X and Y are filtered to only the two digits
# the data set is then split into training and testing
def classifer2(digit1, digit2):
    y1 = Y.astype(int)
    X1 = X[(y1 == digit1) | (y1 == digit2)]
    y1 = Y[(y1 == digit1) | (y1 == digit2)]
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

# returns training and testing data for three digits separation
# the input parameters requires three digits
# the targeting data is turn into int and
# the X and Y are filtered to only the three digits
# the data set is then split into training and testing
def classifer3(digit1, digit2, digit3):
    y1 = Y.astype(int)
    X1 = X[(y1 == digit1) | (y1 == digit2) | (y1 == digit3)]
    y1 = Y[(y1 == digit1) | (y1 == digit2) | (y1 == digit3)]
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
    
```
**Figure 1**: Python code for the classifier for two digits and three digits that return training and testing data.

The assignment asks us to separate every two digits in the MNIST data set, therefore, there should be 45 combinations which I created using an array of pairs shown in Figure 2. Also, each classifier as an array of accuaracy scores shown in Figure 2. Then, I used the LDA, for loops, and the classifier2 function to create an array of accuracy scores where the index matches the position of the combination array of every two digits. I did the same process for both the DTC and SVM. 

```python
# creates an array of different combinations of two digits
comb = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],
        [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],
        [2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],
        [3,4],[3,5],[3,6],[3,7],[3,8],[3,9],
        [4,5],[4,6],[4,7],[4,8],[4,9],
        [5,6],[5,7],[5,8],[5,9],
        [6,7],[6,8],[6,9],
        [7,8],[7,9],
        [8,9]]
```
```python
scoreArrayLDA = []
```
```python
scoreArrayDTC = []
```
```python
scoreArraySVM = []
```
**Figure 2**: The array of combination of two digits and the array of each classifier scores

Afterward, we can start comparing the easiest digits to separate and the hardest digits to separate. I used the np.max() code and the np.min() to find the easiest and hardest digits to separate since the output will be the percentage of accuracy. I would print the accuracy score and the two digits that were easier and harder to separate. The results will be listed in the results portion of the assignment. 

Lastly, the assignment didn’t ask to visualize any of the classifiers but to help with understanding why the LDA has one pair of digits easier to separate compared to the other ones. I needed to create a 2D plot of two LDA components. However, I was having problems with just filtering two digits from the X data set so I would need to add another number so that the LDA allows the n components to be 2. With these numbers, I used the transform function for the classifier and plot the data with a scatter plot. The output is three groups of digits but in the result portion of the narrative, I would need to ignore the third digit and compare the first and second digits. I will explain why the digits were easier to separate than others with LDA in the results portion. 

## Section 4: Results

In Figure 3, we can see the stem graph of the singular values. The assignment asks us to identify the modes that can produce a good reconstruction of the images. In this case, I decided to find the 90 percent variance of the singular values. I used a while loop to find the number of modes that meets the 90 percent variance of the singular values. The resulting modes that are good enough for the reconstruction of the images are 54 modes. 

![Figure 3](https://github.com/SamQLuong/Comparing-Classifiers-like-LDA-SVM-and-DTC/blob/main/SVD%20Spectrum.png)

**Figure 3**: The singular value spectrum graph using a stem plot.

In Figure 4, we can see the 3D plot of the three V-mode projections. The graph shows that the data is clumped into a single ball of datasets. We can see that the dataset is labeled with a clear area where the digits are separated. However, there are some digits that have datasets that blend together into one area.

![Figure 4](https://github.com/SamQLuong/Comparing-Classifiers-like-LDA-SVM-and-DTC/blob/main/Vmode%20Projection.png)

**Figure 4**: The 3D plot of the V-mode projection of column 2, 3, and 5. 

I timed the classifiers to see how long each classifier would take to find the accuracy score for each combination of the two digits. The LDA took about **81 seconds** to complete while the DTC took about **66 seconds**. However, the SVM took the longest time to complete which was about **203 seconds**. Now, the easiest digits to separate based on the LDA classifier was the 6 and 7 digit which makes sense because the 6 and 7 are clearly different. However, the hardest digits to separate were 3 and 5 because the digits are similar in shape. The score for 6 and 7 was about **99.5%** for LDA, and the score for 3 and 5 was about **94.8%**. Comparing the score to the other classifiers, the DTC score for 6 and 7 was **99.4%**, and for 3 and 5 was **95.4%**. The SVM scoring for 6 and 7 was **100%** while the 3 and 5 were **99.2%**. We can see that the SVM has the highest score for the hardest and easiest digits while the LDA and DTC were similar in percentages but the DTC took the shortest amount of time to compute. 
The following code result in figure 5 is the results from performing all three classifiers
```
Max Index:  39
Min Index:  25
DTC Max:  0.9924723594448365
SVM Max:  1.0
DTC Min:  0.9578895219222194
SVM Min:  0.9920733217735943
LDA Max Score:  0.9955304634203717
Easiest digits to Separate:  [6, 7]
LDA Min Score:  0.9484765915283626
Hardest digits to Separate:  [3, 5]
```
```
Length of Time for LDA:  80.85871005058289
```
```
Length of Time for DTC:  66.19771194458008
```
```
Length of Time for SVM:  203.96178460121155
```
Finally, in Figure 6, we can see the LDA graphs for both the hardest and easiest digits to separate. The yellow portion of the graph is the added digit so that we can graph the LDA in a 2D plot. Therefore, ignoring the yellow portion, we can see that the 6 and 7 digits have a small gap between the two clumps of data. Therefore, the accuracy score is higher because the LDA was able to separate the clumps easily. On the other hand, we can see that the 3 and 5 have a few data points clumped together which would make the classifier have a hard time separating the digits. 

![Figure 6 part 1](https://github.com/SamQLuong/Comparing-Classifiers-like-LDA-SVM-and-DTC/blob/main/LDA%20Easiest%20Separation.png)
![Figure 6 part 2](https://github.com/SamQLuong/Comparing-Classifiers-like-LDA-SVM-and-DTC/blob/main/LDA%20Hardest%20Separation.png)

**Figure 5**: The top graph is the easiest digits to separate where digit 6 and 7 and an added 8 for the 2D plot is shown. The bottom graph is the hardest digits to separate where the digit 3 and 5 and an added 8 for the 2D plot is shown. 

## Section 5: Conclusion

The three classifiers clearly have several differences in their performance. The SVM took the longest time to compute but provides the highest accuracy in scoring. The high accuracy can be great for smaller data sets because the length of time it takes is more than double the time it takes compared to DTC. The LDA and DTC have similar accuracy scores but the DTC has a higher score for the hardest-to-separate digits. We can infer that the classifier was able to separate the hardest digits easier than the LDA.  However, DTC is expensive to run so the LDA can be used for easier use of the classifier because of the similar accuracy scores. 




