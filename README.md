This repo contains the implementation of The LeNet, AlexNet, and GoogleNet CNN architectures for the recognition of 43 classes of traffic signs. These three models were implemented in succession. This workflow was designed with the purpose of incremental learning: starting with the most basic CNN architecture (LeNet) and working up towards a robust model (GoogleNet). The three models were programmed using TensorFlow 2 and slightly differ from their original forms. All three models were trained, tested, and evaluated on the GTSRB data set. KerasTuner, an easy-to-use, hyperparameter optimization framework that uses search algorithms to find the best hyperparameters, was used in this project.

# LeNet:

This project’s adoption of the LeNet Architecture can be found in the table below:
<img width="557" alt="Screen Shot 2022-08-24 at 3 07 49 PM" src="https://user-images.githubusercontent.com/56232251/186502760-2a7c0308-d180-422d-a761-4b2b6337a5e8.png">


Several deep learning techniques were introduced to improve the performance of the model:
- Dropout is a regularization method that removes network connections and forces hidden units to not co-adapt. Through trial and error, the best performance was achieved using a 30% dropout rate on two dense layers at the end of the network. 
- The original LeNet paper uses the sigmoid activation function. The sigmoid activation function often saturates, causing vanishing gradient descent and making it difficult for the optimization algorithm to adjust weights. To prevent vanishing gradients and for faster training, the ReLU activation function was used. 
- The stochastic gradient descent optimizer, with a batch size of 128 was used. Mini-batch gradient descent is advantageous as it produces stable error gradients and reliable convergence. 
- The KerasTuner library with early stopping found that a learning rate of 0.01 and 60 training epochs maximized test accuracy.


Results when evaluating the LeNet model: Training Accuracy: 98.36%, Test Accuracy: 91.05%, Validation Accuracy: 92.68%. The difference between training and test accuracy implies overfitting. The LeNet model is prone to overfitting due to the simplicity of the model. This can be confirmed from analyzing Figure 3. The AlexNet and GoogleNet models will introduce different methods to address the overfitting problem.

![LeNet (1)](https://user-images.githubusercontent.com/56232251/186503992-636c6475-cf1d-4395-8ad1-31dc75f48aa5.png)



# AlexNet
The AlexNet architeture is an extension of LeNet, but it is deeper and bigger. The performance of this AlexNet is significantly better than LeNet due to the use of techniques such as Batch Normalization. Minor changes were made to the original AlexNet architecture to account for the input and output sizes of the GTRSB dataset. The AlexNet architecture can be found below:
<img width="555" alt="Screen Shot 2022-08-24 at 3 22 30 PM" src="https://user-images.githubusercontent.com/56232251/186505263-4e633d27-7a78-40ef-b9e2-e3b564a12360.png">

- Batch Normalization was used after each convolution and pooling layer to reduce the internal covariate shift of the network, smoothen the loss functions, reduce overfitting, and speed up training. 
-Adam optimizer, overlapping pooling, and padding were introduced to improve the performance of the AlexNet model. The Adam optimizer is an efficient optimization method that combines the properties of Adagrad and RMSprop. 
- Adam is easy to implement, computationally efficient, and requires less memory. 
- Overlapping pooling slightly reduces overfitting, and describes the scenario when the stride is less than the pool size. 
- Padding mitigates information loss that comes with dimensionality reduction by ensuring border pixels are convolved more times. 
- The KerasTuner library with early stopping found the optimal learning rate (0.0001) and epoch choice (47) to maximize test accuracy. 

Results when evaluating the AlexNet model: Training Accuracy: 99.99%, Test Accuracy: 96.08%, Validation Accuracy: 97.07%. These results show a significant improvement over LeNet with little overfitting.

![Alex (1)](https://user-images.githubusercontent.com/56232251/186502008-8d9323a6-1671-4fa5-9e03-5d0f04413e65.png)

# GoogleNet
GoogLeNet was the winner of the 2014 ILSVRC. Researchers discovered there is a correlation between increasing layers within a network and performance gain. However, Large networks are prone to overfitting, costly to train, and are affected by the vanishing gradient problem. The large network problems were handled by GoogleNet through the innovation of Inception modules - a sparse neural network architecture that uses convolutions in parallel to make the network wider instead of deeper. The Inception module greatly reduces the parameter size of the network.

![inception](https://user-images.githubusercontent.com/56232251/186501742-c75beae9-e2b2-422b-820e-39dce081c580.jpg)


The GoogleNet consists of 24 layers, including nine Inception modules. GoogleNet uses 1x1 convolutions to reduce dimensionality while retaining features - this is an alternative to down sampling which suffers from information loss. The model is trained over 30 epochs, using the Adam optimizer and a learning rate of 1e-4. Max-pooling layers are inserted between some inception modules to reduce the input size between the Inception module. This is as an effective method for lessening the network’s computational load. A dropout layer(50%) is utilised just before the dense layer to further reduce overfitting. The GoogleNet layer-by-layer architecture can be seen below.

<img width="558" alt="Screen Shot 2022-08-24 at 3 24 21 PM" src="https://user-images.githubusercontent.com/56232251/186505563-ac644599-72f2-4066-8999-2fb061582b85.png">



Results when evaluating the GoogleNet model: Training Accuracy: 99.91%, Test Accuracy: 96.53%, Validation Accuracy: 97.26%. There is little difference between these accuracies, implying very little overfitting. GoogleNet and Alexnet significantly outperform LeNet for image classification. The training and testing accuracies of LeNet, AlexNet, and GoogleNet can be compared in Figure 3 below. 

![googlenet](https://user-images.githubusercontent.com/56232251/186502024-82b92495-c428-4e01-9d86-e3e7759c888c.png)


