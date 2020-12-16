# SiameseNet for MNIST Dataset
Learn the Siamese Net with MNIST and Tensorflow 2.3 and Python 3.8

### prerequisite 
Creating this network to learn about the basic siamese net.
The requirement for running the code:
```
pip install tensorflow-gpu==2.3.1
```

If you do not have the dedicated GPU to run the code:
```
pip install tensorflow==2.3.1
```

Some libraries that need to be included:
```
pip install idx2numpy
pip install numpy
pip install matplotlib
```

### How to run the code
To execute the code, you may enter the following command:
```
python test.py
```

To test input of two values using Siamese network, enter the following command:
```
python customTest.py
```
Inside the folder of "TestData" would have several samples for you to try. 
In addition, it is possible to include your own data but need to rename the variable inside the "CustomTest.py".

### Design and Explaination
This is based of a simple design of CNN network with 2 layers (convolutional layer + max pooling layer).
The activation of the CNN layer is 1120 embedded features where serve as legs for the Siamese network.

As for the Siamese network, we would have 2 legs, which are denoted as left leg and right leg connected one dense layer with only 1 vector output.
The vector ouput a floating number, which 0.0 denotes both pair of data not similar and 1.0 represents highly similar.

The model will save a copy of ".h5" model in the "model" folder for further usage.


### Other works that can be referred
I have tried some of the other works but only some of them works due to different environment requiremnet. 
So I highly suggesting you guys using Anaconda/Enthought to create a testing environment for testing out the code.

https://github.com/ywpkwon/siamese_tf_mnist

https://github.com/vnherdeiro/siamese-mnist

https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning
