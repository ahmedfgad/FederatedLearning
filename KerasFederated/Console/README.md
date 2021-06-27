# Federated Learning using Keras and PyGAD

Training a Keras model using the genetic algorithm ([PyGAD](https://pygad.readthedocs.io)) using federated learning of multiple clients.

To know more about training Keras models using [PyGAD](https://pygad.readthedocs.io), please read this tutorial: [How To Train Keras Models Using the Genetic Algorithm with PyGAD](https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad)

# Project Files

The project has the following files:

- `server.py`: The server app. It creates a Keras model that is trained on the clients' devices using FL with PyGAD.
- `client1.py`: The client app which trains the Keras model sent by the server using just 2 samples of the XOR problem.
- `client2.py`: Another client app that trains the server's Keras model using the other 2 samples in the XOR problem.

# Install PyGAD

Before running the project, the [PyGAD](https://pygad.readthedocs.io) library must be installed.

```
pip install pygad
```

# Running the Project

Start the project by running the [`server.py`](https://github.com/ahmedfgad/FederatedLearning/blob/master/KerasFederated/Console/server.py) file. Please use appropriate IPv4 and port number.

After running the server, next is to run one or more clients. The project creates 2 clients but you can add more. The only expected change among the different clients is the data being used for training the model sent by the server.

For [`client1.py`](https://github.com/ahmedfgad/FederatedLearning/blob/master/KerasFederated/client1.py), here is the training data (2 samples of the XOR problem):

```python
# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[0, 1],
                           [0, 0]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([[0, 1], 
                            [1, 0]])
```

Here is the training data (other 2 samples of the XOR problem) for the other client ([`client2.py`](https://github.com/ahmedfgad/FederatedLearning/blob/master/KerasFederated/Console/client2.py)):

```python
# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[1, 0],
                           [1, 1]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([[0, 1], 
                            [1, 0]])
```

# For More Information

There are a number of resources to get started with federated learning and Kivy.

## Tutorial: [Introduction to Federated Learning](https://heartbeat.fritz.ai/introduction-to-federated-learning-40eb122754a2)

This tutorial describes the pipeline of training a machine learning model using federated learning.

[![](https://miro.medium.com/max/3240/1*6gRmlrDPp5J42HR3QWLYew.jpeg)](https://heartbeat.fritz.ai/introduction-to-federated-learning-40eb122754a2)

## Tutorial: [How To Train Keras Models Using the Genetic Algorithm with PyGAD](https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad)

Use [PyGAD](https://pygad.readthedocs.io), a Python 3 easy-to-use genetic algorithm library, to train Keras models using the genetic algorithm. The tutorial is detailed to explain all the steps needed to build and train the model.

[![](https://user-images.githubusercontent.com/16560492/101267295-c74c0180-375f-11eb-9ad0-f8e37bd796ce.png)](https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad)

## Tutorial: [Breaking Privacy in Federated Learning](https://heartbeat.fritz.ai/breaking-privacy-in-federated-learning-77fa08ccac9a)

Even that federated learning does not disclose the private user data, there are some cases in which the privacy of federated learning can be broken.

[![](https://miro.medium.com/max/3240/1*nZQg-E4a1wOvIH2AmkUUsQ.jpeg)](https://heartbeat.fritz.ai/breaking-privacy-in-federated-learning-77fa08ccac9a)

## Tutorial: [Python for Android: Start Building Kivy Cross-Platform Applications](https://www.linkedin.com/pulse/python-android-start-building-kivy-cross-platform-applications-gad)

This tutorial titled [Python for Android: Start Building Kivy Cross-Platform Applications](https://www.linkedin.com/pulse/python-android-start-building-kivy-cross-platform-applications-gad) covers the steps for creating an Android app out of the Kivy app.

[![Kivy-Tutorial](https://user-images.githubusercontent.com/16560492/86205332-dfdd3d80-bb69-11ea-91fb-cb0143cb1e5e.png)](https://www.linkedin.com/pulse/python-android-start-building-kivy-cross-platform-applications-gad)

## Book: [Building Android Apps in Python Using Kivy with Android Studio](https://www.amazon.com/Building-Android-Python-Using-Studio/dp/1484250303)

To get started with Kivy app development and how to built Android apps out of the Kivy app, check the book titled [Building Android Apps in Python Using Kivy with Android Studio](https://www.amazon.com/Building-Android-Python-Using-Studio/dp/1484250303)

[![kivy-book](https://user-images.githubusercontent.com/16560492/86205093-575e9d00-bb69-11ea-82f7-23fef487ce3c.jpg)](https://www.amazon.com/Building-Android-Python-Using-Studio/dp/1484250303)

# Contact Us

- E-mail: [ahmed.f.gad@gmail.com](mailto:ahmed.f.gad@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
- [Amazon Author Page](https://amazon.com/author/ahmedgad)
- [Heartbeat](https://heartbeat.fritz.ai/@ahmedfgad)
- [Paperspace](https://blog.paperspace.com/author/ahmed)
- [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
- [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
- [GitHub](https://github.com/ahmedfgad)