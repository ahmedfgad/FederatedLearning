# Federated Learning Demo in Python using Socket Programming: Part 3

This is [Part 3](https://github.com/ahmedfgad/FederatedLearning/tree/master/TutorialProject/Part3) of the federated learning (FL) demo project in Python using socket programming. In this part, [PyGAD](https://pygad.readthedocs.io) is used to create a ML model at the server which is then sent to the clients to be trained using the genetic algorithm (GA). The problem used to demonstrate how things work is XOR.

In [Part 4](https://github.com/ahmedfgad/FederatedLearning/tree/master/TutorialProject/Part4), a GUI is created using Kivy for both the server and the client apps. Moreover, both the server and client apps will be made available for Android.

# Project Files

The project has the following files:

- `server.py`: The server app. It creates a model that is trained on the clients' devices using FL.
- `client1.py`: A client app which trains the model sent by the server using just 2 samples of the XOR problem.
- `client2.py`: Another client app that trains the server's model using the other 2 samples in the XOR problem.

# Install PyGAD

The project uses the [PyGAD](https://pypi.org/project/pygad) library for building and training the ML model. To install [PyGAD](https://pypi.org/project/pygad), simply use pip to download and install the library from [PyPI](https://pypi.org/project/pygad) (Python Package Index). The library lives a PyPI at this page https://pypi.org/project/pygad.

For Windows, issue the following command:

```
pip install pygad
```

For Linux and Mac, replace `pip` by use `pip3` because the library only supports Python 3.

```
pip3 install pygad
```

PyGAD is developed in Python 3.7.3 and depends on NumPy for creating and manipulating arrays and Matplotlib for creating figures. The exact NumPy version used in developing PyGAD is 1.16.4. For Matplotlib, the version is 3.1.0.

To get started with PyGAD, please read the documentation at [Read The Docs](https://pygad.readthedocs.io/) [https://pygad.readthedocs.io](https://pygad.readthedocs.io/).

# Running the Project

Start the project by running the `server.py` file from the terminal using the following command:

```
python server.py
```

For Mac/Linux, use `python3` rather than `python`:

```
python3 server.py
```

After running the server, next is to run one or more clients. The project creates 2 clients but you can add more. The only expected change among the different clients is the data being used for training the model sent by the server.

For `client1.py`, here is the training data (2 samples of the XOR problem):

```python
# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[0, 1],
                           [0, 0]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([1, 
                            0])
```

Here is the training data (other 2 samples of the XOR problem) for the other client (`client2.py`):

```python
# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[1, 0],
                           [1, 1]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([1, 
                            0])
```

To run a client, simply issue the following terminal command while replacing `<client-script-name>.py` by the client's script name.

```
python <client-script-name>.py
```

For Mac/Linux, use `python3` rather than `python`:

```
python3 <client-script-name>.py
```

# Contact Us

- E-mail: [ahmed.f.gad@gmail.com](mailto:ahmed.f.gad@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
- [Amazon Author Page](https://amazon.com/author/ahmedgad)
- [Heartbeat](https://heartbeat.fritz.ai/@ahmedfgad)
- [Paperspace](https://blog.paperspace.com/author/ahmed)
- [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
- [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
- [GitHub](https://github.com/ahmedfgad)