# Federated Learning Demo in Python using Socket Programming: Part 2

This is [Part 2](https://github.com/ahmedfgad/FederatedLearning/tree/master/TutorialProject/Part2) of the federated learning (FL) demo project in Python using socket programming. In this part, the server app created in [Part 1](https://github.com/ahmedfgad/FederatedLearning/tree/master/TutorialProject/Part1) is extended so that it can:

* Accept multiple connections at the same time using threading.
* Send and receive multiple messages within the same connection.

In [Part 3](https://github.com/ahmedfgad/FederatedLearning/tree/master/TutorialProject/Part3), a machine learning model is created using [PyGAD](https://pygad.readthedocs.io) at the server and then sent to the clients over the socket to be trained using the genetic algorithm. The trained model at the client is then sent back to the server.

# Project Files

The project has the following files:

- `server.py`: The server app. The server receives a text message from the client and replies with another text message.
- `client1.py`: A client app which sends a text message trains to the server and receives a text response.
- `client2.py`: A client app which sends a text message trains to the server and receives a text response.

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

After running the server, next is to run one or more clients. The project creates 2 clients but you can add more. 

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