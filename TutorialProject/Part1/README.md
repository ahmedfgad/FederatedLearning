# Federated Learning Demo in Python using Socket Programming: Part 1

This is [Part 1](https://github.com/ahmedfgad/FederatedLearning/tree/master/TutorialProject/Part1) of the federated learning (FL) demo project in Python using socket programming. In this part, a simple client-server app is created to send and receive text messages.

In [Part 2](https://github.com/ahmedfgad/FederatedLearning/tree/master/TutorialProject/Part2), the server app will be extended to allow accepting multiple connections at the same time in addition to sending and receiving multiple messages within the same connection.

# Project Files

The project has the following files:

- `server.py`: The server app. The server receives a text message from the client and replies with another text message.
- `client.py`: The client app which sends a text message trains to the server and receives a text response.

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

After running the server, next is to run the client using the `client.py` script. 

```
python client.py
```

For Mac/Linux, use `python3` rather than `python`:

```
python3 client.py
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