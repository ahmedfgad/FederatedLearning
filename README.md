# Federated Learning Demo in Python using Socket Programming

This is a demo project for applying the concepts of federated learning (FL) in Python using socket programming by building and training machine learning (ML) models using FL. The ML model is created using [PyGAD](https://pygad.readthedocs.io) which trains ML models using the genetic algorithm (GA). The problem used to demonstrate how things work is XOR.

The project builds GUI for the server and the client using [Kivy](https://kivy.org). This has a number of benefits.

- Easy way to manage the client application.
- Ability to make the client available in mobile devices because Kivy supports deploying its desktop apps into mobile apps. As a result, machine learning models could be trained using federated learning by the massive private data available in mobile devices. 

# Project Files

The project has the following files:

- `server.py`: The server Kivy app. It creates a model that is trained on the clients' devices using FL.
- `client1.py`: The client Kivy app which trains the model sent by the server using just 2 samples of the XOR problem.
- `client2.py`: Another client Kivy app that trains the server's model using the other 2 samples in the XOR problem.
- `pygad.py`: The implementation of the `pygad` module in the PyGAD library.
- `nn.py`: The implementation of the `pygad.nn` module in the PyGAD library which builds artificial neural networks (ANNs).
- `gann.py`: The implementation of the `pygad.gann` module in the PyGAD library which trains ANNs using GA.

# Running the Project

Start the project by running the `server.py` file. The GUI of the server Kivy app is shown below. Follow these steps to make sure the server is running and listening for connections.

* Click on the **Create Socket** button to create a socket. 

* Enter the IPv4 address and port number of the server's socket. `localhost` is used if both the server and the clients are running on the same machine. This is just for testing purposes. Practically, they run on different machines. Thus, the user need to specify the IPv4 address (e.g. 192.168.1.4).
* Click on the **Bind Socket** button to bind the create socket to the entered IPv4 address and port number.
* Click on the **Listen to Connections** button to start listening and accepting incoming connections. Each connected device receives the current model to be trained by its local data. Once the model is trained, then no more models will be sent to the connected devices.

![Fig01](https://user-images.githubusercontent.com/16560492/86205885-5af32380-bb6b-11ea-9ca6-149c0170e82b.png)

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

Just run any client and a GUI will appear like that. You can either run the client at a desktop or a mobile device.

Follow these steps to run the client:

* Click on the **Create Socket** button to create a socket. 

* Enter the IPv4 address and port number of the server's socket. If both the client and the server are running on the same machine, just use `localhost` for the IPv4 address. Otherwise, specify the IPv4 address (e.g. 192.168.1.4).s
* Click on the **Connect to Server** button to create a TCP connection with the server.
* Click on the **Receive & Train Model** button to ask the server to send its current ML model. The model will be trained by the client's local private data. The updated model will be sent back to the server. Once the model is trained, the message **Model is Trained** will appear.

![Fig03](https://user-images.githubusercontent.com/16560492/86206222-292e8c80-bb6c-11ea-9311-1ef4bb467188.jpg)

# For More Information

There are a number of resources to get started with Kivy.

## Book: [Building Android Apps in Python Using Kivy with Android Studio](https://www.amazon.com/Building-Android-Python-Using-Studio/dp/1484250303)

To get started with Kivy app development and how to built Android apps out of the Kivy app, check the book titled [Building Android Apps in Python Using Kivy with Android Studio](https://www.amazon.com/Building-Android-Python-Using-Studio/dp/1484250303)

[![kivy-book](https://user-images.githubusercontent.com/16560492/86205093-575e9d00-bb69-11ea-82f7-23fef487ce3c.jpg)](https://www.amazon.com/Building-Android-Python-Using-Studio/dp/1484250303)

## Tutorial: [Python for Android: Start Building Kivy Cross-Platform Applications](https://www.linkedin.com/pulse/python-android-start-building-kivy-cross-platform-applications-gad)

This tutorial titled [Python for Android: Start Building Kivy Cross-Platform Applications](https://www.linkedin.com/pulse/python-android-start-building-kivy-cross-platform-applications-gad) covers the steps for creating an Android app out of the Kivy app.

[![Kivy-Tutorial](https://user-images.githubusercontent.com/16560492/86205332-dfdd3d80-bb69-11ea-91fb-cb0143cb1e5e.png)](https://www.linkedin.com/pulse/python-android-start-building-kivy-cross-platform-applications-gad)

# Contact Us

- E-mail: [ahmed.f.gad@gmail.com](mailto:ahmed.f.gad@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
- [Amazon Author Page](https://amazon.com/author/ahmedgad)
- [Heartbeat](https://heartbeat.fritz.ai/@ahmedfgad)
- [Paperspace](https://blog.paperspace.com/author/ahmed)
- [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
- [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
- [GitHub](https://github.com/ahmedfgad)