# Federated Learning Demo in Python using Socket Programming

Part 3 of the demo project which trains a [PyGAD](https://pygad.readthedocs.io) model using the genetic algorithm. 

Within this part, the server creates a model and send it to the connected clients. The model is trained at the clients using their local data and sent back to the server. The server may ask the clients to retrain the model again until reaching a desired accuracy/error.