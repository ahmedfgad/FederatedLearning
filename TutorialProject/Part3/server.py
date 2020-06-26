import socket
import pickle
import threading
import time

import pygad
import pygad.nn
import pygad.gann
import numpy

model = None

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[1, 1],
                           [1, 0],
                           [0, 1],
                           [0, 0]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([0, 
                            1, 
                            1, 
                            0])

num_classes = 2
num_inputs = 2

num_solutions = 6
GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=num_inputs,
                                num_neurons_hidden_layers=[2],
                                num_neurons_output=num_classes,
                                hidden_activations=["relu"],
                                output_activation="softmax")

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if data == b'': # Nothing received from the client.
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0 # 0 means the connection is no longer active and it should be closed.

                elif str(data)[-2] == '.':
                    print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def model_averaging(self, model, other_model):
        model_weights = pygad.nn.layers_weights(last_layer=model, initial=False)
        other_model_weights = pygad.nn.layers_weights(last_layer=other_model, initial=False)
        
        new_weights = (model_weights + other_model_weights)/2

        pygad.nn.update_layers_trained_weights(last_layer=model, final_weights=new_weights)

    def reply(self, received_data):
        global GANN_instance, data_inputs, data_outputs, model
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                subject = received_data["subject"]
                print("Client's Message Subject is {subject}.".format(subject=subject))

                print("Replying to the Client.")
                if subject == "echo":
                    try:
                        data = {"subject": "model", "data": GANN_instance}
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                elif subject == "model":
                    try:
                        GANN_instance = received_data["data"]
                        best_model_idx = received_data["best_solution_idx"]

                        best_model = GANN_instance.population_networks[best_model_idx]
                        if model is None:
                            model = best_model
                        else:
                            self.model_averaging(model, best_model)

                        # print(best_model.trained_weights)
                        # print(model.trained_weights)

                        predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                        print("Model Predictions: {predictions}".format(predictions=predictions))

                        error = numpy.sum(numpy.abs(predictions - data_outputs))
                        print("Error = {error}".format(error=error))

                        if error != 0:
                            data = {"subject": "model", "data": GANN_instance}
                            response = pickle.dumps(data)
                        else:
                            data = {"subject": "done", "data": None}
                            response = pickle.dumps(data)

                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                else:
                    response = pickle.dumps("Response from the Server")
                            
                try:
                    self.connection.sendall(response)
                except BaseException as e:
                    print("Error Sending Data to the Client: {msg}.\n".format(msg=e))

            else:
                print("The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {d_keys}.".format(d_keys=received_data.keys()))
        else:
            print("A dictionary is expected to be received from the client but {d_type} received.".format(d_type=type(received_data)))

    def run(self):
        print("Running a Thread for the Connection with {client_info}.".format(client_info=self.client_info))

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
            print(date_time)
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break

            # print(received_data)
            self.reply(received_data)

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

# Timeout after which the socket will be closed.
# soc.settimeout(5)

soc.bind(("localhost", 10000))
print("Socket Bound to IPv4 Address & Port Number.\n")

soc.listen(1)
print("Socket is Listening for Connections ....\n")

all_data = b""
while True:
    try:
        connection, client_info = soc.accept()
        print("New Connection from {client_info}.".format(client_info=client_info))
        socket_thread = SocketThread(connection=connection,
                                     client_info=client_info, 
                                     buffer_size=1024,
                                     recv_timeout=10)
        socket_thread.start()
    except:
        soc.close()
        print("(Timeout) Socket Closed Because no Connections Received.\n")
        break
