import socket
import pickle
import threading
import time
import numpy

import nn
import gann

import kivy.app
import kivy.uix.button
import kivy.uix.label
import kivy.uix.textinput
import kivy.uix.boxlayout

class ServerApp(kivy.app.App):
    
    def __init__(self):
        super().__init__()

    def create_socket(self, *args):
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.label.text = "Socket Created"

        self.create_socket_btn.disabled = True
        self.bind_btn.disabled = False
        self.close_socket_btn.disabled = False

    def bind_socket(self, *args):
        ipv4_address = self.server_ip.text
        port_number = self.server_port.text
        self.soc.bind((ipv4_address, int(port_number)))
        self.label.text = "Socket Bound to IPv4 & Port Number"

        self.bind_btn.disabled = True
        self.listen_btn.disabled = False

    def listen_accept(self, *args):
        self.soc.listen(1)
        self.label.text = "Socket is Listening for Connections"

        self.listen_btn.disabled = True

        self.listenThread = ListenThread(kivy_app=self)
        self.listenThread.start()

    def close_socket(self, *args):
        self.soc.close()
        self.label.text = "Socket Closed"

        self.create_socket_btn.disabled = False
        self.bind_btn.disabled = True
        self.listen_btn.disabled = True
        self.close_socket_btn.disabled = True

    def build(self):
        self.create_socket_btn = kivy.uix.button.Button(text="Create Socket", disabled=False)
        self.create_socket_btn.bind(on_press=self.create_socket)

        self.server_ip = kivy.uix.textinput.TextInput(hint_text="IPv4 Address", text="localhost")
        self.server_port = kivy.uix.textinput.TextInput(hint_text="Port Number", text="10000")

        self.server_socket_box_layout = kivy.uix.boxlayout.BoxLayout(orientation="horizontal")
        self.server_socket_box_layout.add_widget(self.server_ip)
        self.server_socket_box_layout.add_widget(self.server_port)

        self.bind_btn = kivy.uix.button.Button(text="Bind Socket", disabled=True)
        self.bind_btn.bind(on_press=self.bind_socket)

        self.listen_btn = kivy.uix.button.Button(text="Listen to Connections", disabled=True)
        self.listen_btn.bind(on_press=self.listen_accept)

        self.close_socket_btn = kivy.uix.button.Button(text="Close Socket", disabled=True)
        self.close_socket_btn.bind(on_press=self.close_socket)

        self.label = kivy.uix.label.Label(text="Socket Status")

        self.box_layout = kivy.uix.boxlayout.BoxLayout(orientation="vertical")

        self.box_layout.add_widget(self.create_socket_btn)
        self.box_layout.add_widget(self.server_socket_box_layout)
        self.box_layout.add_widget(self.bind_btn)
        self.box_layout.add_widget(self.listen_btn)
        self.box_layout.add_widget(self.close_socket_btn)
        self.box_layout.add_widget(self.label)

        return self.box_layout

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
GANN_instance = gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=num_inputs,
                                num_neurons_hidden_layers=[2],
                                num_neurons_output=num_classes,
                                hidden_activations=["relu"],
                                output_activation="softmax")

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, kivy_app, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout
        self.kivy_app = kivy_app

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
                    self.kivy_app.label.text = "All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data))

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            self.kivy_app.label.text = "Error Decoding the Client's Data"
                            return None, 0

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                self.kivy_app.label.text = "Error Receiving Data from the Client"
                return None, 0

    def model_averaging(self, model, other_model):
        model_weights = nn.layers_weights(last_layer=model, initial=False)
        other_model_weights = nn.layers_weights(last_layer=other_model, initial=False)

        new_weights = numpy.array(model_weights + other_model_weights)/2

        nn.update_layers_trained_weights(last_layer=model, final_weights=new_weights)

    def reply(self, received_data):
        global GANN_instance, data_inputs, data_outputs, model
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                subject = received_data["subject"]
                print("Client's Message Subject is {subject}.".format(subject=subject))
                self.kivy_app.label.text = "Client's Message Subject is {subject}".format(subject=subject)

                print("Replying to the Client.")
                self.kivy_app.label.text = "Replying to the Client"
                if subject == "echo":
                    if model is None:
                        data = {"subject": "model", "data": GANN_instance}
                    else:
                        predictions = nn.predict(last_layer=model, data_inputs=data_inputs)
                        error = numpy.sum(numpy.abs(predictions - data_outputs))
                        # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                        if error == 0:
                            data = {"subject": "done", "data": None}
                        else:
                            data = {"subject": "model", "data": GANN_instance}

                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {msg}.\n".format(msg=e))
                        self.kivy_app.label.text = "Error Encoding the Message"
                elif subject == "model":
                    try:
                        GANN_instance = received_data["data"]
                        best_model_idx = received_data["best_solution_idx"]

                        best_model = GANN_instance.population_networks[best_model_idx]
                        if model is None:
                            model = best_model
                        else:
                            predictions = nn.predict(last_layer=model, data_inputs=data_inputs)
    
                            error = numpy.sum(numpy.abs(predictions - data_outputs))
    
                            # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                            if error == 0:
                                data = {"subject": "done", "data": None}
                                response = pickle.dumps(data)
                                return

                            self.model_averaging(model, best_model)

                        # print(best_model.trained_weights)
                        # print(model.trained_weights)

                        predictions = nn.predict(last_layer=model, data_inputs=data_inputs)
                        print("Model Predictions: {predictions}".format(predictions=predictions))

                        error = numpy.sum(numpy.abs(predictions - data_outputs))
                        print("Prediction Error = {error}".format(error=error))
                        self.kivy_app.label.text = "Prediction Error = {error}".format(error=error)

                        if error != 0:
                            data = {"subject": "model", "data": GANN_instance}
                            response = pickle.dumps(data)
                        else:
                            data = {"subject": "done", "data": None}
                            response = pickle.dumps(data)

                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                        self.kivy_app.label.text = "Error Decoding the Client's Data"
                else:
                    response = pickle.dumps("Response from the Server")
                            
                try:
                    self.connection.sendall(response)
                except BaseException as e:
                    print("Error Sending Data to the Client: {msg}.\n".format(msg=e))
                    self.kivy_app.label.text = "Error Sending Data to the Client: {msg}".format(msg=e)

            else:
                print("The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {d_keys}.".format(d_keys=received_data.keys()))
                self.kivy_app.label.text = "Error Parsing Received Dictionary"
        else:
            print("A dictionary is expected to be received from the client but {d_type} received.".format(d_type=type(received_data)))
            self.kivy_app.label.text = "A dictionary is expected but {d_type} received.".format(d_type=type(received_data))

    def run(self):
        print("Running a Thread for the Connection with {client_info}.".format(client_info=self.client_info))
        self.kivy_app.label.text = "Running a Thread for the Connection with {client_info}.".format(client_info=self.client_info)

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
            print(date_time)
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                self.kivy_app.label.text = "Connection Closed with {client_info}".format(client_info=self.client_info)
                print("Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break

            # print(received_data)
            self.reply(received_data)

class ListenThread(threading.Thread):

    def __init__(self, kivy_app):
        threading.Thread.__init__(self)
        self.kivy_app = kivy_app

    def run(self):
        while True:
            try:
                connection, client_info = self.kivy_app.soc.accept()
                self.kivy_app.label.text = "New Connection from {client_info}".format(client_info=client_info)
                socket_thread = SocketThread(connection=connection,
                                             client_info=client_info, 
                                             kivy_app=self.kivy_app,
                                             buffer_size=1024,
                                             recv_timeout=10)
                socket_thread.start()
            except BaseException as e:
                self.kivy_app.soc.close()
                print(e)
                self.kivy_app.label.text = "Socket is No Longer Accepting Connections"
                self.kivy_app.create_socket_btn.disabled = False
                self.kivy_app.close_socket_btn.disabled = True
                break

serverApp = ServerApp()
serverApp.title="Server App"
serverApp.run()