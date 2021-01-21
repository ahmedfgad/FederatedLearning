import socket
import pickle
import numpy
import threading

import tensorflow.keras
import pygad.kerasga
import pygad

import kivy.app
import kivy.uix.button
import kivy.uix.label
import kivy.uix.boxlayout
import kivy.uix.textinput

class ClientApp(kivy.app.App):
    
    def __init__(self):
        super().__init__()

    def create_socket(self, *args):
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.label.text = "Socket Created"

        self.create_socket_btn.disabled = True
        self.connect_btn.disabled = False
        self.close_socket_btn.disabled = False

    def connect(self, *args):
        try:
            self.soc.connect((self.server_ip.text, int(self.server_port.text)))
            self.label.text = "Successful Connection to the Server"
    
            self.connect_btn.disabled = True
            self.recv_train_model_btn.disabled = False

        except BaseException as e:
            self.label.text = "Error Connecting to the Server"
            print("Error Connecting to the Server: {msg}".format(msg=e))

            self.connect_btn.disabled = False
            self.recv_train_model_btn.disabled = True

    def recv_train_model(self, *args):
        global keras_ga

        self.recv_train_model_btn.disabled = True
        recvThread = RecvThread(kivy_app=self, buffer_size=1024, recv_timeout=10)
        recvThread.start()

    def close_socket(self, *args):
        self.soc.close()
        self.label.text = "Socket Closed"

        self.create_socket_btn.disabled = False
        self.connect_btn.disabled = True
        self.recv_train_model_btn.disabled = True
        self.close_socket_btn.disabled = True

    def build(self):
        self.create_socket_btn = kivy.uix.button.Button(text="Create Socket")
        self.create_socket_btn.bind(on_press=self.create_socket)

        self.server_ip = kivy.uix.textinput.TextInput(hint_text="Server IPv4 Address", text="localhost")
        self.server_port = kivy.uix.textinput.TextInput(hint_text="Server Port Number", text="10000")

        self.server_info_boxlayout = kivy.uix.boxlayout.BoxLayout(orientation="horizontal")
        self.server_info_boxlayout.add_widget(self.server_ip)
        self.server_info_boxlayout.add_widget(self.server_port)

        self.connect_btn = kivy.uix.button.Button(text="Connect to Server", disabled=True)
        self.connect_btn.bind(on_press=self.connect)

        self.recv_train_model_btn = kivy.uix.button.Button(text="Receive & Train Model", disabled=True)
        self.recv_train_model_btn.bind(on_press=self.recv_train_model)

        self.close_socket_btn = kivy.uix.button.Button(text="Close Socket", disabled=True)
        self.close_socket_btn.bind(on_press=self.close_socket)

        self.label = kivy.uix.label.Label(text="Socket Status")

        self.box_layout = kivy.uix.boxlayout.BoxLayout(orientation="vertical")
        self.box_layout.add_widget(self.create_socket_btn)
        self.box_layout.add_widget(self.server_info_boxlayout)
        self.box_layout.add_widget(self.connect_btn)
        self.box_layout.add_widget(self.recv_train_model_btn)
        self.box_layout.add_widget(self.close_socket_btn)
        self.box_layout.add_widget(self.label)

        return self.box_layout

def fitness_func(solution, sol_idx):
    global keras_ga, data_inputs, data_outputs

    model = keras_ga.model

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    predictions = model.predict(data_inputs)
    bce = tensorflow.keras.losses.BinaryCrossentropy()
    solution_fitness = 1.0 / (bce(data_outputs, predictions).numpy() + 0.00000001)

    return solution_fitness

"""
def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, 
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

#    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
#    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
#    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
"""

#last_fitness = 0

def prepare_GA(server_data):
    global keras_ga

    population_weights = server_data["population_weights"]
    model_json = server_data["model_json"]
    num_solutions = server_data["num_solutions"]

    model = tensorflow.keras.models.model_from_json(model_json)
    keras_ga = pygad.kerasga.KerasGA(model=model,
                                     num_solutions=num_solutions)

    keras_ga.population_weights = population_weights

    population_vectors = keras_ga.population_weights

    # To prepare the initial population, there are 2 ways:
    # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
    # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
    initial_population = population_vectors.copy()
    
    num_parents_mating = 4 # Number of solutions to be selected as parents in the mating pool.
    
    num_generations = 50 # Number of generations.

    mutation_percent_genes = 5 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.

    ga_instance = pygad.GA(num_generations=num_generations, 
                           num_parents_mating=num_parents_mating, 
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           mutation_percent_genes=mutation_percent_genes)

    return ga_instance

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[1, 0],
                           [1, 1]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([[0, 1], 
                            [1, 0]])

class RecvThread(threading.Thread):

    def __init__(self, kivy_app, buffer_size, recv_timeout):
        threading.Thread.__init__(self)
        self.kivy_app = kivy_app
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True: # str(received_data)[-2] != '.':
            try:
                self.kivy_app.soc.settimeout(self.recv_timeout)
                received_data += self.kivy_app.soc.recv(self.buffer_size)

                try:
                    pickle.loads(received_data)
                    self.kivy_app.label.text = "All data is received from the server."
                    print("All data is received from the server.")
                    # If the previous pickle.loads() statement is passed, this means all the data is received.
                    # Thus, no need to continue the loop and a break statement should be excuted.
                    break
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

            except socket.timeout:
                print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=self.recv_timeout))
                self.kivy_app.label.text = "{recv_timeout} Seconds of Inactivity. socket.timeout Exception Occurred".format(recv_timeout=self.recv_timeout)
                return None, 0
            except BaseException as e:
                return None, 0
                print("Error While Receiving Data from the Server: {msg}.".format(msg=e))
                self.kivy_app.label.text = "Error While Receiving Data from the Server"

        try:
            received_data = pickle.loads(received_data)
        except BaseException as e:
            print("Error Decoding the Data: {msg}.\n".format(msg=e))
            self.kivy_app.label.text = "Error Decoding the Client's Data"
            return None, 0
    
        return received_data, 1

    def run(self):
        global server_data

        subject = "echo"
        server_data = None
        best_sol_idx = -1
        best_model_weights_vector = None

        while True:
            data_dict = {"best_model_weights_vector": best_model_weights_vector}
            data = {"subject": subject, "data": data_dict}

            # data = {"subject": subject, "data": keras_ga, "best_solution_idx": best_sol_idx}
            data_byte = pickle.dumps(data)

            self.kivy_app.label.text = "Sending a Message of Type {subject} to the Server".format(subject=subject)
            try:
                self.kivy_app.soc.sendall(data_byte)
            except BaseException as e:
                self.kivy_app.label.text = "Error Connecting to the Server. The server might has been closed."
                print("Error Connecting to the Server: {msg}".format(msg=e))
                break

            self.kivy_app.label.text = "Receiving Reply from the Server"
            received_data, status = self.recv()
            if status == 0:
                self.kivy_app.label.text = "Nothing Received from the Server"
                break
            else:
                self.kivy_app.label.text = "New Message from the Server"

            subject = received_data["subject"]
            if subject == "model":
                server_data = received_data["data"]
            elif subject == "done":
                self.kivy_app.label.text = "Model is Trained"
                break
            else:
                self.kivy_app.label.text = "Unrecognized Message Type: {subject}".format(subject=subject)
                break

            ga_instance = prepare_GA(server_data)

            ga_instance.run()

            subject = "model"
            best_sol_idx = ga_instance.best_solution()[2]
            best_model_weights_vector = ga_instance.population[best_sol_idx, :]

clientApp = ClientApp()
clientApp.title = "Client App"
clientApp.run()
