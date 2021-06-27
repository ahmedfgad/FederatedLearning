import socket
import pickle
import numpy
import threading

import tensorflow.keras
import pygad.kerasga
import pygad

def close_socket(*args):
    soc.close()
    print("Socket Closed")

def fitness_func(solution, sol_idx):
    global keras_ga, data_inputs, data_outputs

    model = keras_ga.model

    predictions = pygad.kerasga.predict(model, 
                                        solution, 
                                        data_inputs)

    bce = tensorflow.keras.losses.BinaryCrossentropy()
    solution_fitness = 1.0 / (bce(data_outputs, predictions).numpy() + 0.00000001)

    return solution_fitness

def prepare_GA(server_data):
    global keras_ga

    population_weights = server_data["population_weights"]
    model_json = server_data["model_json"]
    num_solutions = server_data["num_solutions"]

    model = tensorflow.keras.models.model_from_json(model_json)
    keras_ga = pygad.kerasga.KerasGA(model=model,
                                     num_solutions=num_solutions)

    keras_ga.population_weights = population_weights

    ga_instance = pygad.GA(num_generations=150, 
                           num_parents_mating=4, 
                           initial_population=keras_ga.population_weights.copy(),
                           fitness_func=fitness_func)

    return ga_instance

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[1, 0],
                           [1, 1]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([[0, 1], 
                            [1, 0]])

class RecvThread(threading.Thread):

    def __init__(self, buffer_size, recv_timeout):
        threading.Thread.__init__(self)
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                soc.settimeout(self.recv_timeout)
                received_data += soc.recv(self.buffer_size)

                try:
                    pickle.loads(received_data)
                    print("All data ({data_len} bytes) is received from the server.".format(data_len=len(received_data)), end="\n")
                    # If the previous pickle.loads() statement is passed, this means all the data is received.
                    # Thus, no need to continue the loop and a break statement should be excuted.
                    break
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

            except socket.timeout:
                print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=self.recv_timeout))
                print("{recv_timeout} Seconds of Inactivity. socket.timeout Exception Occurred".format(recv_timeout=self.recv_timeout))
                return None, 0
            except BaseException as e:
                return None, 0
                print("Error While Receiving Data from the Server: {msg}.".format(msg=e))

        try:
            received_data = pickle.loads(received_data)
        except BaseException as e:
            print("Error Decoding the Data: {msg}.\n".format(msg=e))
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

            print("Sending a Message of Type {subject} to the Server".format(subject=subject))
            try:
                soc.sendall(data_byte)
            except BaseException as e:
                print("Error Connecting to the Server. The server might has been closed: {msg}".format(msg=e))
                break

            print("Receiving Reply from the Server")
            received_data, status = self.recv()
            if status == 0:
                print("Nothing Received from the Server")
                break
            else:
                print("New Message from the Server with subject {sub}".format(sub=received_data["subject"]))

            subject = received_data["subject"]
            if subject == "model":
                server_data = received_data["data"]
            elif subject == "done":
                print("Model is Trained")
                break
            else:
                print("Unrecognized Message Type: {subject}".format(subject=subject))
                break

            ga_instance = prepare_GA(server_data)

            ga_instance.run()

            subject = "model"
            best_sol_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)[2]
            best_model_weights_vector = ga_instance.population[best_sol_idx, :]

            predictions = keras_ga.model.predict(data_inputs)
            ba = tensorflow.keras.metrics.BinaryAccuracy()
            ba.update_state(data_outputs, predictions)
            accuracy = ba.result().numpy()
            print("Accuracy {acc}".format(acc=accuracy), end="\n\n")

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created")

try:
    soc.connect(("192.168.0.10", int("5000")))
    print("Successful Connection to the Server")
except BaseException as e:
    print("Error Connecting to the Server: {msg}".format(msg=e))

recvThread = RecvThread(buffer_size=1024, recv_timeout=10)
recvThread.start()
