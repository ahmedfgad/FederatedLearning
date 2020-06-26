import socket
import pickle
import numpy

import pygad
import pygad.nn
import pygad.gann

def fitness_func(solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                           data_inputs=data_inputs)
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions/data_outputs.size)*100

    return solution_fitness

def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, 
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))

last_fitness = 0


def prepare_GA(GANN_instance):
    # population does not hold the numerical weights of the network instead it holds a list of references to each last layer of each network (i.e. solution) in the population. A solution or a network can be used interchangeably.
    # If there is a population with 3 solutions (i.e. networks), then the population is a list with 3 elements. Each element is a reference to the last layer of each network. Using such a reference, all details of the network can be accessed.
    population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
    
    # To prepare the initial population, there are 2 ways:
    # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
    # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
    initial_population = population_vectors.copy()
    
    num_parents_mating = 4 # Number of solutions to be selected as parents in the mating pool.
    
    num_generations = 500 # Number of generations.
    
    mutation_percent_genes = 5 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
    
    parent_selection_type = "sss" # Type of parent selection.
    
    crossover_type = "single_point" # Type of the crossover operator.
    
    mutation_type = "random" # Type of the mutation operator.
    
    keep_parents = 1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
    
    init_range_low = -2
    init_range_high = 5
    
    ga_instance = pygad.GA(num_generations=num_generations, 
                           num_parents_mating=num_parents_mating, 
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           mutation_percent_genes=mutation_percent_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           keep_parents=keep_parents,
                           callback_generation=callback_generation)

    return ga_instance

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[1, 1],
                           [1, 0]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([0, 
                            1])

def recv(soc, buffer_size=1024, recv_timeout=10):
    received_data = b""
    while str(received_data)[-2] != '.':
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=recv_timeout))
            return None, 0
        except BaseException as e:
            return None, 0
            print("An error occurred while receiving data from the server {msg}.".format(msg=e))

    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
        return None, 0

    return received_data, 1

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

try:
    soc.connect(("localhost", 10000))
except BaseException as e:
    print("Error Connecting to the Server: {msg}".format(msg=e))
    soc.close()
    print("Socket Closed.")

print("Successful Connection to the Server.\n")

subject = "echo"
GANN_instance = None
best_sol_idx = -1

while True:
    data = {"subject": subject, "data": GANN_instance, "best_solution_idx": best_sol_idx}
    data_byte = pickle.dumps(data)
    
    print("Sending Data to the Server.\n")
    soc.sendall(data_byte)
    
    print("Receiving Reply from the Server.")
    received_data, status = recv(soc=soc, 
                                 buffer_size=1024, 
                                 recv_timeout=10)
    if status == 0:
        print("Nothing Received from the Server.")
        break
    else:
        print(received_data, end="\n\n")
    
    subject = received_data["subject"]
    if subject == "model":
        GANN_instance = received_data["data"]
    elif subject == "done":
        print("Model is trained.")
        break
    else:
        print("Unrecognized message type.")
        break

    ga_instance = prepare_GA(GANN_instance)

    ga_instance.run()

    ga_instance.plot_result()

    subject = "model"
    best_sol_idx = ga_instance.best_solution()[2]

soc.close()
print("Socket Closed.\n")
