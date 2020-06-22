import socket
import pickle

soc = socket.socket()
print("Socket is created.")

soc.bind(("localhost", 10000))
print("Socket is bound to an address & port number.")

soc.listen(1)
print("Listening for incoming connection ...")

connected = False
accept_timeout = 10
soc.settimeout(accept_timeout)
try:
    connection, address = soc.accept()
    print("Connected to a client: {client_info}.".format(client_info=address))
    connected = True
except socket.timeout:
    print("A socket.timeout exception occurred because the server did not receive any connection for {accept_timeout} seconds.".format(accept_timeout=accept_timeout))

received_data = b''
if connected:
    while str(received_data)[-2] != '.':
        data = connection.recv(8)
        received_data += data
    received_data = pickle.loads(received_data)
    print("Received data from the client: {received_data}".format(received_data=received_data))

    msg = "Reply from the server."
    msg = pickle.dumps(msg)
    connection.sendall(msg)
    print("Server sent a message to the client.")
    
    connection.close()
    print("Connection is closed with: {client_info}.".format(client_info=address))

soc.close()
print("Socket is closed.")
