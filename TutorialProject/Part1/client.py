import socket
import pickle

soc = socket.socket()
print("Socket is created.")

soc.connect(("localhost", 10000))
print("Connected to the server.")

msg = "A message from the client."
msg = pickle.dumps(msg)
soc.sendall(msg)
print("Client sent a message to the server.")

received_data = b''
while str(received_data)[-2] != '.':
    data = soc.recv(8)
    received_data += data

received_data = pickle.loads(received_data)
print("Received data from the client: {received_data}".format(received_data=received_data))

soc.close()
print("Socket is closed.")