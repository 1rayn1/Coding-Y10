import socket, threading 

host = '127.0.0.1'
port = 9000 

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server.bind((host, port)) 
server.listen()

clients = []
nicknames = []

def broadcast(message): 
    for client in clients:
        client.send(message)


def handle(client):
    while True:
        try:
            message = client.recv(1024)
            text = message.decode('ascii')
            if text.startswith("/dm "):
                parts = text.split(" ", 2)
                if len(parts) == 3:
                    target, dm = parts[1], parts[2]
                    if target in nicknames:
                        idx = nicknames.index(target)
                        clients[idx].send(f"[DM from {nicknames[clients.index(client)]}] {dm}".encode('ascii'))
                    else:
                        client.send(f"User '{target}' not found.".encode('ascii'))
                continue          # don’t broadcast the DM

            # normal group broadcast
            broadcast(message)
        except:
            index = clients.index(client)
            clients.remove(client)
            client.close()
            nickname = nicknames[index]
            broadcast('{} left!'.format(nickname).encode('ascii'))
            nicknames.remove(nickname)
            break


def receive():  # accepting multiple clients
    while True:
        client, address = server.accept()
        print("Connected with {}".format(str(address)))
        nickname = client.recv(1024).decode('ascii')
        nicknames.append(nickname)
        clients.append(client)
        print("Nickname is {}".format(nickname))
        broadcast("{} joined!".format(nickname).encode('ascii'))
        client.send(' Connected to server!'.encode('ascii'))
        thread = threading.Thread(target=handle, args=(client,))
        thread.start()


receive()
print("Server running!")