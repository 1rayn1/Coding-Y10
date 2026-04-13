"""
Chat server process.

This module implements a simple TCP chat server that accepts multiple clients,
broadcasts normal chat messages to everyone, and supports direct messages,
friendship commands, and friend list queries.
"""

# Import socket module for network communication
import socket
# Import threading module for handling multiple client connections concurrently
import threading
# Import sys module to access command-line arguments
import sys
# Import time module for sleep functionality in retry logic
import time

# Define the host and port for the directory hub server
HUB_HOST = "127.0.0.1"
HUB_PORT = 9100

# Define the host and port for this chat server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9000
# Default server name if not provided via command line
SERVER_NAME = "no"

# Create the main listening socket for accepting chat client connections
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the specified host and port
server_sock.bind((SERVER_HOST, SERVER_PORT))
# Start listening for incoming connections
server_sock.listen()

# List to keep track of active client sockets
clients = []
# Parallel list to keep track of client nicknames (same index as clients)
nicknames = []
# Dictionary mapping nickname to a set of friend nicknames
friends = {}
# Threading lock to ensure thread-safe access to shared data structures
lock = threading.Lock()


def register_with_hub():
    """Register this chat server with the directory hub."""
    # Loop indefinitely until registration succeeds
    while True:
        try:
            # Create a new socket for connecting to the hub
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Connect to the hub at the specified host and port
            s.connect((HUB_HOST, HUB_PORT))
            # Create the registration message with server name, host, and port
            msg = f"REGISTER {SERVER_NAME} {SERVER_HOST} {SERVER_PORT}\n"
            # Send the registration message encoded as ASCII
            s.sendall(msg.encode("ascii"))
            # Receive the response from the hub
            resp = s.recv(1024).decode("ascii").strip()
            # Print the hub's response for logging
            print(f"[server] hub response: {resp}")
            # Close the connection to the hub
            s.close()
            # Break out of the retry loop since registration succeeded
            break
        except Exception as e:
            # If registration failed, print the error and retry after 3 seconds
            print(f"[server] failed to register with hub: {e}, retrying in 3s")
            time.sleep(3)


def broadcast(message, exclude=None):
    """Send a raw byte message to all connected clients, optionally skipping one."""
    # Acquire the lock to safely access the clients list
    with lock:
        # Iterate through all connected clients
        for c in clients:
            # Skip the excluded client if specified
            if c is exclude:
                continue
            try:
                # Send the message to the current client
                c.sendall(message)
            except:
                # Ignore exceptions (client may have disconnected)
                pass


def send_to_nickname(nick, text):
    """Send a text message to a specific connected nickname."""
    # Acquire the lock to safely access the nicknames and clients lists
    with lock:
        # Check if the target nickname is currently connected
        if nick in nicknames:
            # Find the index of the nickname in the list
            idx = nicknames.index(nick)
            try:
                # Send the text message encoded as ASCII to the corresponding client
                clients[idx].sendall(text.encode("ascii"))
            except:
                # Ignore exceptions (client may have disconnected)
                pass


def handle_client(client):
    """Receive messages from one client and act on them."""
    # Loop to continuously handle messages from this client
    while True:
        try:
            # Receive up to 1024 bytes of data from the client
            data = client.recv(1024)
            # If no data is received, the connection has been closed
            if not data:
                raise ConnectionError
            # Decode the received bytes to a string and strip whitespace
            text = data.decode("ascii").strip()

            # Determine the sender's nickname
            sender = None
            with lock:
                # Check if the client is still in the clients list
                if client in clients:
                    # Get the nickname corresponding to this client
                    sender = nicknames[clients.index(client)]

            # If sender is not found (client disconnected), skip processing
            if not sender:
                continue

            # Check if the message is a command (starts with "CMD:")
            if text.startswith("CMD:"):
                # Split the message into parts: CMD:command:parameters
                parts = text.split(":", 2)
                # Extract the command name
                cmd = parts[1]

                # Handle direct message command
                if cmd == "DM" and len(parts) == 3:
                    # Direct message format: CMD:DM:target:message
                    try:
                        # Split the parameters into target nickname and message
                        target, msg = parts[2].split(":", 1)
                    except ValueError:
                        # If splitting fails, send error message to client
                        client.sendall(b"[server] invalid DM format\n")
                        continue
                    # Check if the target user is connected
                    if target in nicknames:
                        # Send the DM to the target user
                        send_to_nickname(target, f"[DM from {sender}] {msg}")
                        # Confirm to the sender that the DM was sent
                        client.sendall(f"[server] DM sent to {target}\n".encode("ascii"))
                    else:
                        # Inform sender that the target user was not found
                        client.sendall(f"[server] user '{target}' not found\n".encode("ascii"))
                    continue

                # Handle add friend command
                if cmd == "ADD_FRIEND" and len(parts) == 3:
                    # Extract the target nickname
                    target = parts[2]
                    with lock:
                        # Check if the target user is connected
                        if target in nicknames:
                            # Add the friendship in both directions
                            friends.setdefault(sender, set()).add(target)
                            friends.setdefault(target, set()).add(sender)
                            # Confirm friendship to the sender
                            client.sendall(f"[server] you are now friends with {target}\n".encode("ascii"))
                        else:
                            # Inform sender that the target user was not found
                            client.sendall(f"[server] user '{target}' not found\n".encode("ascii"))
                    continue

                # Handle get friends list command
                if cmd == "GET_FRIENDS":
                    with lock:
                        # Get the sorted list of friends for this user
                        flist = sorted(list(friends.get(sender, set())))
                    # Create the response line with friends separated by commas
                    line = "FRIENDS:" + ",".join(flist) + "\n"
                    # Send the friends list to the client
                    client.sendall(line.encode("ascii"))
                    continue

                # If command is not recognized, send error message
                client.sendall(b"[server] unknown command\n")
                continue

            # If not a command, treat as a normal chat message and broadcast to all clients
            broadcast(f"{sender}: {text}\n".encode("ascii"), exclude=None)

        except Exception:
            # Handle client disconnection or other errors
            with lock:
                # Check if the client is still in the list
                if client in clients:
                    # Find the index of the client
                    idx = clients.index(client)
                    # Get the nickname before removing
                    nickname = nicknames[idx]
                    # Remove the client and nickname from the lists
                    clients.pop(idx)
                    nicknames.pop(idx)
                    # Broadcast that the user left
                    broadcast(f"[server] {nickname} left\n".encode("ascii"))
            # Close the client connection
            client.close()
            # Break out of the message handling loop
            break


def accept_loop():
    """Accept incoming client connections and start a handler thread."""
    # Print a message indicating the server is listening
    print(f"[server] listening on {SERVER_HOST}:{SERVER_PORT}")
    # Loop to continuously accept new connections
    while True:
        # Accept a new client connection
        client, addr = server_sock.accept()
        # Print the address of the connecting client
        print(f"[server] connection from {addr}")
        # Receive the nickname from the client (first message after connection)
        nickname = client.recv(1024).decode("ascii").strip()
        with lock:
            # Add the client socket and nickname to the respective lists
            clients.append(client)
            nicknames.append(nickname)
            # Initialize an empty friends set for this user if not already present
            friends.setdefault(nickname, set())
        # Send a welcome message to the client
        client.sendall(b"[server] connected to chat server\n")
        # Broadcast to all other clients that this user joined
        broadcast(f"[server] {nickname} joined\n".encode("ascii"), exclude=client)
        # Start a new daemon thread to handle messages from this client
        threading.Thread(target=handle_client, args=(client,), daemon=True).start()


# Check if command-line arguments are provided for server name and port
if len(sys.argv) >= 3:
    # Set server name from first argument
    SERVER_NAME = sys.argv[1]
    # Set server port from second argument (convert to int)
    SERVER_PORT = int(sys.argv[2])
    # Re-bind the server socket with the new port
    server_sock.bind((SERVER_HOST, SERVER_PORT))

# Register this server with the directory hub
register_with_hub()
# Start the main loop to accept client connections
accept_loop()
