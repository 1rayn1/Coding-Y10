"""
Directory hub service.

This module runs a simple TCP hub that keeps track of available chat servers.
Chat clients query this hub to discover servers, and chat servers register
with this hub so they can be listed for clients.
"""

# Import the socket module for network communication
import socket
# Import threading module for handling multiple client connections concurrently
import threading

# Define the host IP address where the server will listen (localhost)
HOST = "127.0.0.1"
# Define the port number on which the server will listen
PORT = 9100

# Dictionary to store registered chat servers: key is server name, value is (host, port) tuple
servers = {}
# Threading lock to ensure thread-safe access to the servers dictionary
lock = threading.Lock()

def handle_client(conn):
    """
    Handle an incoming connection from a client or chat server.

    This function runs in a separate thread for each connection.
    It processes commands like REGISTER (from servers) and LIST (from clients).
    """
    # Use a context manager to ensure the connection is properly closed
    with conn:
        # Loop to continuously receive and process data from the client
        while True:
            # Receive up to 1024 bytes of data from the connection
            data = conn.recv(1024)
            # If no data is received, the connection has been closed, so break the loop
            if not data:
                break
            # Decode the received bytes to a string and strip whitespace
            text = data.decode("ascii").strip()
            # Split the text into parts based on whitespace
            parts = text.split()

            # If there are no parts (empty message), skip to the next iteration
            if not parts:
                continue

            # Get the command (first part) and convert to uppercase for case-insensitive matching
            cmd = parts[0].upper()

            # Handle the REGISTER command: expects 4 parts (REGISTER name host port)
            if cmd == "REGISTER" and len(parts) == 4:
                # Extract the server name, host, and port from the command parts
                name, host, port = parts[1], parts[2], parts[3]
                # Acquire the lock to safely modify the shared servers dictionary
                with lock:
                    # Add the server to the dictionary with name as key and (host, port) as value
                    servers[name] = (host, int(port))
                # Send a confirmation response to the client
                conn.sendall(b"OK\n")

            # Handle the LIST command: no additional parameters expected
            elif cmd == "LIST":
                # Acquire the lock to safely read from the shared servers dictionary
                with lock:
                    # If no servers are registered, send an empty list response
                    if not servers:
                        conn.sendall(b"SERVERS:\n")
                    else:
                        # Build a list of server entries in "name,host,port" format
                        items = []
                        for name, (h, p) in servers.items():
                            items.append(f"{name},{h},{p}")
                        # Create the response line with all servers separated by semicolons
                        line = "SERVERS:" + ";".join(items) + "\n"
                        # Send the response encoded as ASCII bytes
                        conn.sendall(line.encode("ascii"))

            # If the command is not recognized, send an error response
            else:
                conn.sendall(b"ERR\n")

# Create a TCP socket using IPv4 and TCP stream
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the specified host and port
s.bind((HOST, PORT))
# Start listening for incoming connections (default backlog)
s.listen()
# Print a message indicating the server is listening
print(f"[directory] listening on {HOST}:{PORT}")
# Main server loop to accept incoming connections
while True:
    # Accept a new connection, getting the connection socket and client address
    conn, addr = s.accept()
    # Start a new daemon thread to handle the client connection
    # Daemon threads exit when the main thread exits
    threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()