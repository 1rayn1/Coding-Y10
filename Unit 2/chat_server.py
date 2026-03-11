import socket
import threading
import sys
import time

HUB_HOST = "127.0.0.1"
HUB_PORT = 9100

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9000
SERVER_NAME = "Main"

server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind((SERVER_HOST, SERVER_PORT))
server_sock.listen()

clients = []          # list of sockets
nicknames = []        # parallel list
friends = {}          # nickname -> set of nicknames
lock = threading.Lock()

def register_with_hub():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HUB_HOST, HUB_PORT))
            msg = f"REGISTER {SERVER_NAME} {SERVER_HOST} {SERVER_PORT}\n"
            s.sendall(msg.encode("ascii"))
            resp = s.recv(1024).decode("ascii").strip()
            print(f"[server] hub response: {resp}")
            s.close()
            break
        except Exception as e:
            print(f"[server] failed to register with hub: {e}, retrying in 3s")
            time.sleep(3)

def broadcast(message, exclude=None):
    with lock:
        for c in clients:
            if c is exclude:
                continue
            try:
                c.sendall(message)
            except:
                pass

def send_to_nickname(nick, text):
    with lock:
        if nick in nicknames:
            idx = nicknames.index(nick)
            try:
                clients[idx].sendall(text.encode("ascii"))
            except:
                pass

def handle_client(client):
    while True:
        try:
            data = client.recv(1024)
            if not data:
                raise ConnectionError
            text = data.decode("ascii").strip()

            sender = None
            with lock:
                if client in clients:
                    sender = nicknames[clients.index(client)]

            if not sender:
                continue

            # COMMANDS
            if text.startswith("CMD:"):
                parts = text.split(":", 2)
                cmd = parts[1]

                # DM
                if cmd == "DM" and len(parts) == 3:
                    # parts[2] = target:message
                    try:
                        target, msg = parts[2].split(":", 1)
                    except ValueError:
                        client.sendall(b"[server] invalid DM format\n")
                        continue
                    if target in nicknames:
                        send_to_nickname(target, f"[DM from {sender}] {msg}")
                        client.sendall(f"[server] DM sent to {target}\n".encode("ascii"))
                    else:
                        client.sendall(f"[server] user '{target}' not found\n".encode("ascii"))
                    continue

                # ADD FRIEND
                if cmd == "ADD_FRIEND" and len(parts) == 3:
                    target = parts[2]
                    with lock:
                        if target in nicknames:
                            friends.setdefault(sender, set()).add(target)
                            friends.setdefault(target, set()).add(sender)
                            client.sendall(f"[server] you are now friends with {target}\n".encode("ascii"))
                        else:
                            client.sendall(f"[server] user '{target}' not found\n".encode("ascii"))
                    continue

                # GET FRIENDS
                if cmd == "GET_FRIENDS":
                    with lock:
                        flist = sorted(list(friends.get(sender, set())))
                    line = "FRIENDS:" + ",".join(flist) + "\n"
                    client.sendall(line.encode("ascii"))
                    continue

                client.sendall(b"[server] unknown command\n")
                continue

            # normal broadcast
            broadcast(f"{sender}: {text}\n".encode("ascii"), exclude=None)

        except Exception:
            with lock:
                if client in clients:
                    idx = clients.index(client)
                    nickname = nicknames[idx]
                    clients.pop(idx)
                    nicknames.pop(idx)
                    broadcast(f"[server] {nickname} left\n".encode("ascii"))
            client.close()
            break

def accept_loop():
    print(f"[server] listening on {SERVER_HOST}:{SERVER_PORT}")
    while True:
        client, addr = server_sock.accept()
        print(f"[server] connection from {addr}")
        nickname = client.recv(1024).decode("ascii").strip()
        with lock:
            clients.append(client)
            nicknames.append(nickname)
            friends.setdefault(nickname, set())
        client.sendall(b"[server] connected to chat server\n")
        broadcast(f"[server] {nickname} joined\n".encode("ascii"), exclude=client)
        threading.Thread(target=handle_client, args=(client,), daemon=True).start()

if len(sys.argv) >= 3:
    SERVER_NAME = sys.argv[1]
    SERVER_PORT = int(sys.argv[2])
    server_sock.bind((SERVER_HOST, SERVER_PORT))
register_with_hub()
accept_loop()
