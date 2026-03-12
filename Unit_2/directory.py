import socket
import threading

HOST = "127.0.0.1"
PORT = 9100

servers = {}
lock = threading.Lock()

def handle_client(conn, addr):
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            text = data.decode("ascii").strip()
            parts = text.split()

            if not parts:
                continue

            cmd = parts[0].upper()

            if cmd == "REGISTER" and len(parts) == 4:
                name, host, port = parts[1], parts[2], parts[3]
                with lock:
                    servers[name] = (host, int(port))
                conn.sendall(b"OK\n")

            elif cmd == "LIST":
                with lock:
                    if not servers:
                        conn.sendall(b"SERVERS:\n")
                    else:
                        items = []
                        for name, (h, p) in servers.items():
                            items.append(f"{name},{h},{p}")
                        line = "SERVERS:" + ";".join(items) + "\n"
                        conn.sendall(line.encode("ascii"))

            else:
                conn.sendall(b"ERR\n")


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
print(f"[directory] listening on {HOST}:{PORT}")
while True:
    conn, addr = s.accept()
    threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()