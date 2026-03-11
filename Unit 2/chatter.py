import curses
import socket
import threading
import sys
from auth import validate_credentials, username_exists, create_user

HUB_HOST = "127.0.0.1"
HUB_PORT = 9100

client = None
chat_history = []
nickname = None
current_server = None  # (name, host, port)

def connect_to_server(nick, host, port):
    global client, nickname, current_server
    nickname = nick
    current_server = (None, host, port)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    client.sendall(nickname.encode("ascii"))

    def receive_messages():
        while True:
            try:
                msg = client.recv(1024).decode("ascii")
                if not msg:
                    break
                chat_history.append(msg.rstrip("\n"))
            except Exception as e:
                chat_history.append(f"[receiver crashed] {e}")
                break

    threading.Thread(target=receive_messages, daemon=True).start()

def fetch_servers_from_hub():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HUB_HOST, HUB_PORT))
    s.sendall(b"LIST\n")
    data = s.recv(4096).decode("ascii").strip()
    s.close()
    if not data.startswith("SERVERS:"):
        return []
    payload = data[len("SERVERS:"):]
    if not payload:
        return []
    entries = payload.split(";")
    result = []
    for e in entries:
        name, host, port = e.split(",")
        result.append((name, host, int(port)))
    return result

def leave(stdscr):
    curses.curs_set(0)
    stdscr.clear()
    options = ["Yes","No"]
    selected = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "== ARE YOU SURE YOU WANT TO LEAVE? ==")
        stdscr.addstr(1, 0, "Use UP/DOWN arrows and ENTER to select.")
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(3 + i, 0, f"> {option}")
            else:
                stdscr.addstr(3 + i, 0, f"  {option}")
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        elif key in [10, 13]:
            chosen = options[selected]
            break
    if chosen == "Yes":
        sys.exit()
    elif chosen == "No":
        home_page(stdscr)

def start_menu(stdscr):
    curses.curs_set(0)
    stdscr.clear()
    options = ["Log In", "Sign Up", "Quit"]
    selected = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "== Welcome to the Chatting Cafe ==")
        stdscr.addstr(1, 0, "Use UP/DOWN arrows and ENTER to select.")
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(3 + i, 0, f"> {option}")
            else:
                stdscr.addstr(3 + i, 0, f"  {option}")
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        elif key in [10, 13]:
            return options[selected]

def log_in(stdscr):
    curses.curs_set(1)
    curses.echo()

    while True:
        stdscr.clear()
        stdscr.addstr(1, 0, "== Log In ==")
        stdscr.addstr(3, 0, "Enter username (max 20 chars): ")
        username = stdscr.getstr(3, 32, 20).decode("utf-8")

        stdscr.addstr(4, 0, "Enter password: ")
        password = stdscr.getstr(4, 16, 50).decode("utf-8")

        if validate_credentials(username, password):
            stdscr.addstr(6, 0, "Login successful! Press any key to continue.")
            stdscr.refresh()
            stdscr.getch()
            return server_menu(stdscr, username)
        else:
            stdscr.addstr(6, 0, "Invalid username or password. Try again.")
            stdscr.refresh()
            stdscr.getch()

def sign_up(stdscr):
    curses.curs_set(1)
    curses.echo()

    while True:
        stdscr.clear()
        stdscr.addstr(1, 0, "== Sign Up ==")
        stdscr.addstr(3, 0, "Choose a username (max 20 chars): ")
        username = stdscr.getstr(3, 34, 20).decode("utf-8")

        if username_exists(username):
            stdscr.addstr(5, 0, "Username already exists. Press any key to try again.")
            stdscr.refresh()
            stdscr.getch()
            continue

        stdscr.addstr(4, 0, "Choose a password: ")
        password = stdscr.getstr(4, 19, 50).decode("utf-8")

        stdscr.addstr(5, 0, "Confirm password: ")
        confirm = stdscr.getstr(5, 18, 50).decode("utf-8")

        if password != confirm:
            stdscr.addstr(7, 0, "Passwords do not match. Press any key to retry.")
            stdscr.refresh()
            stdscr.getch()
            continue

        create_user(username, password)
        stdscr.addstr(7, 0, "Account created successfully! Press any key to continue.")
        stdscr.refresh()
        stdscr.getch()
        return server_menu(stdscr, username)

def server_menu(stdscr, username):
    curses.curs_set(0)
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "== Select Server ==")
        stdscr.addstr(1, 0, "Fetching from hub...")
        stdscr.refresh()

        servers = fetch_servers_from_hub()
        if not servers:
            stdscr.addstr(3, 0, "No servers registered. Press q to quit or r to retry.")
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord('q'), ord('Q')):
                return
            else:
                continue

        selected = 0
        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, "== Select Server ==")
            for i, (name, host, port) in enumerate(servers):
                prefix = "> " if i == selected else "  "
                stdscr.addstr(2 + i, 0, f"{prefix}{name} ({host}:{port})")
            stdscr.addstr(2 + len(servers) + 1, 0, "Use UP/DOWN, ENTER to select, q to quit.")
            stdscr.refresh()

            key = stdscr.getch()
            if key == curses.KEY_UP:
                selected = (selected - 1) % len(servers)
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % len(servers)
            elif key in [10, 13]:
                name, host, port = servers[selected]
                connect_to_server(username, host, port)
                return home_page(stdscr)
            elif key in (ord('q'), ord('Q')):
                return

def settings(stdscr):
    curses.curs_set(0)
    options = ["List of Commands", "Home Page","Quit"]
    selected = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "== Settings ==")
        stdscr.addstr(1, 0, "Use UP/DOWN arrows and ENTER to select.")
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(3 + i, 0, f"> {option}")
            else:
                stdscr.addstr(3 + i, 0, f"  {option}")
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        elif key in [10, 13]:
            chosen =  options[selected]
            break
    if chosen == "List of Commands":
        cmd_list(stdscr)
    elif chosen == "Home Page":
        home_page(stdscr)
    elif chosen == "Quit":
        leave(stdscr)

def cmd_list(stdscr):
    curses.curs_set(0)
    stdscr.clear()
    stdscr.addstr(0, 0, "== Commands ==")
    stdscr.addstr(2, 0, "/dm <user> <message>   - send a direct message")
    stdscr.addstr(3, 0, "/addfriend <user>      - add a friend")
    stdscr.addstr(4, 0, "/friends               - show friends list")
    stdscr.addstr(6, 0, "Press any key to return.")
    stdscr.refresh()
    stdscr.getch()
    home_page(stdscr)

def friends_page(stdscr):
    curses.curs_set(0)
    if not client:
        return
    client.sendall(b"CMD:GET_FRIENDS\n")
    curses.napms(150)

    friends_lines = [line for line in chat_history if line.startswith("FRIENDS:")]
    if friends_lines:
        last = friends_lines[-1]
        payload = last[len("FRIENDS:"):]
        flist = [f for f in payload.split(",") if f]
    else:
        flist = []

    stdscr.clear()
    stdscr.addstr(0, 0, "== Friends ==")
    if not flist:
        stdscr.addstr(2, 0, "No friends yet.")
    else:
        for i, f in enumerate(flist, start=2):
            stdscr.addstr(i, 0, f"- {f}")
    stdscr.addstr(len(flist) + 3, 0, "Press any key to return.")
    stdscr.refresh()
    stdscr.getch()
    home_page(stdscr)

def DMS(stdscr):
    curses.curs_set(1)
    stdscr.nodelay(False)

    height, width = stdscr.getmaxyx()

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "== Chat == (type 'back' to return)")
        stdscr.addstr(1, 0, "Use /dm user message, /addfriend user, /friends")

        start = max(0, len(chat_history) - (height - 5))
        for i, line in enumerate(chat_history[start:], start=3):
            if i >= height - 2:
                break
            stdscr.addstr(i, 0, line[:width - 1])

        stdscr.addstr(height - 2, 0, "Message: ")
        stdscr.move(height - 2, 9)
        stdscr.clrtoeol()
        try:
            raw = stdscr.getstr(height - 2, 9, width - 10)
            message = raw.decode("utf-8")
        except Exception:
            message = ""

        if message.lower() == "back":
            break

        if message.startswith("/dm "):
            parts = message.split(" ", 2)
            if len(parts) == 3:
                target, dm = parts[1], parts[2]
                wire = f"CMD:DM:{target}:{dm}\n"
                client.sendall(wire.encode("ascii"))
            else:
                chat_history.append("[client] usage: /dm user message")
        elif message.startswith("/addfriend "):
            parts = message.split(" ", 1)
            target = parts[1].strip()
            wire = f"CMD:ADD_FRIEND:{target}\n"
            client.sendall(wire.encode("ascii"))
        elif message.strip() == "/friends":
            client.sendall(b"CMD:GET_FRIENDS\n")
        else:
            if client:
                try:
                    client.sendall(f"{message}\n".encode("ascii"))
                except Exception:
                    chat_history.append("[error] failed to send message")
            else:
                chat_history.append("[error] not connected to server")

        stdscr.refresh()

    home_page(stdscr)

def server_page(stdscr):
    # just re-open server menu
    if nickname:
        server_menu(stdscr, nickname)
    else:
        home_page(stdscr)

def home_page(stdscr):
    curses.curs_set(0)
    stdscr.clear()
    options = ["DM's", "Server", "Friends","Settings","Log Out","Quit"]
    selected = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "== Landing Page ==")
        stdscr.addstr(1, 0, "Use UP/DOWN arrows and ENTER to select.")
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(3 + i, 0, f"> {option}")
            else:
                stdscr.addstr(3 + i, 0, f"  {option}")
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        elif key in [10, 13]:
            yes =  options[selected]
            break
    if yes == "DM's":
        DMS(stdscr)
    elif yes == "Server":
        server_page(stdscr)
    elif yes == "Friends":
        friends_page(stdscr)
    elif yes == "Settings":
        settings(stdscr)
    elif yes == "Log Out":
        main(stdscr)
    elif yes == "Quit":
        leave(stdscr)

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(0)
    stdscr.keypad(True)
    choice = start_menu(stdscr)
    if choice == "Log In":
        log_in(stdscr)
    elif choice == "Sign Up":
        sign_up(stdscr)
    elif choice == "Quit":
        leave(stdscr)

curses.wrapper(main)