import curses
import socket, threading
from auth import validate_credentials, username_exists, create_user
import sys

client = None
chat_history = []
nickname = None

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

def connect_to_server(nick, ip='127.0.0.1', port=9000):
    """Establish a socket connection and start the receiver thread."""
    global client, nickname
    nickname = nick
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, port))
    client.send(nickname.encode('ascii'))

    def receive_messages():
        while True:
            try:
                msg = client.recv(1024).decode('ascii')
                if msg:
                    chat_history.append(msg)
            except Exception as e:
                chat_history.append(f"[receiver crashed] {e}")
                break

    threading.Thread(target=receive_messages, daemon=True).start()

#This is the starting menu.
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
            try:
                connect_to_server(username)
            except Exception as e:
                stdscr.addstr(8, 0, f"Could not connect to server: {e}")
                stdscr.refresh()
                stdscr.getch()
            return home_page(stdscr)
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
        try:
            connect_to_server(username)
        except Exception as e:
            stdscr.addstr(9, 0, f"Could not connect to server: {e}")
            stdscr.refresh()
            stdscr.getch()
        return home_page(stdscr)

def guest(stdscr):
    curses.curs_set(1)
    stdscr.nodelay(False)

    return

def settings(stdscr):
    curses.curs_set(1)
    stdscr.nodelay(False)
    options = ["List of Commands", "Edit Profile","Home Page","Quit"]
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
    elif chosen == "Edit Profile":
        profile(stdscr)
    elif chosen == "Home Page":
        home_page(stdscr)
    elif chosen == "Quit":
        leave(stdscr)

def cmd_list(stdscr):
    return

def profile(stdscr):
    return



def server(stdscr):
    curses.curs_set(1)
    stdscr.nodelay(False)

    height, width = stdscr.getmaxyx()

    while True:
        stdscr.clear()

        stdscr.addstr(0, 0, "== Chat == (type 'back' to return)")

        stdscr.refresh()
        start = max(0, len(chat_history) - (height - 4))

        for i, line in enumerate(chat_history[start:], start=2):
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

        if client:
            try:
                client.send(f"{nickname}: {message}".encode("ascii"))
            except Exception:
                chat_history.append("[error] failed to send message")
        else:
            chat_history.append("[error] not connected to server")

        stdscr.refresh()

    home_page(stdscr)


def DMS(stdscr):
    return

def friends(stdscr):
    return

def home_page(stdscr):
    curses.curs_set(0)
    stdscr.clear()
    options = ["DM's", "Server", "Friends","Settings","Back"]
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
        server(stdscr)
    elif yes == "Friends":
        friends(stdscr)
    elif yes == "Settings":
        settings(stdscr)
    elif yes == "Back":
        main(stdscr)

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(0)
    stdscr.keypad(True)
    curses.start_color()
    viewport_width, viewport_height = 20, 10
    choice = start_menu(stdscr)
    if choice == "Log In":
        log_in(stdscr)
    elif choice == "Sign Up":
        sign_up(stdscr)
    elif choice == "Guest":
        guest(stdscr)
    elif choice == "Quit":
        return

curses.wrapper(main)