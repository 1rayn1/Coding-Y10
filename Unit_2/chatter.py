"""
Terminal chat client.

This module provides a curses-based chat client interface.
Users can log in or sign up, select a chat server, and send messages.
The client also receives messages in a background thread and keeps a local
chat history for display.
"""

# Import curses module for terminal-based user interface
import curses
# Import socket module for network communication with servers
import socket
# Import threading module for running message receiving in background
import threading
# Import sys module for system exit functionality
import sys
# Import authentication functions from the auth module
from auth import validate_credentials, username_exists, create_user
# Import random module for selecting random activities
import random
# Import time module for napms function in curses
import time

# Define the host and port for the directory hub server
HUB_HOST = "127.0.0.1"
HUB_PORT = 9100

# Global variable for the client socket connection to the chat server
client = None
# List to store chat message history for display
chat_history = []
# Global variable for the current user's nickname
nickname = None
# Global variable for the current server information (name, host, port)
current_server = None


def receive_messages():
    """Background thread that receives messages from the chat server."""
    # Loop continuously to receive messages
    while True:
        try:
            # Receive a message from the server (up to 1024 bytes)
            msg = client.recv(1024).decode("ascii")
            # If no message received (connection closed), break the loop
            if not msg:
                break
            # Add the message to chat history, removing trailing newline
            chat_history.append(msg.rstrip("\n"))
        except Exception as e:
            # If an error occurs, add an error message to chat history
            chat_history.append(f"[receiver crashed] {e}")
            break


def connect_to_server(nick, host, port):
    """Open a socket connection to the selected chat server."""
    # Declare global variables to modify them
    global client, nickname, current_server
    # Set the user's nickname
    nickname = nick
    # Set the current server information (name is None for now)
    current_server = (None, host, port)
    # Create a new socket for connecting to the chat server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the chat server at the specified host and port
    client.connect((host, port))
    # Send the nickname to the server as the first message
    client.sendall(nickname.encode("ascii"))
    # Start a background thread to receive messages from the server
    threading.Thread(target=receive_messages, daemon=True).start()


def fetch_servers_from_hub():
    """Query the directory hub for a list of available chat servers."""
    # Create a socket to connect to the hub
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the directory hub
    s.connect((HUB_HOST, HUB_PORT))
    # Send a LIST command to request available servers
    s.sendall(b"LIST\n")
    # Receive the response from the hub
    data = s.recv(4096).decode("ascii").strip()
    # Close the connection to the hub
    s.close()
    # Check if the response starts with "SERVERS:"
    if not data.startswith("SERVERS:"):
        # If not, return an empty list
        return []
    # Extract the payload after "SERVERS:"
    payload = data[len("SERVERS:"):]
    # If payload is empty, return empty list
    if not payload:
        return []
    # Split the payload by semicolons to get individual server entries
    entries = payload.split(";")
    # Initialize result list
    result = []
    # Parse each entry into name, host, port
    for e in entries:
        name, host, port = e.split(",")
        # Add the parsed server info to the result list
        result.append((name, host, int(port)))
    # Return the list of available servers
    return result


def leave(stdscr):
    """Confirm quitting the application and exit if the user confirms."""
    # Hide the cursor
    curses.curs_set(0)
    # Clear the screen
    stdscr.clear()
    # Define options for the confirmation dialog
    options = ["Yes","No"]
    # Start with the first option selected
    selected = 0
    # Loop to handle user input for selection
    while True:
        # Clear the screen
        stdscr.clear()
        # Display the confirmation prompt
        stdscr.addstr(0, 0, "== ARE YOU SURE YOU WANT TO LEAVE? ==")
        stdscr.addstr(1, 0, "Use UP/DOWN arrows and ENTER to select.")
        # Display the options with the selected one highlighted
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(3 + i, 0, f"> {option}")
            else:
                stdscr.addstr(3 + i, 0, f"  {option}")
        # Refresh the screen to show changes
        stdscr.refresh()
        # Get user input
        key = stdscr.getch()
        # Handle up arrow key
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        # Handle down arrow key
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        # Handle enter key
        elif key in [10, 13]:
            chosen = options[selected]
            break
    # If user chose "Yes", exit the application
    if chosen == "Yes":
        sys.exit()
    # If user chose "No", return to home page
    elif chosen == "No":
        home_page(stdscr)


def start_menu(stdscr):
    """Display the login/signup menu and return the chosen action."""
    # Hide the cursor
    curses.curs_set(0)
    # Clear the screen
    stdscr.clear()
    # Define menu options
    options = ["Log In", "Sign Up", "Quit"]
    # Start with the first option selected
    selected = 0
    # Loop to handle menu navigation
    while True:
        # Clear the screen
        stdscr.clear()
        # Display the welcome message
        stdscr.addstr(0, 0, "== Welcome to the Chatting Cafe ==")
        stdscr.addstr(1, 0, "Use UP/DOWN arrows and ENTER to select.")
        # Display options with selection highlight
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(3 + i, 0, f"> {option}")
            else:
                stdscr.addstr(3 + i, 0, f"  {option}")
        # Refresh the screen
        stdscr.refresh()
        # Get user input
        key = stdscr.getch()
        # Handle up arrow
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        # Handle down arrow
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        # Handle enter
        elif key in [10, 13]:
            return options[selected]


def log_in(stdscr):
    """Show the login form and authenticate the user."""
    # Show the cursor for input
    curses.curs_set(1)
    # Enable echo for password input
    curses.echo()

    # Loop until successful login
    while True:
        # Clear the screen
        stdscr.clear()
        # Display login header
        stdscr.addstr(1, 0, "== Log In ==")
        # Prompt for username
        stdscr.addstr(3, 0, "Enter username (max 20 chars): ")
        # Get username input
        username = stdscr.getstr(3, 32, 20).decode("utf-8")

        # Prompt for password
        stdscr.addstr(4, 0, "Enter password: ")
        # Get password input
        password = stdscr.getstr(4, 16, 50).decode("utf-8")

        # Validate the credentials using the auth module
        if validate_credentials(username, password):
            # Display success message
            stdscr.addstr(6, 0, "Login successful! Press any key to continue.")
            # Refresh and wait for key press
            stdscr.refresh()
            stdscr.getch()
            # Disable echo
            curses.noecho()
            # Proceed to server selection menu
            return server_menu(stdscr, username)
        else:
            # Display error message
            stdscr.addstr(6, 0, "Invalid username or password. Try again.")
            # Refresh and wait for key press
            stdscr.refresh()
            stdscr.getch()


def sign_up(stdscr):
    """Allow the user to register a new username and password."""
    # Show the cursor
    curses.curs_set(1)
    # Enable echo
    curses.echo()

    # Loop until successful signup
    while True:
        # Clear the screen
        stdscr.clear()
        # Display signup header
        stdscr.addstr(1, 0, "== Sign Up ==")
        # Prompt for username
        stdscr.addstr(3, 0, "Choose a username (max 20 chars): ")
        # Get username input
        username = stdscr.getstr(3, 34, 20).decode("utf-8")

        # Check if username already exists
        if username_exists(username):
            # Display error and wait
            stdscr.addstr(5, 0, "Username already exists. Press any key to try again.")
            stdscr.refresh()
            stdscr.getch()
            continue

        # Prompt for password
        stdscr.addstr(4, 0, "Choose a password: ")
        # Get password input
        password = stdscr.getstr(4, 19, 50).decode("utf-8")

        # Prompt for password confirmation
        stdscr.addstr(5, 0, "Confirm password: ")
        # Get confirmation input
        confirm = stdscr.getstr(5, 18, 50).decode("utf-8")

        # Check if passwords match
        if password != confirm:
            # Display error and wait
            stdscr.addstr(7, 0, "Passwords do not match. Press any key to retry.")
            stdscr.refresh()
            stdscr.getch()
            continue

        # Create the new user account
        create_user(username, password)
        # Display success message
        stdscr.addstr(7, 0, "Account created successfully! Press any key to continue.")
        # Refresh and wait
        stdscr.refresh()
        stdscr.getch()
        # Disable echo
        curses.noecho()
        # Proceed to server selection
        return server_menu(stdscr, username)

def server_menu(stdscr, username):
    """Fetch available servers and let the user choose one to connect to."""
    # Hide the cursor
    curses.curs_set(0)
    # Loop for server selection
    while True:
        # Clear the screen
        stdscr.clear()
        # Display header
        stdscr.addstr(0, 0, "== Select Server ==")
        stdscr.addstr(1, 0, "Fetching from hub...")
        # Refresh to show fetching message
        stdscr.refresh()

        # Fetch the list of available servers
        servers = fetch_servers_from_hub()
        # If no servers available
        if not servers:
            # Display message and options
            stdscr.addstr(3, 0, "No servers registered. Press q to quit or r to retry.")
            stdscr.refresh()
            # Get user input
            key = stdscr.getch()
            # If 'q' pressed, return (quit)
            if key in (ord('q'), ord('Q')):
                return
            else:
                # Retry fetching
                continue

        # Start with first server selected
        selected = 0
        # Loop for server selection
        while True:
            # Clear the screen
            stdscr.clear()
            # Display header
            stdscr.addstr(0, 0, "== Select Server ==")
            # Display each server option
            for i, (name, host, port) in enumerate(servers):
                prefix = "> " if i == selected else "  "
                stdscr.addstr(2 + i, 0, f"{prefix}{name} ({host}:{port})")
            # Display instructions
            stdscr.addstr(2 + len(servers) + 1, 0, "Use UP/DOWN, ENTER to select, q to quit.")
            # Refresh
            stdscr.refresh()

            # Get user input
            key = stdscr.getch()
            # Handle up arrow
            if key == curses.KEY_UP:
                selected = (selected - 1) % len(servers)
            # Handle down arrow
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % len(servers)
            # Handle enter (select server)
            elif key in [10, 13]:
                name, host, port = servers[selected]
                # Connect to the selected server
                connect_to_server(username, host, port)
                # Go to home page
                return home_page(stdscr)
            # Handle 'q' to quit
            elif key in (ord('q'), ord('Q')):
                return


def settings(stdscr):
    """Show settings options and route to the selected screen."""
    # Hide the cursor
    curses.curs_set(0)
    # Define settings options
    options = ["List of Commands", "Home Page","Quit"]
    # Start with first option
    selected = 0
    # Loop for selection
    while True:
        # Clear the screen
        stdscr.clear()
        # Display header
        stdscr.addstr(0, 0, "== Settings ==")
        stdscr.addstr(1, 0, "Use UP/DOWN arrows and ENTER to select.")
        # Display options
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(3 + i, 0, f"> {option}")
            else:
                stdscr.addstr(3 + i, 0, f"  {option}")
        # Refresh
        stdscr.refresh()
        # Get input
        key = stdscr.getch()
        # Handle up arrow
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        # Handle down arrow
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        # Handle enter
        elif key in [10, 13]:
            chosen =  options[selected]
            break
    # Route based on choice
    if chosen == "List of Commands":
        cmd_list(stdscr)
    elif chosen == "Home Page":
        home_page(stdscr)
    elif chosen == "Quit":
        leave(stdscr)


def cmd_list(stdscr):
    """Display available chat commands to the user."""
    # Hide the cursor
    curses.curs_set(0)
    # Clear the screen
    stdscr.clear()
    # Display header
    stdscr.addstr(0, 0, "== Commands ==")
    # Display command list
    stdscr.addstr(2, 0, "/dm <user> <message>   - send a direct message")
    stdscr.addstr(3, 0, "/addfriend <user>      - add a friend")
    stdscr.addstr(4, 0, "/friends               - show friends list")
    # Display return instruction
    stdscr.addstr(6, 0, "Press any key to return.")
    # Refresh and wait for key
    stdscr.refresh()
    stdscr.getch()
    # Return to home page
    home_page(stdscr)


def friends_page(stdscr):
    """Request and display the current friend list from the server."""
    # Hide the cursor
    curses.curs_set(0)
    # Check if connected to a server
    if not client:
        return
    # Send command to get friends list
    client.sendall(b"CMD:GET_FRIENDS\n")
    # Wait a bit for response
    curses.napms(150)

    # Find the latest friends response in chat history
    friends_lines = [line for line in chat_history if line.startswith("FRIENDS:")]
    if friends_lines:
        last = friends_lines[-1]
        payload = last[len("FRIENDS:"):]
        # Split into list, filter out empty strings
        flist = [f for f in payload.split(",") if f]
    else:
        flist = []

    # Clear screen and display friends
    stdscr.clear()
    stdscr.addstr(0, 0, "== Friends ==")
    if not flist:
        stdscr.addstr(2, 0, "No friends yet.")
    else:
        for i, f in enumerate(flist, start=2):
            stdscr.addstr(i, 0, f"- {f}")
    # Display return instruction
    stdscr.addstr(len(flist) + 3, 0, "Press any key to return.")
    # Refresh and wait
    stdscr.refresh()
    stdscr.getch()
    # Return to home page
    home_page(stdscr)


def DMS(stdscr):
    """Main chat screen where the user types messages and sees history."""
    # Show cursor for typing
    curses.curs_set(1)
    # Disable echo (we'll handle input manually)
    curses.noecho()
    # Enable keypad for special keys
    stdscr.keypad(True)
    # Set nodelay to avoid blocking on getch
    stdscr.nodelay(True)
    # List of random activities for the /activity command
    activities = [
        "If you could only eat one food for the rest of your life, what is it?",
        "If you had to delete all but three apps from your smartphone, which ones would you keep?",
        "What is your signature dance move?",
        "What was your dream job as a kid?",
        "If you could have any fictional character as a mentor at work, who would it be?",
        "What is one thing that is always on your desk?",
        "If you could teleport anywhere, where would you go right now?",
        "If you had to live in a video game, which one would it be?"
    ]

    # Get screen dimensions
    height, width = stdscr.getmaxyx()
    # Initialize message buffer
    message = ""

    # Main chat loop
    while True:
        # Clear the screen
        stdscr.clear()
        # Display header
        stdscr.addstr(0, 0, "== Chat == (type 'back' to return)")
        stdscr.addstr(1, 0, "Use /dm user message, /addfriend user, /friends")
        # Refresh header
        stdscr.refresh()

        # Display recent chat history
        start = max(0, len(chat_history) - (height - 5))
        for i, line in enumerate(chat_history[start:], start=3):
            if i >= height - 2:
                break
            stdscr.addstr(i, 0, line[:width - 1])

        # Display current message input
        stdscr.addstr(height - 2, 0, f"Message: {message}")
        # Move cursor to end of input
        stdscr.move(height - 2, 9 + len(message))
        # Refresh
        stdscr.refresh()

        # Get input (non-blocking)
        key = stdscr.getch()
        if key == -1:
            # No key pressed, sleep a bit to avoid high CPU
            curses.napms(100)
            continue
        elif key in [10, 13]:  # Enter key
            # If user types 'back', exit chat
            if message.lower() == "back":
                break
            # Handle /dm command
            if message.startswith("/dm "):
                parts = message.split(" ", 2)
                if len(parts) == 3:
                    target, dm = parts[1], parts[2]
                    # Send DM command to server
                    wire = f"CMD:DM:{target}:{dm}\n"
                    client.sendall(wire.encode("ascii"))
                else:
                    # Display usage error
                    chat_history.append("[client] usage: /dm user message")
            # Handle /addfriend command
            elif message.startswith("/addfriend "):
                parts = message.split(" ", 1)
                target = parts[1].strip()
                # Send add friend command
                wire = f"CMD:ADD_FRIEND:{target}\n"
                client.sendall(wire.encode("ascii"))
            # Handle /friends command
            elif message.strip() == "/friends":
                client.sendall(b"CMD:GET_FRIENDS\n")
            # Handle /activity command (placeholder)
            elif message.strip() == "/activity":
                # Add a random activity to chat history
                chat_history.append("[Server]: " + activities[random.randint(0, 7)])
            else:
                # Send normal message
                if client:
                    try:
                        client.sendall(f"{message}\n".encode("ascii"))
                    except Exception:
                        chat_history.append("[error] failed to send message")
                else:
                    chat_history.append("[error] not connected to server")
            # Clear the message buffer
            message = ""
        elif key == 127 or key == 8:  # Backspace
            # Remove last character from message
            message = message[:-1]
        elif 32 <= key <= 126:  # Printable characters
            # Add character to message
            message += chr(key)

    # Disable nodelay and return to home page
    stdscr.nodelay(False)
    home_page(stdscr)


def server_page(stdscr):
    """Show server selection again when requested from the home page."""
    # If user has a nickname, show server menu
    if nickname:
        server_menu(stdscr, nickname)
    else:
        # Otherwise, go to home page
        home_page(stdscr)


def home_page(stdscr):
    """Main landing page after a successful login."""
    # Hide the cursor
    curses.curs_set(0)
    # Clear the screen
    stdscr.clear()
    # Define home page options
    options = ["DM's", "Server", "Friends","Settings","Log Out","Quit"]
    # Start with first option
    selected = 0
    # Loop for selection
    while True:
        # Clear the screen
        stdscr.clear()
        # Display header
        stdscr.addstr(0, 0, "== Landing Page ==")
        stdscr.addstr(1, 0, "Use UP/DOWN arrows and ENTER to select.")
        # Display options
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(3 + i, 0, f"> {option}")
            else:
                stdscr.addstr(3 + i, 0, f"  {option}")
        # Refresh
        stdscr.refresh()
        # Get input
        key = stdscr.getch()
        # Handle up arrow
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        # Handle down arrow
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        # Handle enter
        elif key in [10, 13]:
            yes =  options[selected]
            break
    # Route based on choice
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
    """Entry point for curses; start the application menu."""
    # Hide the cursor
    curses.curs_set(0)
    # Disable nodelay
    stdscr.nodelay(0)
    # Enable keypad
    stdscr.keypad(True)
    # Show the start menu and get choice
    choice = start_menu(stdscr)
    # Route based on choice
    if choice == "Log In":
        log_in(stdscr)
    elif choice == "Sign Up":
        sign_up(stdscr)
    elif choice == "Quit":
        leave(stdscr)


# Start the curses application with the main function
curses.wrapper(main)