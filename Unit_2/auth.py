"""
Authentication helper module.

This module manages a simple JSON-backed user database stored in users.json.
It exposes functions to load and save users, validate login credentials,
check for existing usernames, and create new user accounts.
"""

# Import the json module for parsing and generating JSON data
import json
# Import the os module for file system operations like checking if a file exists
import os


def load_users(path="users.json"):
    """
    Load user data from a JSON file.

    If the file does not exist, return a default empty structure so the
    caller can still work with a predictable data format.
    """
    # Check if the specified file path exists on the file system
    if not os.path.exists(path):
        # If the file doesn't exist, return a default dictionary with an empty users list
        return {"users": []}
    # Open the file in read mode
    with open(path, "r") as f:
        # Load and parse the JSON data from the file
        return json.load(f)


def save_users(data, path="users.json"):
    """
    Save user data back to the JSON file.

    This overwrites the existing file with formatted JSON.
    """
    # Open the file in write mode (this will overwrite any existing content)
    with open(path, "w") as f:
        # Dump the data dictionary to the file as JSON with 4-space indentation for readability
        json.dump(data, f, indent=4)


def validate_credentials(username, password, path="users.json"):
    """
    Return True when the username and password match an existing user.
    """
    # Load the user data from the JSON file
    data = load_users(path)
    # Iterate through each user in the users list
    for user in data["users"]:
        # Check if both username and password match the provided credentials
        if user["username"] == username and user["password"] == password:
            # If a match is found, return True
            return True
    # If no matching user is found after checking all users, return False
    return False


def username_exists(username, path="users.json"):
    """
    Return True if a username already exists in the user database.
    """
    # Load the user data from the JSON file
    data = load_users(path)
    # Use any() with a generator expression to check if any user has the given username
    # This is more efficient than a full loop since it stops at the first match
    return any(user["username"] == username for user in data["users"])


def create_user(username, password, path="users.json"):
    """
    Add a new user record and save it to the JSON store.
    """
    # Load the existing user data from the JSON file
    data = load_users(path)
    # Append a new user dictionary to the users list with the provided username and password
    data["users"].append({"username": username, "password": password})
    # Save the updated data back to the JSON file
    save_users(data, path)