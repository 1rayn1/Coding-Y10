import json
import os

def load_users(path="users.json"):
    if not os.path.exists(path):
        return {"users": []}
    with open(path, "r") as f:
        return json.load(f)

def save_users(data, path="users.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def validate_credentials(username, password, path="users.json"):
    data = load_users(path)
    for user in data["users"]:
        if user["username"] == username and user["password"] == password:
            return True
    return False

def username_exists(username, path="users.json"):
    data = load_users(path)
    return any(user["username"] == username for user in data["users"])

def create_user(username, password, path="users.json"):
    data = load_users(path)
    data["users"].append({"username": username, "password": password})
    save_users(data, path)