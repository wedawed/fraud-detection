# generate_hashed_passwords.py

import bcrypt

def hash_password(plain_password):
    """
    Hashes a plaintext password using bcrypt.
    
    Parameters:
    - plain_password (str): The plaintext password to hash.
    
    Returns:
    - bytes: The hashed password.
    """
    return bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())

# List of users and their plaintext passwords
users_plain = {
    "admin": "password123",
    "user1": "userpassword",
    # Add more users as needed
}

# Generate and print hashed passwords
users_hashed = {}
for username, pwd in users_plain.items():
    hashed = hash_password(pwd)
    users_hashed[username] = {
        "name": f"{username.capitalize()} User",
        "password": hashed
    }
    print(f"Username: {username}\nHashed Password: {hashed}\n")

# Now, you can use `users_hashed` to update your `auth.py`
