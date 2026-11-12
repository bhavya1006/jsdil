"""Turns a password into the form the database stores, so it never holds a password."""
import hashlib

from config import config


def hash_password(plain_password, password_salt):
    """A salted key derivation of the password, written as hexadecimal text."""
    return hashlib.pbkdf2_hmac(config.PASSWORD_HASH_ALGORITHM, plain_password.encode("utf-8"),
                               password_salt.encode("utf-8"),
                               config.PASSWORD_HASH_ROUNDS).hex()
