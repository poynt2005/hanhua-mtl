"""Generate a passwordless ed25519 SSH keypair into a workspace path.

Avoids `ssh-keygen -N ""` (fails/hangs on Windows) and avoids writing to ~/.ssh.
Usage: python deploy/gen_ssh_key.py <private_key_path>
"""
import os
import stat
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "vast_ssh"
    key = ed25519.Ed25519PrivateKey.generate()
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode("ascii")

    with open(path, "wb") as f:
        f.write(priv)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    with open(path + ".pub", "w", encoding="ascii") as f:
        f.write(pub + "\n")

    print("private: %s" % path)
    print("public : %s.pub" % path)
    print("pubkey : %s" % pub)


if __name__ == "__main__":
    main()
