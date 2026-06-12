import os
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_MASTER_KEY = b'MANTIS_FIELD_DEP_MASTER_KEY_0000'

def encrypt_data(data: bytes) -> bytes:
    """Encrypt data using AES-256-GCM."""
    aesgcm = AESGCM(_MASTER_KEY)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext

def decrypt_data(encrypted_data: bytes) -> bytes:
    """Decrypt data using AES-256-GCM."""
    aesgcm = AESGCM(_MASTER_KEY)
    nonce = encrypted_data[:12]
    ciphertext = encrypted_data[12:]
    return aesgcm.decrypt(nonce, ciphertext, None)

def save_encrypted_json(data: list, filepath: str):
    """Serialize JSON and save encrypted to disk."""
    json_bytes = json.dumps(data).encode('utf-8')
    encrypted_bytes = encrypt_data(json_bytes)
    with open(filepath, 'wb') as f:
        f.write(encrypted_bytes)

def load_encrypted_json(filepath: str) -> list:
    """Load encrypted JSON from disk and deserialize."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'rb') as f:
        encrypted_bytes = f.read()
    json_bytes = decrypt_data(encrypted_bytes)
    return json.loads(json_bytes.decode('utf-8'))
