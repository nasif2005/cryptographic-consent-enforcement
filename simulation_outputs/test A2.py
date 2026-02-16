#!/usr/bin/env python3
"""
Scenario A2: Patient Access to an Existing Episode (crypto processing latency only)

Measures (excluding DB/IPFS/network):
- Manifest parsing (JSON decode)
- Wrapper unwrapping for patient (pairing + KDF + XOR unwrap)
- AES-GCM decryption of 16KB records with label as AAD

Scaling: N ∈ {1, 10, 50}
Outputs: printed summary + CSV of per-run samples.

Requirements:
  pip install pycryptodome bplib petlib
"""

import os
import json
import time
import csv
import hashlib
import statistics
from typing import Any, Dict, List, Tuple

from Crypto.Cipher import AES
from bplib.bp import BpGroup, G1Elem
from petlib.bn import Bn


# -----------------------------
# Utilities
# -----------------------------

def rand_bytes(n: int) -> bytes:
    return os.urandom(n)

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def kdf_wrap_key(gt_elem_bytes: bytes, label_bytes: bytes) -> bytes:
    # 32-byte wrap key derived from pairing result + label binding
    return sha256(gt_elem_bytes + label_bytes)

def xor_wrap(sym_key: bytes, wrap_key: bytes) -> bytes:
    if len(sym_key) != len(wrap_key):
        raise ValueError("sym_key and wrap_key must be same length")
    return bytes(a ^ b for a, b in zip(sym_key, wrap_key))

def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> Tuple[bytes, bytes, bytes]:
    nonce = os.urandom(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    ctext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce, ctext, tag

def aes_gcm_decrypt(key: bytes, nonce: bytes, ctext: bytes, tag: bytes, aad: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    return cipher.decrypt_and_verify(ctext, tag)


# -----------------------------
# Pairing-based wrapper (PK in G2)
# -----------------------------

class PairingWrapper:
    def __init__(self):
        self.G = BpGroup()
        self.g1 = self.G.gen1()
        self.g2 = self.G.gen2()
        self.order: Bn = self.G.order()

    def rand_scalar(self) -> Bn:
        return (Bn.from_binary(os.urandom(32)) % self.order)

    def keygen_g2(self) -> Tuple[Bn, Any]:
        """
        secret x in ZR (Bn), public PK = g2^x in G2.
        """
        x = self.rand_scalar()
        PK = self.g2.mul(x)
        return x, PK

    def wrap_for_recipient(self, sym_key_32: bytes, label_bytes: bytes, PK_g2) -> Tuple[Any, bytes]:
        """
        Produces wrapper (U, wrapped_k):
          U = g1^r
          Z = e(U, PK)
          wrap_key = H(Z || label)
          wrapped_k = k XOR wrap_key
        """
        if len(sym_key_32) != 32:
            raise ValueError("sym_key must be 32 bytes (AES-256 key)")

        r = self.rand_scalar()
        U = self.g1.mul(r)            # G1 element
        Z = self.G.pair(U, PK_g2)     # GT element
        wrap_key = kdf_wrap_key(Z.export(), label_bytes)
        wrapped_k = xor_wrap(sym_key_32, wrap_key)
        return U, wrapped_k

    def unwrap_for_recipient(self, U_g1, wrapped_k: bytes, label_bytes: bytes, x_recipient: Bn) -> bytes:
        """
        Recipient side:
          compute Z = e(U, g2^x)
          wrap_key = H(Z || label)
          k = wrapped_k XOR wrap_key
        """
        PK_from_secret = self.g2.mul(x_recipient)
        Z = self.G.pair(U_g1, PK_from_secret)
        wrap_key = kdf_wrap_key(Z.export(), label_bytes)
        k = xor_wrap(wrapped_k, wrap_key)
        return k


# -----------------------------
# Build a local episode object (setup for A2)
# -----------------------------

def build_episode_local(
    N: int,
    record_size_bytes: int,
    pw: PairingWrapper,
    PK_pt,
    record_type: bytes = b"Prescription",
) -> Dict[str, Any]:
    """
    Creates an episode locally and returns:
      - manifest_bytes: JSON bytes that include wrapper bytes (U export + wrapped_k)
      - records_runtime: list holding the *actual* ciphertext bytes (nonce, ctext, tag)
        so A2 decrypt can run without DB/IPFS/network.
    """
    episode_id = rand_bytes(16)
    plaintext = b"A" * record_size_bytes

    records_runtime = []
    manifest = {"episode_id": episode_id.hex(), "records": []}

    for i in range(N):
        record_id = i.to_bytes(8, "big")
        label_bytes = episode_id + b"|" + record_type + b"|" + record_id

        # Record symmetric key (AES-256)
        k_i = rand_bytes(32)

        # Encrypt record using AES-GCM, bind label as AAD
        nonce, ctext, tag = aes_gcm_encrypt(k_i, plaintext, aad=label_bytes)

        # Patient wrapper only (A2 is patient access)
        U_pt, wrapped_k = pw.wrap_for_recipient(k_i, label_bytes, PK_pt)

        # CID placeholder: hash ciphertext (since storage excluded)
        cid_placeholder = sha256(nonce + ctext + tag).hex()

        # manifest entry stores only serialized wrapper and crypto metadata
        manifest["records"].append({
            "label": label_bytes.hex(),
            "cid_cipher_record": cid_placeholder,
            "nonce": nonce.hex(),
            "tag": tag.hex(),
            "w_patient": {"U": U_pt.export().hex(), "wrapped_k": wrapped_k.hex()},
        })

        # runtime data keeps actual ciphertext bytes for decrypt
        records_runtime.append({
            "label_bytes": label_bytes,
            "nonce": nonce,
            "ctext": ctext,
            "tag": tag,
        })

    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    return {"manifest_bytes": manifest_bytes, "records_runtime": records_runtime}


# -----------------------------
# Scenario A2: Patient access timing
# -----------------------------

def scenario_a2_patient_access(episode_obj: Dict[str, Any], pw: PairingWrapper, x_patient: Bn) -> float:
    """
    Measures crypto processing latency:
      parse manifest + for each record unwrap key + AES-GCM decrypt.
    """
    t0 = time.perf_counter()

    manifest = json.loads(episode_obj["manifest_bytes"].decode("utf-8"))
    runtime_records = episode_obj["records_runtime"]

    for i, entry in enumerate(manifest["records"]):
        label_bytes = bytes.fromhex(entry["label"])

        # Deserialize U (G1 element)
        U_bytes = bytes.fromhex(entry["w_patient"]["U"])
        U = G1Elem.from_bytes(U_bytes, pw.G)

        wrapped_k = bytes.fromhex(entry["w_patient"]["wrapped_k"])
        k_i = pw.unwrap_for_recipient(U, wrapped_k, label_bytes, x_patient)

        rr = runtime_records[i]
        _pt = aes_gcm_decrypt(k_i, rr["nonce"], rr["ctext"], rr["tag"], aad=label_bytes)

    t1 = time.perf_counter()
    return (t1 - t0)


def run_a2(
    N_values=(1, 10, 50),
    R: int = 30,
    record_size_bytes: int = 16 * 1024,
    export_csv_path: str = "a2_patient_access_samples.csv",
) -> Dict[int, Dict[str, float]]:
    """
    For each N: warm-up once, then run R repeats.
    Exports raw samples to CSV and returns aggregated stats.
    """
    pw = PairingWrapper()

    # Patient keypair (PK in G2)
    x_pt, PK_pt = pw.keygen_g2()

    results: Dict[int, Dict[str, float]] = {}
    raw_rows: List[Dict[str, Any]] = []

    for N in N_values:
        # warm-up
        warm_ep = build_episode_local(N, record_size_bytes, pw, PK_pt)
        _ = scenario_a2_patient_access(warm_ep, pw, x_pt)

        samples_ms: List[float] = []
        for rep in range(R):
            ep = build_episode_local(N, record_size_bytes, pw, PK_pt)
            dt_ms = scenario_a2_patient_access(ep, pw, x_pt) * 1000.0
            samples_ms.append(dt_ms)
            raw_rows.append({"N": N, "rep": rep, "latency_ms": dt_ms})

        mean_ms = statistics.mean(samples_ms)
        median_ms = statistics.median(samples_ms)
        stdev_ms = statistics.pstdev(samples_ms)
        amortized = mean_ms / N

        results[N] = {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "stdev_ms": stdev_ms,
            "amortized_ms_per_record_mean": amortized,
        }

    if export_csv_path:
        with open(export_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["N", "rep", "latency_ms"])
            w.writeheader()
            w.writerows(raw_rows)

    return results


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    N_values = (1, 10, 50)
    R = 30
    record_size_bytes = 16 * 1024

    results = run_a2(
        N_values=N_values,
        R=R,
        record_size_bytes=record_size_bytes,
        export_csv_path="a2_patient_access_samples.csv",
    )

    print("\nScenario A2: Patient Access (Crypto Processing Latency Only)")
    print(f"Plaintext per record: {record_size_bytes} bytes (16KB)")
    print(f"Repeats per N: {R}")
    print("Excludes DB/IPFS/network; measures local crypto only.\n")

    for N in N_values:
        s = results[N]
        print(f"N = {N} records")
        print(f"  Mean latency (ms):            {s['mean_ms']:.2f}")
        print(f"  Median latency (ms):          {s['median_ms']:.2f}")
        print(f"  Std dev (ms):                 {s['stdev_ms']:.2f}")
        print(f"  Amortized per-record (ms):    {s['amortized_ms_per_record_mean']:.2f}")
        print()

    print("Raw samples saved to: a2_patient_access_samples.csv")
