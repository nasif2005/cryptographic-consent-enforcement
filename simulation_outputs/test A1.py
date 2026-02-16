#!/usr/bin/env python3
"""
Scenario A1: Episode Creation (Crypto Processing Latency Only)

Measures (excluding DB/IPFS/network):
- Per-record: sym key gen + AES-GCM encrypt(16KB) + label build
            + 3 pairing-based wrappers (patient, HP, EAA)
- Per-episode: manifest construction (JSON serialization)

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
from typing import Dict, Tuple, Any, List

from Crypto.Cipher import AES
from bplib.bp import BpGroup
from petlib.bn import Bn


# -----------------------------
# Utilities
# -----------------------------

def rand_bytes(n: int) -> bytes:
    return os.urandom(n)

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def kdf_wrap_key(gt_elem_bytes: bytes, label_bytes: bytes) -> bytes:
    # 32 bytes -> wrap key
    return sha256(gt_elem_bytes + label_bytes)

def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> Tuple[bytes, bytes, bytes]:
    """
    AES-GCM encryption with Associated Data binding.
    Returns (nonce, ciphertext, tag).
    """
    nonce = os.urandom(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce, ciphertext, tag

def xor_wrap(sym_key: bytes, wrap_key: bytes) -> bytes:
    """
    Lightweight key wrap for benchmarking.
    (Document in paper: DEM wrapping simplified to isolate pairing/KDF overhead.)
    """
    if len(sym_key) != len(wrap_key):
        raise ValueError("sym_key and wrap_key must be same length for xor_wrap")
    return bytes(a ^ b for a, b in zip(sym_key, wrap_key))


# -----------------------------
# Pairing-based wrapper (PK in G2)
# -----------------------------

class PairingWrapper:
    def __init__(self):
        self.G = BpGroup()
        self.g1 = self.G.gen1()
        self.g2 = self.G.gen2()
        # Group order as Bn; use it to reduce random scalars.
        self.order: Bn = self.G.order()

    def rand_scalar(self) -> Bn:
        # Generate scalar uniformly-ish by reducing random bytes mod order.
        # This is fine for benchmarking and avoids petlib bounds issues.
        return (Bn.from_binary(os.urandom(32)) % self.order)

    def keygen_g2(self) -> Tuple[Bn, Any]:
        """
        Secret x in ZR (Bn), public PK = g2^x in G2.
        """
        x = self.rand_scalar()
        PK = self.g2.mul(x)   # ✅ x is Bn
        return x, PK

    def wrap_key_for_recipient(self, sym_key_32: bytes, label_bytes: bytes, PK_g2) -> Tuple[Any, bytes]:
        """
        Wrapper: (U, wrapped_sym_key)
          r ← ZR
          U = g1^r in G1
          Z = e(U, PK) in GT
          wrap_key = H(serialize(Z) || label)
          wrapped_sym_key = sym_key XOR wrap_key
        """
        if len(sym_key_32) != 32:
            raise ValueError("sym_key_32 must be 32 bytes")

        r = self.rand_scalar()
        U = self.g1.mul(r)
        Z = self.G.pair(U, PK_g2)
        z_bytes = Z.export()
        wrap_key = kdf_wrap_key(z_bytes, label_bytes)  # 32 bytes
        wrapped = xor_wrap(sym_key_32, wrap_key)
        return (U, wrapped)


# -----------------------------
# Scenario A1 benchmark
# -----------------------------

def scenario_a1_create_episode(
    N: int,
    record_size_bytes: int,
    pw: PairingWrapper,
    PK_pt,
    PK_hp,
    PK_eaa,
    record_type: bytes = b"Prescription",
) -> float:
    """
    End-to-end crypto processing latency for creating a new episode with N records.
    Excludes storage/network.

    Timing boundary:
      start -> per-record keygen + AES-GCM + label + 3 wrappers -> manifest serialize -> stop
    """
    episode_id = rand_bytes(16)
    plaintext = b"A" * record_size_bytes

    manifest = {
        "episode_id": episode_id.hex(),
        "records": []
    }

    t0 = time.perf_counter()

    for i in range(N):
        record_id = i.to_bytes(8, "big")
        label_bytes = episode_id + b"|" + record_type + b"|" + record_id

        # Symmetric key for record (AES-256)
        k_i = rand_bytes(32)

        # AES-GCM encrypt record; bind to label using AAD
        nonce, ctext, tag = aes_gcm_encrypt(k_i, plaintext, aad=label_bytes)

        # Pairing-based wrappers (patient, HP, EAA)
        w_pt = pw.wrap_key_for_recipient(k_i, label_bytes, PK_pt)
        w_hp = pw.wrap_key_for_recipient(k_i, label_bytes, PK_hp)
        w_eaa = pw.wrap_key_for_recipient(k_i, label_bytes, PK_eaa)

        # CID placeholder (since DB/IPFS excluded): hash ciphertext material
        cid_placeholder = sha256(nonce + ctext + tag).hex()

        manifest["records"].append({
            "label": label_bytes.hex(),
            "cid_cipher_record": cid_placeholder,
            "nonce": nonce.hex(),
            "tag": tag.hex(),
            "w_patient": {"U": w_pt[0].export().hex(), "wrapped_k": w_pt[1].hex()},
            "w_hp":      {"U": w_hp[0].export().hex(), "wrapped_k": w_hp[1].hex()},
            "w_eaa":     {"U": w_eaa[0].export().hex(), "wrapped_k": w_eaa[1].hex()},
        })

    # Manifest construction cost (serialization)
    _manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    t1 = time.perf_counter()
    return (t1 - t0)  # seconds


def run_a1(
    N_values=(1, 10, 50),
    R: int = 30,
    record_size_bytes: int = 16 * 1024,
    export_csv_path: str = "a1_episode_creation_samples.csv",
) -> Dict[int, Dict[str, float]]:
    """
    Runs A1 for each N with R repeats, prints summary, exports raw samples to CSV.
    Returns aggregated stats per N.
    """
    pw = PairingWrapper()

    # Long-term recipient keys (public keys in G2)
    _, PK_pt = pw.keygen_g2()
    _, PK_hp = pw.keygen_g2()
    _, PK_eaa = pw.keygen_g2()

    raw_rows: List[Dict[str, Any]] = []
    results: Dict[int, Dict[str, float]] = {}

    for N in N_values:
        # warm-up (not counted)
        _ = scenario_a1_create_episode(N, record_size_bytes, pw, PK_pt, PK_hp, PK_eaa)

        samples_ms: List[float] = []
        for rep in range(R):
            dt_s = scenario_a1_create_episode(N, record_size_bytes, pw, PK_pt, PK_hp, PK_eaa)
            dt_ms = dt_s * 1000.0
            samples_ms.append(dt_ms)
            raw_rows.append({"N": N, "rep": rep, "latency_ms": dt_ms})

        mean_ms = statistics.mean(samples_ms)
        median_ms = statistics.median(samples_ms)
        stdev_ms = statistics.pstdev(samples_ms)  # population stdev
        amortized = mean_ms / N

        results[N] = {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "stdev_ms": stdev_ms,
            "amortized_ms_per_record_mean": amortized,
        }

    # Export CSV
    if export_csv_path:
        with open(export_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["N", "rep", "latency_ms"])
            w.writeheader()
            w.writerows(raw_rows)

    return results


if __name__ == "__main__":
    N_values = (1, 10, 50)
    R = 30
    record_size_bytes = 16 * 1024

    results = run_a1(
        N_values=N_values,
        R=R,
        record_size_bytes=record_size_bytes,
        export_csv_path="a1_episode_creation_samples.csv",
    )

    print("\nScenario A1: Episode Creation (Crypto Processing Latency Only)")
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

    print("Raw samples saved to: a1_episode_creation_samples.csv")
