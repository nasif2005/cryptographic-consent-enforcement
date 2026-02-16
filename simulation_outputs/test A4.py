#!/usr/bin/env python3
"""
Scenario A4: Patient shares specific element(s) of the health record
Option 2: Benchmark-only proxy transform (simplified, CORRECT decryption)

Measures (excluding DB/IPFS/network):
A4(b) Patient-side:
  - ReKeyGen latency over S labels (pairing + hash model)

A4(c) Delegate-side:
  - Proxy transform latency over S labels (XOR rewrap)
  - Delegate unwrap + AES-GCM decrypt over S labels (pairing + hash + AES-GCM)

Scaling: S ∈ {1, 10, 50}
Record size: 16KB
Repeats: R

Paper disclaimer (recommended):
We model proxy re-encryption as a benchmark-only wrapper transformation:
the proxy updates a patient-bound wrapped key into a delegate-bound wrapped key
using patient-issued rekey material, and we measure its cryptographic cost.
"""

import os
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

def kdf(gt_bytes: bytes, label_bytes: bytes) -> bytes:
    # 32-byte mask
    return sha256(gt_bytes + label_bytes)

def xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("xor length mismatch")
    return bytes(x ^ y for x, y in zip(a, b))


# -----------------------------
# Pairing environment + patient wrapper
# -----------------------------

class PairingEnv:
    def __init__(self):
        self.G = BpGroup()
        self.g1 = self.G.gen1()
        self.g2 = self.G.gen2()
        self.order: Bn = self.G.order()

    def rand_scalar(self) -> Bn:
        return (Bn.from_binary(os.urandom(32)) % self.order)

    def keygen_g2(self) -> Tuple[Bn, Any]:
        """
        Secret x in ZR (Bn), public PK = g2^x in G2.
        """
        x = self.rand_scalar()
        PK = self.g2.mul(x)
        return x, PK

    def patient_wrap_key(self, k: bytes, label_bytes: bytes, PK_pt) -> Dict[str, str]:
        """
        Produce patient wrapper for record key k:
          r <- ZR
          U = g1^r
          mask_pt = H( pair(U, PK_pt) || label )
          wrapped_k_pt = k XOR mask_pt
        Returns {"U": hexbytes, "wrapped_k": hexbytes}
        """
        if len(k) != 32:
            raise ValueError("k must be 32 bytes")

        r = self.rand_scalar()
        U = self.g1.mul(r)                # G1 element
        Z = self.G.pair(U, PK_pt)         # GT
        mask_pt = kdf(Z.export(), label_bytes)
        wrapped_k = xor_bytes(k, mask_pt)

        return {"U": U.export().hex(), "wrapped_k": wrapped_k.hex()}

    def derive_mask(self, U_hex: str, PK_rec, label_bytes: bytes) -> bytes:
        """
        Compute mask_rec = H( pair(U, PK_rec) || label ).
        """
        U = G1Elem.from_bytes(bytes.fromhex(U_hex), self.G)
        Z = self.G.pair(U, PK_rec)
        return kdf(Z.export(), label_bytes)


# -----------------------------
# A4 benchmark-only PRE functions (correct)
# -----------------------------

def benchmark_rekeygen(env: PairingEnv, w_patient: Dict[str, str], label_bytes: bytes, PK_pt, PK_del) -> bytes:
    """
    Patient-side rekeygen (measured):
      mask_pt  = H(pair(U, PK_pt)||label)
      mask_del = H(pair(U, PK_del)||label)
      delta = mask_pt XOR mask_del
    Returns delta (32 bytes).
    """
    U_hex = w_patient["U"]
    mask_pt = env.derive_mask(U_hex, PK_pt, label_bytes)
    mask_del = env.derive_mask(U_hex, PK_del, label_bytes)
    delta = xor_bytes(mask_pt, mask_del)
    return delta

def benchmark_transform(w_patient: Dict[str, str], delta: bytes) -> Dict[str, str]:
    """
    Proxy transform (measured, lightweight):
      wrapped_k_del = wrapped_k_pt XOR delta
    Keeps U unchanged.
    """
    wrapped_k_pt = bytes.fromhex(w_patient["wrapped_k"])
    wrapped_k_del = xor_bytes(wrapped_k_pt, delta)
    return {"U": w_patient["U"], "wrapped_k": wrapped_k_del.hex()}

def benchmark_delegate_unwrap(env: PairingEnv, w_delegate: Dict[str, str], label_bytes: bytes, PK_del) -> bytes:
    """
    Delegate unwrap (measured):
      mask_del = H(pair(U, PK_del)||label)
      k = wrapped_k_del XOR mask_del
    Returns k (32 bytes), enabling AES-GCM decrypt.
    """
    U_hex = w_delegate["U"]
    wrapped_k_del = bytes.fromhex(w_delegate["wrapped_k"])
    mask_del = env.derive_mask(U_hex, PK_del, label_bytes)
    k = xor_bytes(wrapped_k_del, mask_del)
    return k


# -----------------------------
# Build S labeled records for A4 (setup, not timed)
# -----------------------------

def build_episode_for_a4(S: int, env: PairingEnv, PK_pt, record_size_bytes: int = 16 * 1024) -> Dict[str, Any]:
    """
    Creates S labeled records. For each label:
      - encrypt record under random k (AES-GCM, AAD=label)
      - create patient wrapper w_patient for k
    """
    episode_id = rand_bytes(16)
    record_type = b"Prescription"
    plaintext = b"A" * record_size_bytes

    items = []
    for i in range(S):
        record_id = i.to_bytes(8, "big")
        label_bytes = episode_id + b"|" + record_type + b"|" + record_id

        k = rand_bytes(32)
        nonce, ctext, tag = aes_gcm_encrypt(k, plaintext, aad=label_bytes)
        w_patient = env.patient_wrap_key(k, label_bytes, PK_pt)

        items.append({
            "label_bytes": label_bytes,
            "nonce": nonce,
            "ctext": ctext,
            "tag": tag,
            "w_patient": w_patient,
        })

    return {"items": items}


# -----------------------------
# Run A4(b) and A4(c)
# -----------------------------

def run_a4(
    S_values=(1, 10, 50),
    R: int = 30,
    record_size_bytes: int = 16 * 1024,
    csv_rekey: str = "a4b_rekeygen_samples.csv",
    csv_access: str = "a4c_delegate_access_samples.csv",
):
    env = PairingEnv()

    # Patient + Delegate keypairs
    _, PK_pt = env.keygen_g2()
    _, PK_del = env.keygen_g2()

    rekey_rows: List[Dict[str, Any]] = []
    access_rows: List[Dict[str, Any]] = []

    rekey_stats: Dict[int, Dict[str, float]] = {}
    access_stats: Dict[int, Dict[str, float]] = {}

    for S in S_values:
        # warm-up
        ep = build_episode_for_a4(S, env, PK_pt, record_size_bytes)
        _ = benchmark_rekeygen(env, ep["items"][0]["w_patient"], ep["items"][0]["label_bytes"], PK_pt, PK_del)

        samples_rekey_ms: List[float] = []
        samples_access_ms: List[float] = []

        for rep in range(R):
            ep = build_episode_for_a4(S, env, PK_pt, record_size_bytes)

            # ---- A4(b): Patient ReKeyGen over S labels ----
            t0 = time.perf_counter()
            deltas = []
            for it in ep["items"]:
                delta = benchmark_rekeygen(env, it["w_patient"], it["label_bytes"], PK_pt, PK_del)
                deltas.append(delta)
            t1 = time.perf_counter()

            dt_rekey_ms = (t1 - t0) * 1000.0
            samples_rekey_ms.append(dt_rekey_ms)
            rekey_rows.append({"S_shared": S, "rep": rep, "rekey_total_ms": dt_rekey_ms})

            # ---- A4(c): Proxy Transform + Delegate Unwrap + AES-GCM decrypt ----
            t2 = time.perf_counter()
            for it, delta in zip(ep["items"], deltas):
                w_del = benchmark_transform(it["w_patient"], delta)
                k_del = benchmark_delegate_unwrap(env, w_del, it["label_bytes"], PK_del)
                _pt = aes_gcm_decrypt(k_del, it["nonce"], it["ctext"], it["tag"], aad=it["label_bytes"])
            t3 = time.perf_counter()

            dt_access_ms = (t3 - t2) * 1000.0
            samples_access_ms.append(dt_access_ms)
            access_rows.append({"S_shared": S, "rep": rep, "delegate_total_ms": dt_access_ms})

        rekey_stats[S] = {
            "mean_ms": statistics.mean(samples_rekey_ms),
            "median_ms": statistics.median(samples_rekey_ms),
            "stdev_ms": statistics.pstdev(samples_rekey_ms),
            "amortized_ms_per_label": statistics.mean(samples_rekey_ms) / S,
        }
        access_stats[S] = {
            "mean_ms": statistics.mean(samples_access_ms),
            "median_ms": statistics.median(samples_access_ms),
            "stdev_ms": statistics.pstdev(samples_access_ms),
            "amortized_ms_per_label": statistics.mean(samples_access_ms) / S,
        }

    # Save CSVs
    with open(csv_rekey, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["S_shared", "rep", "rekey_total_ms"])
        w.writeheader()
        w.writerows(rekey_rows)

    with open(csv_access, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["S_shared", "rep", "delegate_total_ms"])
        w.writeheader()
        w.writerows(access_rows)

    return rekey_stats, access_stats


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    S_values = (1, 10, 50)
    R = 30
    record_size_bytes = 16 * 1024

    rekey_stats, access_stats = run_a4(
        S_values=S_values,
        R=R,
        record_size_bytes=record_size_bytes,
        csv_rekey="a4b_rekeygen_samples.csv",
        csv_access="a4c_delegate_access_samples.csv",
    )

    print("\nScenario A4(b): Patient Delegation (ReKeyGen) — Crypto Only")
    print(f"Record size: {record_size_bytes} bytes (16KB), repeats: {R}\n")
    for S in S_values:
        s = rekey_stats[S]
        print(f"S = {S} shared labels")
        print(f"  Mean total (ms):          {s['mean_ms']:.2f}")
        print(f"  Median total (ms):        {s['median_ms']:.2f}")
        print(f"  Std dev (ms):             {s['stdev_ms']:.2f}")
        print(f"  Amortized per label (ms): {s['amortized_ms_per_label']:.2f}\n")

    print("Scenario A4(c): Delegate Access (Transform + Decrypt) — Crypto Only\n")
    for S in S_values:
        s = access_stats[S]
        print(f"S = {S} shared labels")
        print(f"  Mean total (ms):          {s['mean_ms']:.2f}")
        print(f"  Median total (ms):        {s['median_ms']:.2f}")
        print(f"  Std dev (ms):             {s['stdev_ms']:.2f}")
        print(f"  Amortized per label (ms): {s['amortized_ms_per_label']:.2f}\n")

    print("Raw samples saved to: a4b_rekeygen_samples.csv and a4c_delegate_access_samples.csv")
