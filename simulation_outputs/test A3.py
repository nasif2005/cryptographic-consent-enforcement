#!/usr/bin/env python3
"""
Scenario A3: HP adds a prescription to an existing episode (crypto-only latency)

Measures (excluding DB/IPFS/network):
- label construction
- symmetric key gen
- AES-GCM encrypt (16KB) with label as AAD
- wrapper generation (patient + HP) [optionally EAA]
- manifest append + serialization

Experiment: vary existing episode size N, append 1 new record.
"""

import os, json, time, csv, hashlib, statistics
from typing import Any, Dict, List, Tuple

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
    return sha256(gt_elem_bytes + label_bytes)

def xor_wrap(sym_key: bytes, wrap_key: bytes) -> bytes:
    if len(sym_key) != len(wrap_key):
        raise ValueError("length mismatch")
    return bytes(a ^ b for a, b in zip(sym_key, wrap_key))

def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> Tuple[bytes, bytes, bytes]:
    nonce = os.urandom(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    ctext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce, ctext, tag


# -----------------------------
# Pairing wrapper (PK in G2)
# -----------------------------

class PairingWrapper:
    def __init__(self):
        self.G = BpGroup()
        self.g1 = self.G.gen1()
        self.g2 = self.G.gen2()
        self.order: Bn = self.G.order()

    def rand_scalar(self) -> Bn:
        return (Bn.from_binary(os.urandom(32)) % self.order)

    def keygen_g2(self):
        x = self.rand_scalar()
        PK = self.g2.mul(x)
        return x, PK

    def wrap_for_recipient(self, sym_key_32: bytes, label_bytes: bytes, PK_g2):
        r = self.rand_scalar()
        U = self.g1.mul(r)
        Z = self.G.pair(U, PK_g2)
        wrap_key = kdf_wrap_key(Z.export(), label_bytes)
        wrapped_k = xor_wrap(sym_key_32, wrap_key)
        return U, wrapped_k


# -----------------------------
# Build an existing episode (not timed)
# -----------------------------

def build_existing_episode_manifest(N: int, episode_id: bytes, record_size_bytes: int) -> Dict[str, Any]:
    """
    Builds an existing manifest of size N with placeholder CID entries.
    This simulates: the manifest is already locally available (retrieved earlier).
    """
    record_type = b"Prescription"
    manifest = {"episode_id": episode_id.hex(), "records": []}

    for i in range(N):
        record_id = i.to_bytes(8, "big")
        label_bytes = episode_id + b"|" + record_type + b"|" + record_id
        # placeholder CID as if record already uploaded
        cid_placeholder = sha256(label_bytes).hex()
        manifest["records"].append({
            "label": label_bytes.hex(),
            "cid_cipher_record": cid_placeholder
        })

    return manifest


# -----------------------------
# Scenario A3: append 1 new record (timed)
# -----------------------------

def scenario_a3_append_one_record(
    existing_manifest: Dict[str, Any],
    pw: PairingWrapper,
    PK_pt,
    PK_hp,
    record_size_bytes: int,
    next_record_index: int
) -> float:
    """
    Measures crypto processing latency for appending ONE new record:
    keygen + AES-GCM encrypt + wrappers + manifest append + serialize.
    """
    episode_id = bytes.fromhex(existing_manifest["episode_id"])
    record_type = b"Prescription"
    plaintext = b"A" * record_size_bytes

    t0 = time.perf_counter()

    # 1) label construction for new record
    record_id = next_record_index.to_bytes(8, "big")
    label_bytes = episode_id + b"|" + record_type + b"|" + record_id

    # 2) symmetric key gen
    k = rand_bytes(32)

    # 3) AES-GCM encrypt (AAD = label)
    nonce, ctext, tag = aes_gcm_encrypt(k, plaintext, aad=label_bytes)

    # 4) wrappers (patient + HP)
    U_pt, w_pt = pw.wrap_for_recipient(k, label_bytes, PK_pt)
    U_hp, w_hp = pw.wrap_for_recipient(k, label_bytes, PK_hp)

    # 5) CID placeholder for "uploaded ciphertext"
    cid_placeholder = sha256(nonce + ctext + tag).hex()

    # 6) manifest update: append entry and serialize
    # (this captures the O(N) serialization cost)
    updated_manifest = {
        "episode_id": existing_manifest["episode_id"],
        "records": existing_manifest["records"] + [{
            "label": label_bytes.hex(),
            "cid_cipher_record": cid_placeholder,
            "nonce": nonce.hex(),
            "tag": tag.hex(),
            "w_patient": {"U": U_pt.export().hex(), "wrapped_k": w_pt.hex()},
            "w_hp": {"U": U_hp.export().hex(), "wrapped_k": w_hp.hex()},
        }]
    }

    _manifest_bytes = json.dumps(updated_manifest, separators=(",", ":")).encode("utf-8")

    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms


def run_a3(
    N_values=(10, 50, 100),
    R=30,
    record_size_bytes=16*1024,
    export_csv_path="a3_episode_update_samples.csv"
):
    pw = PairingWrapper()
    _, PK_pt = pw.keygen_g2()
    _, PK_hp = pw.keygen_g2()

    rows = []
    results = {}

    for N in N_values:
        episode_id = rand_bytes(16)
        base_manifest = build_existing_episode_manifest(N, episode_id, record_size_bytes)

        # warm-up
        _ = scenario_a3_append_one_record(base_manifest, pw, PK_pt, PK_hp, record_size_bytes, next_record_index=N)

        samples = []
        for rep in range(R):
            dt_ms = scenario_a3_append_one_record(base_manifest, pw, PK_pt, PK_hp, record_size_bytes, next_record_index=N)
            samples.append(dt_ms)
            rows.append({"N_existing": N, "rep": rep, "append1_latency_ms": dt_ms})

        results[N] = {
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "stdev_ms": statistics.pstdev(samples)
        }

    if export_csv_path:
        with open(export_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["N_existing", "rep", "append1_latency_ms"])
            w.writeheader()
            w.writerows(rows)

    return results


if __name__ == "__main__":
    N_values = (10, 50, 100)
    R = 30
    record_size_bytes = 16 * 1024

    results = run_a3(N_values=N_values, R=R, record_size_bytes=record_size_bytes)

    print("\nScenario A3: Episode Update (HP appends 1 prescription) — Crypto Only")
    print(f"Plaintext per new record: {record_size_bytes} bytes (16KB)")
    print(f"Repeats per N: {R}")
    print("Excludes DB/IPFS/network; measures local crypto + manifest update only.\n")

    for N in N_values:
        s = results[N]
        print(f"Existing episode size N = {N} records")
        print(f"  Mean append latency (ms):   {s['mean_ms']:.2f}")
        print(f"  Median append latency (ms): {s['median_ms']:.2f}")
        print(f"  Std dev (ms):               {s['stdev_ms']:.2f}")
        print()

    print("Raw samples saved to: a3_episode_update_samples.csv")
