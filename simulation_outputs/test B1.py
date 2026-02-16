#!/usr/bin/env python3
"""
Scenario B1: Emergency (Evidence-Driven) Access — Crypto Processing Latency Only

Fix N_total = 1000 records in an episode.
Vary E_release ∈ {1, 10, 50} records released under bounded emergency scope.
Measure crypto processing latency from start of wrapper unwrap to plaintext recovery
for E_release records. Exclude DB/IPFS/network.

Outputs: b1_emergency_access_samples.csv
Columns: N_total, E_release, rep, latency_ms
"""

import csv
import time
import secrets
from dataclasses import dataclass
from typing import List, Tuple

from Crypto.Cipher import AES
from Crypto.Hash import SHA256

from bplib.bp import BpGroup
from petlib.bn import Bn


# ---------------------------
# Utility: AES-GCM
# ---------------------------
def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> Tuple[bytes, bytes, bytes]:
    nonce = secrets.token_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    ctext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce, ctext, tag


def aes_gcm_decrypt(key: bytes, nonce: bytes, ctext: bytes, tag: bytes, aad: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    return cipher.decrypt_and_verify(ctext, tag)


def kdf_gt_to_32bytes(gt_elem_bytes: bytes) -> bytes:
    return SHA256.new(gt_elem_bytes).digest()  # 32 bytes


def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


# ---------------------------
# Pairing-based wrapper for EAA (keep U as an element, not bytes)
# ---------------------------
@dataclass
class EAAWrapper:
    """
    Wrapper is (U, V) where:
      U = g1^r  (G1 element, kept in-memory)
      V = k XOR H( e(U, PK_eaa) )
    """
    U: object   # G1 element
    V: bytes    # 32 bytes


class PairingWrapper:
    def __init__(self):
        self.G = BpGroup()
        self.g1 = self.G.gen1()
        self.g2 = self.G.gen2()
        self.order: Bn = self.G.order()

    def rand_scalar(self) -> Bn:
        return self.order.random()

    def keygen_g2(self):
        x = self.rand_scalar()     # Bn
        PK = self.g2.mul(x)        # G2 element
        return x, PK

    def wrap_key_for_eaa(self, k32: bytes, PK_eaa_g2) -> EAAWrapper:
        r = self.rand_scalar()
        U = self.g1.mul(r)                 # G1 element
        gt = self.G.pair(U, PK_eaa_g2)     # GT element
        mask = kdf_gt_to_32bytes(gt.export())
        V = xor_bytes(k32, mask)
        return EAAWrapper(U=U, V=V)

    def unwrap_key_for_eaa(self, wrapper: EAAWrapper, sk_eaa: Bn) -> bytes:
        # Use U directly (no deserialize)
        sk_elem = self.g2.mul(sk_eaa)      # g2^sk
        gt = self.G.pair(wrapper.U, sk_elem)
        mask = kdf_gt_to_32bytes(gt.export())
        return xor_bytes(wrapper.V, mask)


# ---------------------------
# Episode generation (setup, not timed)
# ---------------------------
@dataclass
class RecordItem:
    label_bytes: bytes
    nonce: bytes
    ctext: bytes
    tag: bytes
    w_eaa: EAAWrapper


def build_episode(
    N_total: int,
    record_size_bytes: int,
    pw: PairingWrapper,
    PK_eaa
) -> List[RecordItem]:
    episode: List[RecordItem] = []
    plaintext = b"A" * record_size_bytes  # fixed payload size

    for i in range(N_total):
        label = f"ep0|note|{i}".encode("utf-8")
        k32 = secrets.token_bytes(32)

        nonce, ctext, tag = aes_gcm_encrypt(k32, plaintext, aad=label)
        w_eaa = pw.wrap_key_for_eaa(k32, PK_eaa)

        episode.append(RecordItem(
            label_bytes=label,
            nonce=nonce,
            ctext=ctext,
            tag=tag,
            w_eaa=w_eaa
        ))
    return episode


# ---------------------------
# Scenario B1 timing (crypto-only)
# ---------------------------
def scenario_b1_emergency_access(
    episode: List[RecordItem],
    E_release: int,
    pw: PairingWrapper,
    sk_eaa: Bn
) -> float:
    subset = episode[:E_release]  # deterministic subset; OK for timing

    t0 = time.perf_counter()
    for it in subset:
        k32 = pw.unwrap_key_for_eaa(it.w_eaa, sk_eaa)
        _pt = aes_gcm_decrypt(k32, it.nonce, it.ctext, it.tag, aad=it.label_bytes)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def median(xs: List[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def run_b1(
    N_total: int = 1000,
    E_values=(1, 10, 50),
    R: int = 30,
    record_size_bytes: int = 16 * 1024,
    out_csv: str = "b1_emergency_access_samples.csv",
) -> None:
    print("\nScenario B1: Emergency (Evidence-Driven) Access — Crypto Only")
    print(f"Episode size N_total: {N_total}")
    print(f"Record plaintext size: {record_size_bytes} bytes (16KB)")
    print(f"Repeats per E: {R}")
    print("Excludes DB/IPFS/network; measures local crypto only.\n")

    pw = PairingWrapper()
    sk_eaa, PK_eaa = pw.keygen_g2()

    # Build episode once (excluded from timing)
    print("Generating encrypted episode + EAA wrappers ...")
    episode = build_episode(N_total, record_size_bytes, pw, PK_eaa)
    print("Episode ready.\n")

    # Warm-up
    _ = scenario_b1_emergency_access(episode, E_release=min(E_values), pw=pw, sk_eaa=sk_eaa)

    rows = []
    for E in E_values:
        samples = []
        for rep in range(R):
            lat_ms = scenario_b1_emergency_access(episode, E_release=E, pw=pw, sk_eaa=sk_eaa)
            samples.append(lat_ms)
            rows.append((N_total, E, rep, lat_ms))

        print(f"E = {E} released records")
        print(f"  Mean latency (ms):            {mean(samples):8.2f}")
        print(f"  Median latency (ms):          {median(samples):8.2f}")
        print(f"  Std dev (ms):                 {stdev(samples):8.2f}")
        print(f"  Amortized per-record (ms):    {mean(samples)/E:8.2f}\n")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N_total", "E_release", "rep", "latency_ms"])
        w.writerows(rows)

    print(f"Raw samples saved to: {out_csv}")


if __name__ == "__main__":
    run_b1(
        N_total=1000,
        E_values=(1, 10, 50),
        R=30,
        record_size_bytes=16 * 1024,
        out_csv="b1_emergency_access_samples.csv",
    )
