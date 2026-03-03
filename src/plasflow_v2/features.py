from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
import math

VALID = {"A", "C", "G", "T"}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
SCALAR_FEATURES = ["length", "log_length", "gc", "entropy4", "n_frac"]


def safe_gc_content(sequence: str) -> float:
    if not sequence:
        return 0.0
    gc = sum(1 for ch in sequence if ch in {"G", "C"})
    return gc / max(len(sequence), 1)


def iter_kmers(sequence: str, k: int):
    seq = sequence.upper()
    for i in range(0, max(len(seq) - k + 1, 0)):
        kmer = seq[i : i + k]
        if all(ch in VALID for ch in kmer):
            yield kmer


def reverse_complement(sequence: str) -> str:
    seq = sequence.upper()
    return "".join(COMPLEMENT.get(ch, "N") for ch in reversed(seq))


def canonical_kmer(kmer: str) -> str:
    upper = kmer.upper()
    rc = reverse_complement(upper)
    return upper if upper <= rc else rc


@lru_cache(maxsize=16)
def canonical_kmer_order(k: int) -> tuple[str, ...]:
    kmers = set()
    for parts in product("ACGT", repeat=k):
        kmer = "".join(parts)
        kmers.add(canonical_kmer(kmer))
    return tuple(sorted(kmers))


def kmer_counts(sequence: str, k: int, canonical: bool = False) -> Counter[str]:
    counts: Counter[str] = Counter()
    if canonical:
        counts.update(canonical_kmer(kmer) for kmer in iter_kmers(sequence, k))
    else:
        counts.update(iter_kmers(sequence, k))
    return counts


def kmer_entropy(sequence: str, k: int = 4) -> float:
    counts = kmer_counts(sequence, k)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
    max_entropy = math.log(4**k, 2)
    return entropy / max_entropy if max_entropy else 0.0


def n_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    return sum(1 for ch in sequence.upper() if ch == "N") / len(sequence)


def sequence_features(sequence: str) -> dict[str, float]:
    length = len(sequence)
    return {
        "length": float(length),
        "log_length": math.log10(max(length, 1)),
        "gc": safe_gc_content(sequence),
        "entropy4": kmer_entropy(sequence, 4),
        "n_frac": n_fraction(sequence),
    }


def build_feature_manifest(
    k_values: tuple[int, ...] = (4, 5, 6),
    include_scalar: bool = True,
    canonical: bool = True,
) -> dict[str, object]:
    if not canonical:
        raise ValueError("Only canonical k-mer features are supported in v2 manifests")

    feature_order: list[str] = []
    if include_scalar:
        feature_order.extend(SCALAR_FEATURES)

    for k in k_values:
        for kmer in canonical_kmer_order(k):
            feature_order.append(f"k{k}:{kmer}")

    return {
        "version": "v2-kmer-canonical-1",
        "canonical": True,
        "k_values": list(k_values),
        "scalar_features": SCALAR_FEATURES[:],
        "feature_order": feature_order,
    }


@lru_cache(maxsize=2)
def default_feature_manifest() -> dict[str, object]:
    return build_feature_manifest()


def _kmer_frequency_feature_map(sequence: str, k_values: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in k_values:
        counts = kmer_counts(sequence, k, canonical=True)
        total = sum(counts.values())
        order = canonical_kmer_order(k)
        if total == 0:
            for kmer in order:
                out[f"k{k}:{kmer}"] = 0.0
            continue
        denom = float(total)
        for kmer in order:
            out[f"k{k}:{kmer}"] = float(counts.get(kmer, 0)) / denom
    return out


def sequence_feature_map(sequence: str, feature_manifest: dict[str, object] | None = None) -> dict[str, float]:
    manifest = feature_manifest or default_feature_manifest()
    k_values = [int(k) for k in manifest.get("k_values", [4, 5, 6])]

    out = sequence_features(sequence)
    out.update(_kmer_frequency_feature_map(sequence, k_values))
    return out


def vectorize_sequence(sequence: str, feature_manifest: dict[str, object] | None = None) -> list[float]:
    manifest = feature_manifest or default_feature_manifest()
    feature_order = [str(name) for name in manifest.get("feature_order", [])]
    if not feature_order:
        raise ValueError("feature_manifest.feature_order is empty")

    fmap = sequence_feature_map(sequence, manifest)
    return [float(fmap.get(name, 0.0)) for name in feature_order]


def vectorize_sequences(sequences: list[str], feature_manifest: dict[str, object] | None = None) -> list[list[float]]:
    return [vectorize_sequence(seq, feature_manifest) for seq in sequences]
