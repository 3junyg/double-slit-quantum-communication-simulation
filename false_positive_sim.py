"""False-positive simulation for double-slit interference detection.

The null hypothesis is that no eavesdropper is present.  Each trial draws
N photon positions from the same ideal interference distribution and compares
the empirical histogram with that distribution using total variation distance.
The threshold is chosen from the upper (1-alpha) percentile of null trials.

Run:
    python false_positive_sim.py
    python false_positive_sim.py --photons 1000 --trials 10000 --alpha 0.01
"""

from __future__ import annotations

import argparse
import math
import random
from statistics import mean, pstdev


def interference_probabilities(bins: int, wavelength: float, slit_gap: float,
                               distance: float, phase: float, visibility: float) -> list[float]:
    """Return a normalized position histogram for one encoded state."""
    # Dimensionless screen coordinate: covers two full fringe periods.
    values = []
    for j in range(bins):
        x = -1.0 + 2.0 * (j + 0.5) / bins
        oscillation = 2.0 * math.pi * slit_gap * x / (wavelength * distance) + phase
        values.append(max(0.0, 1.0 + visibility * math.cos(oscillation)))
    total = sum(values)
    return [v / total for v in values]


def sample_histogram(probabilities: list[float], photons: int, rng: random.Random) -> list[int]:
    """Draw photon positions with inverse-CDF sampling (stdlib only)."""
    cumulative = []
    running = 0.0
    for p in probabilities:
        running += p
        cumulative.append(running)
    counts = [0] * len(probabilities)
    for _ in range(photons):
        u = rng.random()
        lo, hi = 0, len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u <= cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        counts[lo] += 1
    return counts


def total_variation(counts: list[int], expected: list[float]) -> float:
    n = sum(counts)
    return 0.5 * sum(abs(c / n - p) for c, p in zip(counts, expected))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photons", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Target false-positive rate, e.g. 0.01 = 1%%")
    parser.add_argument("--wavelength", type=float, default=1.0)
    parser.add_argument("--slit-gap", type=float, default=0.25)
    parser.add_argument("--distance", type=float, default=10.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--visibility", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if not (0 < args.alpha < 1):
        raise SystemExit("--alpha must be between 0 and 1")
    if min(args.photons, args.bins, args.trials) <= 0:
        raise SystemExit("photons, bins, and trials must be positive")

    expected = interference_probabilities(
        args.bins, args.wavelength, args.slit_gap, args.distance,
        args.phase, args.visibility)
    rng = random.Random(args.seed)
    distances = []
    for _ in range(args.trials):
        counts = sample_histogram(expected, args.photons, rng)
        distances.append(total_variation(counts, expected))

    threshold = percentile(distances, 1.0 - args.alpha)
    false_alarms = sum(d > threshold for d in distances)
    print("Null (no-eavesdropper) false-positive simulation")
    print(f"photons per block : {args.photons}")
    print(f"position bins     : {args.bins}")
    print(f"null trials       : {args.trials}")
    print(f"target alpha      : {args.alpha:.4f}")
    print(f"TV mean ± sd      : {mean(distances):.6f} ± {pstdev(distances):.6f}")
    print(f"threshold         : {threshold:.6f}")
    print(f"measured FP rate  : {false_alarms / args.trials:.6f} ({false_alarms}/{args.trials})")
    print("Decision rule: declare a possible eavesdropper when TV distance > threshold.")


if __name__ == "__main__":
    main()