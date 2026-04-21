import csv
import sys
from pathlib import Path


def read_rows(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "threads": int(row["threads"]),
                    "speedup": float(row["speedup"]),
                    "efficiency": float(row["efficiency"]),
                }
            )
    return rows


def plot(csv_paths):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib import failed: {e}")

    series = []
    all_threads = set()
    for p in csv_paths:
        rows = read_rows(p)
        size = p.stem.split("_")[-1]
        threads = [r["threads"] for r in rows]
        speedup = [r["speedup"] for r in rows]
        efficiency = [r["efficiency"] for r in rows]
        series.append((size, threads, speedup, efficiency))
        all_threads.update(threads)

    if not series:
        raise RuntimeError("No CSV files provided.")

    sorted_threads = sorted(all_threads)
    out_speedup = csv_paths[0].parent / "speedup_task1_stdthread.pdf"
    out_eff = csv_paths[0].parent / "efficiency_task1_stdthread.pdf"

    plt.figure(figsize=(9, 6))
    for size, threads, speedup, _eff in series:
        plt.plot(threads, speedup, marker="o", linewidth=2, label=f"N={size}")
    plt.plot(sorted_threads, sorted_threads, linestyle="--", color="gray", label="Ideal x=y")
    plt.title("DGEMV std::thread: Speedup")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(sorted_threads)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_speedup)
    plt.close()

    plt.figure(figsize=(9, 6))
    for size, threads, _speedup, efficiency in series:
        plt.plot(threads, efficiency, marker="s", linewidth=2, label=f"N={size}")
    plt.title("DGEMV std::thread: Efficiency")
    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(sorted_threads)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_eff)
    plt.close()

    return out_speedup, out_eff


def main(argv):
    if len(argv) < 3:
        print("Usage: python plot_speedup_task1.py speedup_dgemv_stdthread_20000.csv speedup_dgemv_stdthread_40000.csv")
        return 2
    csv_paths = [Path(p) for p in argv[1:]]
    out_speedup, out_eff = plot(csv_paths)
    print(f"Wrote {out_speedup}")
    print(f"Wrote {out_eff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
