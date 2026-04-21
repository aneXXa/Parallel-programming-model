import csv
import sys
from pathlib import Path


def read_csv(path: Path):
    threads = []
    speedups = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            threads.append(int(row["threads"]))
            speedups.append(float(row["speedup"]))
    return threads, speedups


def plot_combined(csv_paths):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate PDF plots.\n"
            f"Import error: {e}\n\n"
            "Fallback: open the CSV in Excel/LibreOffice, build the plot, and export as PDF."
        )

    all_threads = set()
    series = []
    for csv_path in csv_paths:
        threads, speedups = read_csv(csv_path)
        size = csv_path.stem.replace("speedup_DGEMV_", "")
        series.append((size, threads, speedups))
        all_threads.update(threads)

    if not series:
        raise RuntimeError("No CSV files provided.")

    sorted_threads = sorted(all_threads)
    out_pdf = csv_paths[0].parent / "speedup_DGEMV_combined.pdf"

    plt.figure(figsize=(9, 6))
    for size, threads, speedups in series:
        plt.plot(threads, speedups, marker="o", linewidth=2, label=f"{size}x{size}")
        if threads and speedups:
            x_last = threads[-1]
            y_last = speedups[-1]
            plt.annotate(
                f"{y_last:.1f}x",
                xy=(x_last, y_last),
                xytext=(8, -10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "fc": "#fff59d", "ec": "#9e9e9e", "alpha": 0.95},
            )
    plt.plot(sorted_threads, sorted_threads, linestyle="--", color="gray", label="Ideal speedup x=y")
    plt.title("Matrix-vector multiplication: Speedup graph")
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(sorted_threads)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    return out_pdf


def main(argv):
    if len(argv) < 3:
        print("Usage: python plot_speedup.py speedup_DGEMV_20000.csv speedup_DGEMV_40000.csv")
        return 2

    csv_paths = [Path(p) for p in argv[1:]]
    try:
        out = plot_combined(csv_paths)
        print(f"Wrote {out}")
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

