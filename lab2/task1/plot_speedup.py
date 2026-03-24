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


def plot_one(csv_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate PDF plots.\n"
            f"Import error: {e}\n\n"
            "Fallback: open the CSV in Excel/LibreOffice, build the plot, and export as PDF."
        )

    threads, speedups = read_csv(csv_path)
    size = csv_path.stem.replace("speedup_DGEMV_", "")
    out_pdf = csv_path.with_suffix(".pdf")
    efficiencies = [s / t for s, t in zip(speedups, threads)]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12), constrained_layout=True)

    axes[0].plot(threads, speedups, marker="o", label="Measured speedup")
    axes[0].plot(threads, threads, linestyle="--", color="gray", label="Ideal speedup x=y")
    axes[0].set_title(f"Speedup vs threads (M=N={size})")
    axes[0].set_xlabel("Number of threads")
    axes[0].set_ylabel("Speedup S_n = T_serial / T_n")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_xticks(threads)
    axes[0].legend()

    axes[1].plot(threads, efficiencies, marker="o", label="Measured efficiency")
    axes[1].set_title(f"Efficiency vs threads (M=N={size})")
    axes[1].set_xlabel("Number of threads")
    axes[1].set_ylabel("Efficiency E_n = S_n / n")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_xticks(threads)
    axes[1].legend()

    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


def main(argv):
    if len(argv) < 2:
        print("Usage: python plot_speedup.py speedup_DGEMV_20000.csv [speedup_DGEMV_40000.csv ...]")
        return 2

    rc = 0
    for p in argv[1:]:
        csv_path = Path(p)
        try:
            out = plot_one(csv_path)
            print(f"Wrote {out}")
        except Exception as e:
            print(str(e), file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

