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

    plt.figure(figsize=(8, 6))
    plt.plot(threads, speedups, marker="o")
    plt.title(f"Speedup vs threads (M=N={size})")
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup S_n = T_serial / T_n")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(threads)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
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

