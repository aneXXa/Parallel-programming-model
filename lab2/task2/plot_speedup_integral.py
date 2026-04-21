import csv
import sys
from pathlib import Path


def read_csv(path: Path):
    threads = []
    s_par = []
    s_at = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            threads.append(int(row["threads"]))
            s_par.append(float(row["S_par"]))
            s_at.append(float(row["S_at"]))
    return threads, s_par, s_at


def plot_one(csv_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate PDF plots.\n"
            f"Import error: {e}\n\n"
            "Fallback: open the CSV in Excel/LibreOffice, build the plot, and export as PDF."
        )

    threads, s_par, s_at = read_csv(csv_path)
    out_speedup_pdf = csv_path.with_suffix(".pdf")
    out_eff_pdf = csv_path.with_name(f"{csv_path.stem}_efficiency.pdf")
    e_par = [s / t for s, t in zip(s_par, threads)]
    e_at = [s / t for s, t in zip(s_at, threads)]

    plt.figure(figsize=(8, 6))
    plt.plot(threads, s_par, marker="o", linewidth=2, label="integrate_omp (local+atomic)")
    plt.plot(threads, s_at, marker="s", linewidth=2, label="integrate_omp_atomic")
    plt.plot(threads, threads, linestyle="--", color="gray", label="Ideal speedup x=y")
    plt.title("Numerical integration: Speedup graph")
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(threads)
    plt.legend()
    if threads:
        plt.annotate(
            f"{s_par[-1]:.1f}x",
            xy=(threads[-1], s_par[-1]),
            xytext=(8, -10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": "#fff59d", "ec": "#9e9e9e", "alpha": 0.95},
        )
        plt.annotate(
            f"{s_at[-1]:.1f}x",
            xy=(threads[-1], s_at[-1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": "#fff59d", "ec": "#9e9e9e", "alpha": 0.95},
        )
    plt.tight_layout()
    plt.savefig(out_speedup_pdf)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(threads, e_par, marker="o", linewidth=2, label="integrate_omp (local+atomic)")
    plt.plot(threads, e_at, marker="s", linewidth=2, label="integrate_omp_atomic")
    plt.title("Numerical integration: Efficiency graph")
    plt.xlabel("Number of threads")
    plt.ylabel("Efficiency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(threads)
    plt.legend()
    plt.text(
        0.98,
        0.78,
        r"Efficiency formula: $E_n = \frac{S_n}{n}$",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#9e9e9e", "alpha": 0.95},
    )
    plt.tight_layout()
    plt.savefig(out_eff_pdf)
    plt.close()

    return out_speedup_pdf, out_eff_pdf


def main(argv):
    if len(argv) < 2:
        print("Usage: python plot_speedup_integral.py speedup_integral_40000000.csv")
        return 2

    rc = 0
    for p in argv[1:]:
        csv_path = Path(p)
        try:
            out_speedup, out_eff = plot_one(csv_path)
            print(f"Wrote {out_speedup}")
            print(f"Wrote {out_eff}")
        except Exception as e:
            print(str(e), file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

