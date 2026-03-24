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
    size = csv_path.stem.replace("speedup_integral_", "")
    out_pdf = csv_path.with_suffix(".pdf")
    e_par = [s / t for s, t in zip(s_par, threads)]
    e_at = [s / t for s, t in zip(s_at, threads)]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12), constrained_layout=True)

    axes[0].plot(threads, s_par, marker="o", label="integrate_omp (local+atomic)")
    axes[0].plot(threads, s_at, marker="s", label="integrate_omp_atomic")
    # Keep measured curves visible even when ideal speedup has a very different scale.
    max_measured = max(max(s_par), max(s_at))
    axes[0].set_ylim(0.0, max_measured * 1.15 if max_measured > 0 else 1.0)

    ax_ideal = axes[0].twinx()
    ax_ideal.plot(threads, threads, linestyle="--", color="gray", label="Ideal speedup x=y")
    ax_ideal.set_ylabel("Ideal speedup scale")
    ax_ideal.set_ylim(0.0, max(threads) * 1.15 if threads else 1.0)
    axes[0].set_title(f"Speedup vs threads (integral, nsteps={size})")
    axes[0].set_xlabel("Number of threads")
    axes[0].set_ylabel("Speedup S_n = T_serial / T_n")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_xticks(threads)
    handles_main, labels_main = axes[0].get_legend_handles_labels()
    handles_ideal, labels_ideal = ax_ideal.get_legend_handles_labels()
    axes[0].legend(handles_main + handles_ideal, labels_main + labels_ideal, loc="upper left")

    axes[1].plot(threads, e_par, marker="o", label="integrate_omp (local+atomic)")
    axes[1].plot(threads, e_at, marker="s", label="integrate_omp_atomic")
    axes[1].set_title(f"Efficiency vs threads (integral, nsteps={size})")
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
        print("Usage: python plot_speedup_integral.py speedup_integral_40000000.csv")
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

