import csv
import sys
from pathlib import Path


def read_csv(path: Path):
    threads = []
    t_serial = []
    t_blocks = []
    s_blocks = []
    e_blocks = []
    t_whole = []
    s_whole = []
    e_whole = []

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            threads.append(int(row["threads"]))
            t_serial.append(float(row["T_serial"]))
            t_blocks.append(float(row["T_blocks"]))
            s_blocks.append(float(row["S_blocks"]))
            e_blocks.append(float(row["E_blocks"]))
            t_whole.append(float(row["T_whole"]))
            s_whole.append(float(row["S_whole"]))
            e_whole.append(float(row["E_whole"]))
    return threads, t_serial, t_blocks, s_blocks, e_blocks, t_whole, s_whole, e_whole


def draw_single(ax, title: str, xlabel: str, ylabel: str, x, y, label: str, add_diag: bool = False):
    ax.plot(x, y, marker="o", label=label)
    if add_diag:
        ax.plot(x, x, linestyle="--", color="gray", label="Ideal speedup x=y")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.legend()


def draw_compare(
    ax, title: str, xlabel: str, ylabel: str, x, y1, y2, label1: str, label2: str, add_diag: bool = False
):
    ax.plot(x, y1, marker="o", label=label1)
    ax.plot(x, y2, marker="s", label=label2)
    if add_diag:
        ax.plot(x, x, linestyle="--", color="gray", label="Ideal speedup x=y")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.legend()


def plot_task3(csv_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate PDF plots.\n"
            f"Import error: {e}\n\n"
            "Fallback: open CSV in Excel/LibreOffice, build charts, and export to PDF."
        )

    threads, t_serial, t_blocks, s_blocks, e_blocks, t_whole, s_whole, e_whole = read_csv(csv_path)
    suffix = csv_path.stem.replace("speedup_jacobi_", "")
    out_dir = csv_path.parent
    out_pdf = out_dir / f"task3_all_graphs_{suffix}.pdf"

    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 34), constrained_layout=True)

    draw_single(
        axes[0],
        f"Task3 Variant1 (parallel-for each loop): Time vs Threads (N={suffix})",
        "Number of threads",
        "Execution time, sec",
        threads,
        t_blocks,
        "T_blocks",
    )
    draw_single(
        axes[1],
        f"Task3 Variant1 (parallel-for each loop): Speedup vs Threads (N={suffix})",
        "Number of threads",
        "Speedup S_n = T_serial / T_n",
        threads,
        s_blocks,
        "S_blocks",
        add_diag=True,
    )
    draw_single(
        axes[2],
        f"Task3 Variant1 (parallel-for each loop): Efficiency vs Threads (N={suffix})",
        "Number of threads",
        "Efficiency E_n = S_n / n",
        threads,
        e_blocks,
        "E_blocks",
    )

    draw_single(
        axes[3],
        f"Task3 Variant2 (single parallel region): Time vs Threads (N={suffix})",
        "Number of threads",
        "Execution time, sec",
        threads,
        t_whole,
        "T_whole",
    )
    draw_single(
        axes[4],
        f"Task3 Variant2 (single parallel region): Speedup vs Threads (N={suffix})",
        "Number of threads",
        "Speedup S_n = T_serial / T_n",
        threads,
        s_whole,
        "S_whole",
        add_diag=True,
    )
    draw_single(
        axes[5],
        f"Task3 Variant2 (single parallel region): Efficiency vs Threads (N={suffix})",
        "Number of threads",
        "Efficiency E_n = S_n / n",
        threads,
        e_whole,
        "E_whole",
    )

    draw_compare(
        axes[6],
        f"Task3 Comparison: Speedup vs Threads (N={suffix})",
        "Number of threads",
        "Speedup S_n = T_serial / T_n",
        threads,
        s_blocks,
        s_whole,
        "Variant1: parallel-for each loop",
        "Variant2: single parallel region",
        add_diag=True,
    )

    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


def main(argv):
    if len(argv) < 2:
        print("Usage: python plot_task3_metrics.py speedup_jacobi_<N>.csv")
        return 2

    rc = 0
    for p in argv[1:]:
        csv_path = Path(p)
        try:
            out = plot_task3(csv_path)
            print(f"Wrote {out}")
        except Exception as e:
            print(str(e), file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

