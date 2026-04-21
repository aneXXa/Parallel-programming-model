import csv
import sys
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    return float(v)


def to_int(v):
    return int(v)


def plot_variants(csv_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib import failed: {e}")

    rows = read_rows(csv_path)
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[to_int(row["variant"])].append(row)

    for variant in by_variant:
        by_variant[variant].sort(key=lambda r: to_int(r["threads"]))

    plt.figure(figsize=(8, 6))
    for variant in sorted(by_variant):
        th = [to_int(r["threads"]) for r in by_variant[variant]]
        tm = [to_float(r["time_sec"]) for r in by_variant[variant]]
        plt.plot(th, tm, marker="o", linewidth=2, label=f"Variant {variant}")
    plt.title("Richardson OpenMP: execution time")
    plt.xlabel("Threads")
    plt.ylabel("Time, s")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_time = csv_path.with_name(csv_path.stem + "_time.pdf")
    plt.savefig(out_time)
    plt.close()

    plt.figure(figsize=(8, 6))
    for variant in sorted(by_variant):
        th = [to_int(r["threads"]) for r in by_variant[variant]]
        sp = [to_float(r["speedup"]) for r in by_variant[variant]]
        plt.plot(th, sp, marker="o", linewidth=2, label=f"Variant {variant}")
        plt.plot(th, th, linestyle="--", color="gray", alpha=0.35)
    plt.title("Richardson OpenMP: speedup")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_speedup = csv_path.with_name(csv_path.stem + "_speedup.pdf")
    plt.savefig(out_speedup)
    plt.close()

    plt.figure(figsize=(8, 6))
    for variant in sorted(by_variant):
        th = [to_int(r["threads"]) for r in by_variant[variant]]
        ef = [to_float(r["efficiency"]) for r in by_variant[variant]]
        plt.plot(th, ef, marker="o", linewidth=2, label=f"Variant {variant}")
    plt.title("Richardson OpenMP: efficiency")
    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_eff = csv_path.with_name(csv_path.stem + "_efficiency.pdf")
    plt.savefig(out_eff)
    plt.close()

    print(f"Wrote {out_time}")
    print(f"Wrote {out_speedup}")
    print(f"Wrote {out_eff}")


def plot_schedule(csv_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib import failed: {e}")

    rows = read_rows(csv_path)
    labels = [f"{r['schedule']}:{r['chunk']}" for r in rows]
    times = [to_float(r["time_sec"]) for r in rows]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, times)
    plt.title("Richardson OpenMP: schedule comparison")
    plt.xlabel("schedule:chunk")
    plt.ylabel("Time, s")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_pdf = csv_path.with_name(csv_path.stem + "_schedule.pdf")
    plt.savefig(out_pdf)
    plt.close()
    print(f"Wrote {out_pdf}")


def main(argv):
    if len(argv) != 3:
        print("Usage: python plot_task3_metrics.py <variants|schedule> <csv_path>")
        return 2
    mode = argv[1]
    path = Path(argv[2])
    if mode == "variants":
        plot_variants(path)
        return 0
    if mode == "schedule":
        plot_schedule(path)
        return 0
    print(f"Unknown mode: {mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
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


def read_schedule_csv(path: Path):
    chunks = []
    t_blocks = []
    s_blocks = []
    t_whole = []
    s_whole = []
    schedule = None
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            schedule = row["schedule"]
            chunks.append(int(row["chunk"]))
            t_blocks.append(float(row["T_blocks"]))
            s_blocks.append(float(row["S_blocks"]))
            t_whole.append(float(row["T_whole"]))
            s_whole.append(float(row["S_whole"]))
    return schedule, chunks, t_blocks, s_blocks, t_whole, s_whole


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
    ax.plot(x, y1, marker="o", linewidth=2.2, markersize=7, label=label1, zorder=3)
    ax.plot(x, y2, marker="s", linewidth=2.2, markersize=7, label=label2, zorder=3)
    if add_diag:
        ax.plot(
            x,
            x,
            linestyle="--",
            color="gray",
            alpha=0.65,
            linewidth=1.6,
            label="Ideal speedup x=y",
            zorder=1,
        )
    y_min = min(min(y1), min(y2))
    y_max = max(max(y1), max(y2))
    pad = max(0.05 * (y_max - y_min), 0.05)
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xticks(x)
    ax.legend(loc="upper left", framealpha=0.95)


def plot_schedule_comparison(static_csv: Path, guided_csv: Path, dynamic_csv: Path, mode: str = "blocks"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate PDF plots.\n"
            f"Import error: {e}\n\n"
            "Fallback: open CSV in Excel/LibreOffice, build charts, and export to PDF."
        )

    mode_field = "S_blocks" if mode == "blocks" else "S_whole"
    time_field = "T_blocks" if mode == "blocks" else "T_whole"
    mode_label = "parallel-for each loop" if mode == "blocks" else "single parallel region"

    series = []
    all_chunks = set()
    for csv_path in [static_csv, guided_csv, dynamic_csv]:
        schedule, chunks, t_blocks, s_blocks, t_whole, s_whole = read_schedule_csv(csv_path)
        speedup_y = s_blocks if mode == "blocks" else s_whole
        time_y = t_blocks if mode == "blocks" else t_whole
        series.append((schedule, chunks, speedup_y, time_y))
        all_chunks.update(chunks)

    sorted_chunks = sorted(all_chunks)
    n_suffix = static_csv.stem.replace("schedule_jacobi_static_n", "").replace("_t20", "")
    out_speedup_pdf = static_csv.parent / f"schedule_jacobi_compare_{mode_field}_{n_suffix}.pdf"
    out_speedup_metrics_pdf = static_csv.parent / f"schedule_jacobi_speedup_vs_chunk_metrics_{n_suffix}.pdf"
    out_mean_time_pdf = static_csv.parent / f"schedule_jacobi_mean_{time_field}_{n_suffix}.pdf"
    out_chunk_time_pdf = static_csv.parent / f"schedule_jacobi_chunk_{time_field}_{n_suffix}.pdf"
    out_v1_v2_pdf = static_csv.parent / f"schedule_jacobi_v1_vs_v2_{n_suffix}.pdf"

    # 1) Speedup comparison: static/guided/dynamic on one graph.
    plt.figure(figsize=(9, 6))
    for schedule, chunks, speedups, _times in series:
        plt.plot(chunks, speedups, marker="o", linewidth=2, label=schedule)
        if chunks and speedups:
            x_last = chunks[-1]
            y_last = speedups[-1]
            plt.annotate(
                f"{y_last:.2f}x",
                xy=(x_last, y_last),
                xytext=(8, -10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "fc": "#fff59d", "ec": "#9e9e9e", "alpha": 0.95},
            )

    plt.title(f"Jacobi speedup by schedule ({mode_label})")
    plt.xlabel("Chunk size")
    plt.ylabel(f"Speedup ({mode_field})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(sorted_chunks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_speedup_pdf)
    plt.close()

    # 1b) Speedup vs chunk for both metrics (V1 and V2) on separate subplots.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 11), constrained_layout=True)
    for csv_path in [static_csv, guided_csv, dynamic_csv]:
        schedule, chunks, _t_blocks, s_blocks, _t_whole, s_whole = read_schedule_csv(csv_path)
        axes[0].plot(chunks, s_blocks, marker="o", linewidth=2, label=schedule)
        axes[1].plot(chunks, s_whole, marker="o", linewidth=2, label=schedule)
    axes[0].set_title("Speedup vs chunk: Variant1 (parallel-for each loop)")
    axes[0].set_xlabel("Chunk size")
    axes[0].set_ylabel("Speedup (S_blocks)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_xticks(sorted_chunks)
    axes[0].legend()
    axes[1].set_title("Speedup vs chunk: Variant2 (single parallel region)")
    axes[1].set_xlabel("Chunk size")
    axes[1].set_ylabel("Speedup (S_whole)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_xticks(sorted_chunks)
    axes[1].legend()
    fig.savefig(out_speedup_metrics_pdf)
    plt.close(fig)

    # 2) Mean execution time bar chart for each method.
    schedules = [schedule for schedule, _chunks, _speedups, _times in series]
    mean_times = []
    for _schedule, _chunks, _speedups, times in series:
        mean_times.append(sum(times) / len(times) if times else 0.0)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(schedules, mean_times, color=["#4e79a7", "#f28e2b", "#59a14f"])
    for bar, value in zip(bars, mean_times):
        plt.annotate(
            f"{value:.3f}s",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": "#fff59d", "ec": "#9e9e9e", "alpha": 0.95},
        )
    plt.title(f"Mean execution time by schedule ({mode_label})")
    plt.xlabel("Scheduling method")
    plt.ylabel(f"Mean time ({time_field}), sec")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_mean_time_pdf)
    plt.close()

    # 3) Chunk dependence bar chart: grouped bars by chunk for all methods.
    plt.figure(figsize=(10, 6))
    width = 0.25
    x_pos = list(range(len(sorted_chunks)))
    schedule_to_times = {schedule: times for schedule, _chunks, _speedups, times in series}
    plot_order = ["static", "guided", "dynamic"]
    for idx, schedule in enumerate(plot_order):
        times = schedule_to_times.get(schedule, [])
        bar_positions = [x + (idx - 1) * width for x in x_pos]
        plt.bar(bar_positions, times, width=width, label=schedule)
    plt.title(f"Chunk dependence of execution time ({mode_label})")
    plt.xlabel("Chunk size")
    plt.ylabel(f"Execution time ({time_field}), sec")
    plt.xticks(x_pos, sorted_chunks)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_chunk_time_pdf)
    plt.close()

    # 4) Final standalone V1 vs V2 graph (mean speedup over methods by chunk).
    mean_s_blocks = []
    mean_s_whole = []
    for chunk_idx in range(len(sorted_chunks)):
        vals_blocks = []
        vals_whole = []
        for csv_path in [static_csv, guided_csv, dynamic_csv]:
            _schedule, chunks, _t_blocks, s_blocks, _t_whole, s_whole = read_schedule_csv(csv_path)
            if chunk_idx < len(chunks):
                vals_blocks.append(s_blocks[chunk_idx])
                vals_whole.append(s_whole[chunk_idx])
        mean_s_blocks.append(sum(vals_blocks) / len(vals_blocks) if vals_blocks else 0.0)
        mean_s_whole.append(sum(vals_whole) / len(vals_whole) if vals_whole else 0.0)

    plt.figure(figsize=(9, 6))
    plt.plot(
        sorted_chunks,
        mean_s_blocks,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Variant1: parallel-for each loop",
        zorder=3,
    )
    plt.plot(
        sorted_chunks,
        mean_s_whole,
        marker="s",
        linewidth=2.2,
        markersize=7,
        label="Variant2: single parallel region",
        zorder=3,
    )
    plt.title("Final comparison: V1 vs V2 speedup by chunk")
    plt.xlabel("Chunk size")
    plt.ylabel("Mean speedup over methods")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xticks(sorted_chunks)
    plt.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_v1_v2_pdf)
    plt.close()

    return out_speedup_pdf, out_speedup_metrics_pdf, out_mean_time_pdf, out_chunk_time_pdf, out_v1_v2_pdf


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
        print("Usage:")
        print("  python plot_task3_metrics.py speedup_jacobi_<N>.csv")
        print(
            "  python plot_task3_metrics.py "
            "schedule_jacobi_static_n<N>_t20.csv "
            "schedule_jacobi_guided_n<N>_t20.csv "
            "schedule_jacobi_dynamic_n<N>_t20.csv [blocks|whole]"
        )
        return 2

    if len(argv) >= 4 and "schedule_jacobi_" in Path(argv[1]).name:
        static_csv = Path(argv[1])
        guided_csv = Path(argv[2])
        dynamic_csv = Path(argv[3])
        mode = argv[4].lower() if len(argv) >= 5 else "blocks"
        if mode not in {"blocks", "whole"}:
            print("Mode must be 'blocks' or 'whole'.", file=sys.stderr)
            return 2
        try:
            out_speedup, out_speedup_metrics, out_mean_time, out_chunk_time, out_v1_v2 = plot_schedule_comparison(
                static_csv, guided_csv, dynamic_csv, mode=mode
            )
            print(f"Wrote {out_speedup}")
            print(f"Wrote {out_speedup_metrics}")
            print(f"Wrote {out_mean_time}")
            print(f"Wrote {out_chunk_time}")
            print(f"Wrote {out_v1_v2}")
            return 0
        except Exception as e:
            print(str(e), file=sys.stderr)
            return 1

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

