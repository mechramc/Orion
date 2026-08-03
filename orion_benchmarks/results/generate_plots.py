import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_results(results_dir=None):
    # If this script lives inside results/, use that folder by default
    results_path = Path(results_dir) if results_dir else Path(__file__).resolve().parent

    data = []
    files = sorted(results_path.glob("*.json"))
    print(f"Found {len(files)} json files in {results_path}")

    for filepath in files:
        fname = filepath.name

        if fname.startswith("final_summary") or "_error_" in fname:
            print(f"Skipping {fname}")
            continue

        # Expect: burst_llama-3.2-3b_mlx_AC_20260803_175254.json
        stem = filepath.stem
        try:
            mode, rest = stem.split("_", 1)
            model, backend, power, date_str, time_str = rest.rsplit("_", 4)
        except ValueError:
            print(f"Skipping unexpected filename format: {fname}")
            continue

        try:
            with open(filepath, "r") as f:
                content = json.load(f)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue

        stats = content.get("statistics", {})

        data.append({
            "Mode": mode.capitalize(),
            "Model": model,
            "Backend": backend.upper() if backend == "mlx" else "Orion (ANE)",
            "Power": power,
            "Throughput (tok/s)": stats.get("mean_tok_s", 0),
            "Latency p50 (ms)": stats.get("mean_p50_ms", 0),
            "Energy (J/run)": stats.get("mean_energy_j", 0),
            "Thermal": stats.get("thermal_pressure_mode", "Unknown"),
        })

    df = pd.DataFrame(data)
    return df

def plot_thermal_degradation(df, output_file="thermal_degradation.png"):
    df_ac = df[df["Power"] == "AC"].copy()
    if df_ac.empty:
        print("No AC rows found; skipping thermal plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = df_ac["Model"].unique()
    width = 0.35
    x = np.arange(len(models))

    burst_vals = []
    sustained_vals = []

    for m in models:
        burst_row = df_ac[(df_ac["Model"] == m) & (df_ac["Mode"] == "Burst")]
        sustained_row = df_ac[(df_ac["Model"] == m) & (df_ac["Mode"] == "Sustained")]

        if burst_row.empty or sustained_row.empty:
            print(f"Missing burst/sustained row for {m}; skipping.")
            continue

        burst_vals.append(burst_row["Throughput (tok/s)"].values[0])
        sustained_vals.append(sustained_row["Throughput (tok/s)"].values[0])

    if not burst_vals:
        print("No comparable burst/sustained data found; skipping thermal plot.")
        return

    ax.bar(x[:len(burst_vals)] - width/2, burst_vals, width, label="Burst", edgecolor="black")
    ax.bar(x[:len(sustained_vals)] + width/2, sustained_vals, width, label="Sustained (5 min)", edgecolor="black")

    ax.set_ylabel("Throughput (tokens/sec)", fontsize=12)
    ax.set_title("Apple M1 Air (Fanless): Burst vs Sustained Throughput (AC Power)", fontsize=14)
    ax.set_xticks(x[:len(burst_vals)])
    ax.set_xticklabels(["Llama-3.2-3B\n(MLX)", "Phi-3-Mini\n(MLX)", "GPT-2-124M\n(Orion/ANE)"][:len(burst_vals)])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for i in range(len(burst_vals)):
        deg = (burst_vals[i] - sustained_vals[i]) / burst_vals[i] * 100
        color = "red" if deg > 5 else "green"
        ax.text(x[i] + width/2, sustained_vals[i] + 2, f"-{deg:.1f}%",
                ha="center", color=color, fontweight="bold", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")

def plot_power_efficiency(df, output_file="power_efficiency.png"):
    df = df[df["Energy (J/run)"] > 0].copy()
    if df.empty:
        print("No rows with energy > 0; skipping power-efficiency plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"AC": "C0", "Battery": "C1"}
    markers = {"Burst": "o", "Sustained": "X"}

    for power in df["Power"].unique():
        for mode in df["Mode"].unique():
            subset = df[(df["Power"] == power) & (df["Mode"] == mode)]
            if subset.empty:
                continue

            ax.scatter(
                subset["Throughput (tok/s)"],
                subset["Energy (J/run)"],
                c=colors.get(power, "C2"),
                marker=markers.get(mode, "o"),
                s=150,
                label=f"{power} - {mode}",
                edgecolors="black",
            )

            for _, row in subset.iterrows():
                ax.annotate(
                    f"{row['Model']}\n{row['Backend']}",
                    (row["Throughput (tok/s)"], row["Energy (J/run)"]),
                    textcoords="offset points",
                    xytext=(10, 5),
                    fontsize=8,
                )

    ax.set_xlabel("Throughput (tokens/sec)", fontsize=12)
    ax.set_ylabel("Energy per Run (Joules)", fontsize=12)
    ax.set_title("Performance vs. Energy Tradeoff on M1 Air", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    # If you're running this from inside results/, this is correct:
    df = load_results(".")

    print("Loaded data:")
    if df.empty:
        print("No valid rows found.")
    else:
        print(df[["Mode", "Model", "Backend", "Power", "Throughput (tok/s)", "Thermal"]])

        plot_thermal_degradation(df)
        plot_power_efficiency(df)