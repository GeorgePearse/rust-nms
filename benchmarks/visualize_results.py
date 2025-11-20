#!/usr/bin/env python3
"""
Visualize benchmark results from divan.

Usage:
    cargo bench | python visualize_results.py

    # or with saved results
    cargo bench > results.txt
    python visualize_results.py results.txt
"""

import sys
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_divan_results(lines):
    """Parse benchmark results from divan output."""
    results = {}
    current_benchmark = None

    for line in lines:
        line = line.strip()

        # Check if this is a benchmark name
        if line.startswith("bench_"):
            current_benchmark = line.split()[0]
            results[current_benchmark] = {}

        # Check for benchmark results
        elif line and current_benchmark is not None and ":" in line:
            if "time:" in line.lower():
                # Extract the parameter and time
                param_match = re.search(r"\[(.*?)\]", line)
                time_match = re.search(r"time: (\d+\.?\d*) ([µn]s|ms|s)", line)

                if param_match and time_match:
                    param = param_match.group(1)
                    time_val = float(time_match.group(1))
                    time_unit = time_match.group(2)

                    # Convert time to consistent unit (milliseconds)
                    if time_unit == "ns":
                        time_val /= 1_000_000
                    elif time_unit == "µs":
                        time_val /= 1_000
                    elif time_unit == "s":
                        time_val *= 1_000

                    # Store the result
                    if "args" not in results[current_benchmark]:
                        results[current_benchmark]["args"] = []
                        results[current_benchmark]["times"] = []

                    results[current_benchmark]["args"].append(param)
                    results[current_benchmark]["times"].append(time_val)

    return results


def create_visualizations(results):
    """Create visualizations of the benchmark results."""
    fig_count = 0

    for benchmark, data in results.items():
        if "args" not in data or not data["args"]:
            continue

        # Format the benchmark name for display
        display_name = benchmark.replace("bench_", "").replace("_", " ").title()

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Determine plot type based on args
        args = data["args"]
        times = data["times"]

        if "true" in args or "false" in args:
            # Bar plot for boolean args
            labels = args
            plt.bar(labels, times)
            plt.ylabel("Time (ms)")
            plt.title(f"{display_name} - Performance Comparison")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
        else:
            # Try to convert args to numeric values for line plot
            try:
                x_values = [int(arg) for arg in args]
                plt.plot(x_values, times, "o-", linewidth=2)
                plt.xlabel("Parameter Value")
                plt.ylabel("Time (ms)")
                plt.title(f"{display_name} - Performance Scaling")
                plt.grid(True, linestyle="--", alpha=0.7)

                # Add log scale if range is large
                if max(x_values) / min(x_values) > 10:
                    plt.xscale("log")
                    plt.xticks(x_values, [str(x) for x in x_values])
            except ValueError:
                # Fallback to bar plot for non-numeric args
                plt.bar(args, times)
                plt.ylabel("Time (ms)")
                plt.title(f"{display_name} - Performance Comparison")
                plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"benchmark_{benchmark}.png", dpi=150)
        print(f"Saved visualization for {benchmark} to benchmark_{benchmark}.png")
        fig_count += 1

    return fig_count


def main():
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], "r") as f:
            lines = f.readlines()
    else:
        # Read from stdin
        lines = sys.stdin.readlines()

    results = parse_divan_results(lines)
    fig_count = create_visualizations(results)

    if fig_count == 0:
        print(
            "No benchmark results found to visualize. Make sure you're piping divan output."
        )
    else:
        print(f"Created {fig_count} visualizations.")


if __name__ == "__main__":
    main()
