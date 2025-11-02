#!/usr/bin/env python3
"""
Generate performance charts from benchmark history.

This script reads the benchmark history and creates static PNG charts
that are embedded directly in the README.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_history(history_path: Path) -> List[Dict[str, Any]]:
    """Load benchmark history from JSON file."""
    if not history_path.exists():
        print(f"Warning: {history_path} not found. Creating empty history.")
        return []

    with open(history_path, 'r') as f:
        return json.load(f)


def create_nms_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Create NMS performance trends chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("NMS: Time vs. Input Size (COCO Dataset)", "NMS: Throughput vs. Input Size"),
        vertical_spacing=0.15,
    )

    # Extract NMS benchmarks
    nms_cols = [col for col in df.columns if col.startswith('nms_') and not col.endswith('_throughput')]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, col in enumerate(nms_cols):
        box_count = col.replace('nms_', '')
        throughput_col = f"{col}_throughput"

        # Time plot (row 1)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df[col],
                mode='lines+markers',
                name=f'{box_count} boxes',
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8),
                legendgroup=box_count,
            ),
            row=1, col=1
        )

        # Throughput plot (row 2)
        if throughput_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[throughput_col] / 1000,  # Convert to k boxes/sec
                    mode='lines+markers',
                    name=f'{box_count} boxes',
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=8),
                    legendgroup=box_count,
                    showlegend=False,
                ),
                row=2, col=1
            )

    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True)
    fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Throughput (k boxes/sec)", row=2, col=1, showgrid=True)

    fig.update_layout(
        height=700,
        title_text="NMS Performance Over Time",
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(size=12),
    )

    return fig


def create_mask_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Create mask_to_polygons performance trends chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Mask to Polygons: Time vs. Image Size",
            "Mask to Polygons: Throughput vs. Image Size"
        ),
        vertical_spacing=0.15,
    )

    # Extract mask benchmarks
    mask_cols = [col for col in df.columns if col.startswith('mask_to_polygons_') and not col.endswith('_throughput')]

    colors = ['#9467bd', '#8c564b', '#e377c2']

    for idx, col in enumerate(mask_cols):
        size = col.replace('mask_to_polygons_', '')
        throughput_col = f"{col}_throughput"

        # Time plot (row 1)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df[col],
                mode='lines+markers',
                name=f'{size}x{size}',
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8),
                legendgroup=size,
            ),
            row=1, col=1
        )

        # Throughput plot (row 2)
        if throughput_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[throughput_col],
                    mode='lines+markers',
                    name=f'{size}x{size}',
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=8),
                    legendgroup=size,
                    showlegend=False,
                ),
                row=2, col=1
            )

    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True)
    fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Throughput (megapixels/sec)", row=2, col=1, showgrid=True)

    fig.update_layout(
        height=700,
        title_text="Mask to Polygons Performance Over Time",
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(size=12),
    )

    return fig


def history_to_dataframe(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert history to a pandas DataFrame."""
    if not history:
        return pd.DataFrame()

    records = []
    for entry in history:
        record = {
            'timestamp': entry['timestamp'],
            'commit': entry['commit'],
            'branch': entry['branch'],
        }

        # Extract benchmark metrics
        for bench_name, bench_data in entry['benchmarks'].items():
            if 'time_ms' in bench_data:
                record[bench_name] = bench_data['time_ms']

            if 'throughput_boxes_per_sec' in bench_data:
                record[f"{bench_name}_throughput"] = bench_data['throughput_boxes_per_sec']
            elif 'throughput_megapixels_per_sec' in bench_data:
                record[f"{bench_name}_throughput"] = bench_data['throughput_megapixels_per_sec']

        records.append(record)

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def generate_charts(history_path: Path, output_dir: Path):
    """Generate performance charts and save as PNG images."""
    print(f"Loading benchmark history from {history_path}...")
    history = load_history(history_path)

    if not history:
        print("No benchmark data found. Skipping chart generation.")
        return

    df = history_to_dataframe(history)
    print(f"Generating charts with {len(df)} benchmark runs...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save NMS chart
    nms_chart = create_nms_performance_chart(df)
    nms_path = output_dir / "nms_performance.png"
    nms_chart.write_image(nms_path, width=1200, height=700, scale=2)
    print(f"✓ Generated: {nms_path}")

    # Generate and save mask chart
    mask_chart = create_mask_performance_chart(df)
    mask_path = output_dir / "mask_performance.png"
    mask_chart.write_image(mask_path, width=1200, height=700, scale=2)
    print(f"✓ Generated: {mask_path}")

    print(f"\nCharts saved to {output_dir}")
    print("These images are referenced in benchmarks/README.md")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    history_path = script_dir / "history.json"
    output_dir = script_dir

    generate_charts(history_path, output_dir)


if __name__ == "__main__":
    main()
