#!/usr/bin/env python3
"""
Clean Results Extractor for Sentence Transformers
================================================

Extracts NDCG, FLOPS, and Active Dims metrics with:
- Sub-columns by context length (256, 512)
- Bold separators between custom and pretrained models
- Visual highlighting for best performers
- Timestamp-based file naming in timestamped subfolders
- Excel and HTML outputs
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Configuration
RESULTS_DIR = "/home/user/Desktop/project/sentence-transformers/results"
AGGREGATE_BASE_DIR = "/home/user/Desktop/project/sentence-transformers/res_analyse/aggregate"
CONTEXT_LENGTHS = [256, 512]
CUSTOM_MODEL_PREFIX = "models__"

DATASETS = [
    "NanoMSMARCO",
    "NanoNQ",
    "NanoFEVER",
    "NanoHotpotQA",
    "NanoFiQA2018",
    "NanoArguAna",
    "NanoSciFact",
    "NanoQuoraRetrieval",
    "NanoDBPedia",
    "NanoSCIDOCS",
    "NanoNFCorpus",
    "NanoClimateFEVER",
    "NanoTouche2020",
]

METRICS = {
    "ndcg": "dot-NDCG@10",
    "query_dims": "query_active_dims",
    "corpus_dims": "corpus_active_dims",
    "flops": "avg_flops",
}

# Styling
BEST_FONT = Font(bold=True, color="FF0000")
TOP_10_FONT = Font(underline="single")
HEADER_FONT = Font(bold=True, size=12)
SEPARATOR_FILL = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
CONTEXT_FILL = PatternFill(start_color="E6F3FF", end_color="E6F3FF", fill_type="solid")


def clean_model_name(dir_name):
    """Extract clean model name and determine if custom"""
    is_custom = dir_name.startswith(CUSTOM_MODEL_PREFIX)
    name = re.sub(r"^(models__|naver__|prithivida__|opensearch-project__|ibm-granite__)", "", dir_name)
    name = re.sub(r"__checkpoint-\d+$", "", name)
    return name, is_custom


def load_results(model_dir, context_length):
    """Load mean and dataset results for a model"""
    results_path = os.path.join(RESULTS_DIR, model_dir, f"NanoBEIR_{context_length}")

    # Load mean results
    mean_file = os.path.join(results_path, "NanoBEIR_evaluation_mean_results.csv")
    mean_data = None
    if os.path.exists(mean_file):
        df = pd.read_csv(mean_file)
        mean_data = df.iloc[-1]

    # Load dataset results
    dataset_data = {}
    for dataset in DATASETS:
        file_path = os.path.join(results_path, f"Information-Retrieval_evaluation_{dataset}_results.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dataset_data[dataset] = df.iloc[-1]

    return mean_data, dataset_data


def format_value(value, decimal_places=2):
    """Format numeric value"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimal_places}f}"


def extract_all_data():
    """Extract all model data"""
    models = {"custom": [], "pretrained": []}
    all_data = {}

    for model_dir in sorted(os.listdir(RESULTS_DIR)):
        if not os.path.isdir(os.path.join(RESULTS_DIR, model_dir)):
            continue

        model_name, is_custom = clean_model_name(model_dir)
        category = "custom" if is_custom else "pretrained"
        models[category].append(model_name)

        all_data[model_name] = {"type": category}

        for context_length in CONTEXT_LENGTHS:
            mean_data, dataset_data = load_results(model_dir, context_length)
            all_data[model_name][context_length] = {"mean": mean_data, "datasets": dataset_data}

    return models, all_data


def create_summary_table(models, all_data):
    """Create summary table with sub-columns"""
    rows = []

    # Add custom models
    for model_name in sorted(models["custom"]):
        row = {"Model": model_name, "Type": "Custom"}

        for context in CONTEXT_LENGTHS:
            mean_data = all_data[model_name][context]["mean"]
            if mean_data is not None:
                row[f"{context}_NDCG"] = format_value(mean_data.get(METRICS["ndcg"], np.nan) * 100)
                row[f"{context}_FLOPS"] = format_value(mean_data.get(METRICS["flops"], np.nan))
                row[f"{context}_QueryDims"] = format_value(mean_data.get(METRICS["query_dims"], np.nan), 1)
                row[f"{context}_CorpusDims"] = format_value(mean_data.get(METRICS["corpus_dims"], np.nan), 1)
            else:
                for metric in ["NDCG", "FLOPS", "QueryDims", "CorpusDims"]:
                    row[f"{context}_{metric}"] = "N/A"
        rows.append(row)

    # Add separator
    if models["custom"] and models["pretrained"]:
        sep_row = {"Model": "--- PRETRAINED MODELS ---", "Type": "Separator"}
        for context in CONTEXT_LENGTHS:
            for metric in ["NDCG", "FLOPS", "QueryDims", "CorpusDims"]:
                sep_row[f"{context}_{metric}"] = "---"
        rows.append(sep_row)

    # Add pretrained models
    for model_name in sorted(models["pretrained"]):
        row = {"Model": model_name, "Type": "Pretrained"}

        for context in CONTEXT_LENGTHS:
            mean_data = all_data[model_name][context]["mean"]
            if mean_data is not None:
                row[f"{context}_NDCG"] = format_value(mean_data.get(METRICS["ndcg"], np.nan) * 100)
                row[f"{context}_FLOPS"] = format_value(mean_data.get(METRICS["flops"], np.nan))
                row[f"{context}_QueryDims"] = format_value(mean_data.get(METRICS["query_dims"], np.nan), 1)
                row[f"{context}_CorpusDims"] = format_value(mean_data.get(METRICS["corpus_dims"], np.nan), 1)
            else:
                for metric in ["NDCG", "FLOPS", "QueryDims", "CorpusDims"]:
                    row[f"{context}_{metric}"] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows)


def create_dataset_table(models, all_data, context_length):
    """Create dataset table for specific context length"""
    rows = []

    # Custom models
    for model_name in sorted(models["custom"]):
        row = {"Model": model_name, "Type": "Custom"}

        mean_data = all_data[model_name][context_length]["mean"]
        dataset_data = all_data[model_name][context_length]["datasets"]

        # Add dataset columns
        for dataset in DATASETS:
            if dataset in dataset_data:
                data = dataset_data[dataset]
                ndcg = format_value(data.get(METRICS["ndcg"], np.nan) * 100)
                flops = format_value(data.get(METRICS["flops"], np.nan))
                row[dataset] = f"{ndcg} ({flops})"
            else:
                row[dataset] = "N/A"

        # Add average
        if mean_data is not None:
            ndcg = format_value(mean_data.get(METRICS["ndcg"], np.nan) * 100)
            flops = format_value(mean_data.get(METRICS["flops"], np.nan))
            row["Average"] = f"{ndcg} ({flops})"
        else:
            row["Average"] = "N/A"

        rows.append(row)

    # Separator
    if models["custom"] and models["pretrained"]:
        sep_row = {"Model": "--- PRETRAINED MODELS ---", "Type": "Separator"}
        for col in DATASETS + ["Average"]:
            sep_row[col] = "---"
        rows.append(sep_row)

    # Pretrained models
    for model_name in sorted(models["pretrained"]):
        row = {"Model": model_name, "Type": "Pretrained"}

        mean_data = all_data[model_name][context_length]["mean"]
        dataset_data = all_data[model_name][context_length]["datasets"]

        # Add dataset columns
        for dataset in DATASETS:
            if dataset in dataset_data:
                data = dataset_data[dataset]
                ndcg = format_value(data.get(METRICS["ndcg"], np.nan) * 100)
                flops = format_value(data.get(METRICS["flops"], np.nan))
                row[dataset] = f"{ndcg} ({flops})"
            else:
                row[dataset] = "N/A"

        # Add average
        if mean_data is not None:
            ndcg = format_value(mean_data.get(METRICS["ndcg"], np.nan) * 100)
            flops = format_value(mean_data.get(METRICS["flops"], np.nan))
            row["Average"] = f"{ndcg} ({flops})"
        else:
            row["Average"] = "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def identify_best_performers(df):
    """Find best and top 10% performers"""
    performance = {}

    for col in df.columns:
        if col in ["Model", "Type"]:
            continue

        valid_data = df[(df["Type"] != "Separator") & (df[col] != "---") & (df[col] != "N/A")]
        if len(valid_data) == 0:
            continue

        # Determine if higher or lower is better
        is_ndcg = "NDCG" in col or any(d in col for d in DATASETS) or col == "Average"
        ascending = not is_ndcg  # NDCG higher is better, FLOPS lower is better

        # Extract numeric values
        numeric_vals = []
        for idx, value in valid_data[col].items():
            try:
                if "(" in str(value):
                    num_val = float(str(value).split("(")[0].strip())
                else:
                    num_val = float(value)
                numeric_vals.append((idx, num_val))
            except (ValueError, TypeError):
                continue

        if not numeric_vals:
            continue

        # Sort and identify best/top 10%
        sorted_vals = sorted(numeric_vals, key=lambda x: x[1], reverse=not ascending)
        best_idx = sorted_vals[0][0]
        top_10_count = max(1, len(sorted_vals) // 10)
        top_10_indices = [idx for idx, _ in sorted_vals[:top_10_count]]

        performance[col] = {"best": best_idx, "top_10": top_10_indices}

    return performance


def save_excel_with_formatting(summary_df, dataset_dfs, perf_data, output_dir):
    """Save to Excel with proper formatting"""
    filename = "results.xlsx"
    filepath = os.path.join(output_dir, filename)

    wb = Workbook()
    wb.remove(wb.active)

    # Summary sheet
    ws_summary = wb.create_sheet("Summary")

    # Create multi-level headers
    ws_summary.cell(1, 1, "Model")
    ws_summary.cell(1, 2, "Type")
    ws_summary.merge_cells("A1:A2")
    ws_summary.merge_cells("B1:B2")

    col_idx = 3
    col_mapping = {}

    for context in CONTEXT_LENGTHS:
        start_col = col_idx
        ws_summary.cell(1, col_idx, f"Context {context}")

        for metric in ["NDCG", "FLOPS", "QueryDims", "CorpusDims"]:
            ws_summary.cell(2, col_idx, metric)
            col_mapping[f"{context}_{metric}"] = col_idx
            col_idx += 1

        end_col = col_idx - 1
        ws_summary.merge_cells(f"{get_column_letter(start_col)}1:{get_column_letter(end_col)}1")

    # Add data
    for row_idx, (_, row) in enumerate(summary_df.iterrows(), start=3):
        ws_summary.cell(row_idx, 1, row["Model"])
        ws_summary.cell(row_idx, 2, row["Type"])

        for col_name, col_pos in col_mapping.items():
            value = row.get(col_name, "N/A")
            cell = ws_summary.cell(row_idx, col_pos, value)

            # Apply highlighting
            if col_name in perf_data.get("summary", {}) and row["Type"] != "Separator":
                col_perf = perf_data["summary"][col_name]
                df_row_idx = row_idx - 3

                if df_row_idx == col_perf.get("best"):
                    cell.font = BEST_FONT
                elif df_row_idx in col_perf.get("top_10", []):
                    cell.font = TOP_10_FONT

    # Format headers
    for col in range(1, col_idx):
        ws_summary.cell(1, col).font = HEADER_FONT
        ws_summary.cell(1, col).alignment = Alignment(horizontal="center")
        ws_summary.cell(2, col).font = HEADER_FONT
        ws_summary.cell(2, col).alignment = Alignment(horizontal="center")
        if col > 2:
            ws_summary.cell(1, col).fill = CONTEXT_FILL

    # Format separator rows
    for row_idx in range(3, len(summary_df) + 3):
        if summary_df.iloc[row_idx - 3]["Type"] == "Separator":
            for col in range(1, col_idx):
                cell = ws_summary.cell(row_idx, col)
                cell.fill = SEPARATOR_FILL
                cell.font = Font(bold=True)

    # Auto-adjust column widths
    for col in range(1, col_idx):
        ws_summary.column_dimensions[get_column_letter(col)].width = 15

    # Dataset sheets
    for context, df in dataset_dfs.items():
        ws = wb.create_sheet(f"Datasets_{context}")

        # Add headers
        for col_idx, col_name in enumerate(df.columns, 1):
            ws.cell(1, col_idx, col_name).font = HEADER_FONT

        # Add data
        for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
            for col_idx, (col_name, value) in enumerate(row.items(), 1):
                cell = ws.cell(row_idx, col_idx, value)

                # Apply highlighting for separator rows
                if row["Type"] == "Separator":
                    cell.fill = SEPARATOR_FILL
                    cell.font = Font(bold=True)

        # Auto-adjust widths
        for col in range(1, len(df.columns) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 12

    wb.save(filepath)
    print(f"Saved Excel: {filename}")


def save_html_with_tabs(summary_df, dataset_dfs, perf_data, output_dir, timestamp):
    """Save to HTML with tabbed interface"""
    filename = "results.html"
    filepath = os.path.join(output_dir, filename)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentence Transformers Results - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .tabs {{ display: flex; background-color: #f1f1f1; border-radius: 8px 8px 0 0; }}
            .tab {{ background-color: inherit; border: none; outline: none; cursor: pointer;
                   padding: 14px 20px; transition: 0.3s; font-size: 16px; }}
            .tab:hover {{ background-color: #ddd; }}
            .tab.active {{ background-color: #007acc; color: white; }}
            .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; }}
            .tabcontent.active {{ display: block; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .context-header {{ background-color: #e6f3ff; font-weight: bold; font-size: 14px; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .separator {{ background-color: #cccccc; font-weight: bold; }}
            .best {{ color: red; font-weight: bold; }}
            .top10 {{ text-decoration: underline; }}
            .model-name {{ text-align: left; font-weight: bold; }}
            h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
            h2 {{ color: #666; margin-bottom: 20px; }}
            .timestamp {{ text-align: center; color: #888; margin-bottom: 20px; }}
        </style>
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tabs;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].classList.remove("active");
                }}
                tabs = document.getElementsByClassName("tab");
                for (i = 0; i < tabs.length; i++) {{
                    tabs[i].classList.remove("active");
                }}
                document.getElementById(tabName).classList.add("active");
                evt.currentTarget.classList.add("active");
            }}
        </script>
    </head>
    <body>
        <h1>🚀 Sentence Transformers Model Results</h1>
        <div class="timestamp">Generated on: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</div>

        <div class="tabs">
            <button class="tab active" onclick="openTab(event, 'summary')">📊 Summary</button>
            <button class="tab" onclick="openTab(event, 'datasets_256')">📋 Context 256</button>
            <button class="tab" onclick="openTab(event, 'datasets_512')">📋 Context 512</button>
        </div>

        <div id="summary" class="tabcontent active">
            <h2>Summary - Average Metrics by Context Length</h2>
            {df_to_html_with_subcolumns(summary_df, perf_data.get("summary", {}))}
        </div>

        <div id="datasets_256" class="tabcontent">
            <h2>Individual Datasets - Context Length 256</h2>
            {df_to_html_simple(dataset_dfs[256], perf_data.get("datasets_256", {}))}
        </div>

        <div id="datasets_512" class="tabcontent">
            <h2>Individual Datasets - Context Length 512</h2>
            {df_to_html_simple(dataset_dfs[512], perf_data.get("datasets_512", {}))}
        </div>

    </body>
    </html>
    """

    with open(filepath, "w") as f:
        f.write(html_content)
    print(f"Saved HTML: {filename}")


def df_to_html_with_subcolumns(df, perf_data):
    """Convert summary DataFrame to HTML with sub-column headers"""
    html = "<table>"

    # Create multi-level header
    html += "<tr>"
    html += "<th rowspan='2'>Model</th>"
    html += "<th rowspan='2'>Type</th>"

    # Context length headers
    for context in CONTEXT_LENGTHS:
        html += f"<th colspan='4' class='context-header'>Context {context}</th>"
    html += "</tr>"

    # Sub-headers
    html += "<tr>"
    for context in CONTEXT_LENGTHS:
        html += "<th>NDCG</th><th>FLOPS</th><th>QueryDims</th><th>CorpusDims</th>"
    html += "</tr>"

    # Data rows
    for idx, row in df.iterrows():
        row_class = "separator" if row["Type"] == "Separator" else ""
        html += f"<tr class='{row_class}'>"

        html += f"<td class='model-name'>{row['Model']}</td>"
        html += f"<td>{row['Type']}</td>"

        # Add metric values
        for context in CONTEXT_LENGTHS:
            for metric in ["NDCG", "FLOPS", "QueryDims", "CorpusDims"]:
                col_name = f"{context}_{metric}"
                value = row.get(col_name, "N/A")

                cell_class = ""
                if col_name in perf_data and row["Type"] != "Separator":
                    col_perf = perf_data[col_name]
                    if idx == col_perf.get("best"):
                        cell_class = "best"
                    elif idx in col_perf.get("top_10", []):
                        cell_class = "top10"

                html += f"<td class='{cell_class}'>{value}</td>"

        html += "</tr>"

    html += "</table>"
    return html


def df_to_html_simple(df, perf_data):
    """Convert DataFrame to HTML with simple styling"""
    html = "<table>"

    # Header
    html += "<tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"

    # Rows
    for idx, row in df.iterrows():
        row_class = "separator" if row["Type"] == "Separator" else ""
        html += f"<tr class='{row_class}'>"

        for col_name, value in row.items():
            cell_class = ""

            # Apply performance highlighting
            if col_name in perf_data and row["Type"] != "Separator":
                col_perf = perf_data[col_name]
                if idx == col_perf.get("best"):
                    cell_class = "best"
                elif idx in col_perf.get("top_10", []):
                    cell_class = "top10"

            # Special formatting for model names
            if col_name == "Model":
                cell_class += " model-name"

            html += f"<td class='{cell_class}'>{value}</td>"

        html += "</tr>"

    html += "</table>"
    return html


def main():
    """Main execution"""
    print("Extracting Sentence Transformers Results...")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped output directory
    output_dir = os.path.join(AGGREGATE_BASE_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Extract data
    models, all_data = extract_all_data()
    print(f"Found {len(models['custom'])} custom models, {len(models['pretrained'])} pretrained models")

    # Create tables
    summary_df = create_summary_table(models, all_data)
    dataset_dfs = {}

    for context in CONTEXT_LENGTHS:
        dataset_dfs[context] = create_dataset_table(models, all_data, context)

    # Identify best performers
    summary_perf = identify_best_performers(summary_df)
    dataset_perf = {}
    for context, df in dataset_dfs.items():
        dataset_perf[f"datasets_{context}"] = identify_best_performers(df)

    perf_data = {"summary": summary_perf, **dataset_perf}

    # Save files
    save_excel_with_formatting(summary_df, dataset_dfs, perf_data, output_dir)
    save_html_with_tabs(summary_df, dataset_dfs, perf_data, output_dir, timestamp)

    # Save CSV files
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    for context, df in dataset_dfs.items():
        df.to_csv(os.path.join(output_dir, f"datasets_{context}.csv"), index=False)

    print(f"Results saved in timestamped folder: {timestamp}")
    print("Done! 🎉")


if __name__ == "__main__":
    main()
