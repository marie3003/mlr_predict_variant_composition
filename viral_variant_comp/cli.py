import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from viral_variant_comp.estimate import (
    BaseCompositionEstimator,
    BFGSCompositionEstimator,
    StepwiseBFGSCompositionEstimator,
    choose_pivot_variant,
)
from viral_variant_comp.simulate_count_data import preprocess_covid_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit viral variant growth rates and frequencies from lineage count data."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the lineage count CSV. Expected columns: date, count, grouping column.",
    )
    parser.add_argument(
        "--grouping-column",
        default="nextcladePangoLineage",
        help="Column name used to group sequences into variants (default: nextcladePangoLineage).",
    )
    parser.add_argument(
        "--estimator",
        choices=("bfgs", "stepwise"),
        default="stepwise",
        help="Estimator to use for the multinomial logistic regression fit (default: stepwise).",
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        default=12,
        help="Partition size for the stepwise estimator; ignored when using plain BFGS.",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=6,
        help="Overlap size for the stepwise estimator; ignored when using plain BFGS.",
    )
    parser.add_argument(
        "--pivot-variant",
        help="Optional variant name to use as the fixed pivot (first column) in the counts matrix.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to write the results (default: results.json).",
    )
    parser.add_argument(
        "--output-format",
        choices=("default", "nextstrain"),
        default="default",
        help="Output format for the results (default: default).",
    )
    return parser.parse_args()


def load_counts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"date", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}")
    return df


def build_estimator(
    name: str, counts, partition_size: Optional[int], overlap_size: Optional[int]
) -> BaseCompositionEstimator:
    if name == "bfgs":
        return BFGSCompositionEstimator(counts)

    if partition_size is None or overlap_size is None:
        raise ValueError("Stepwise estimator requires --partition-size and --overlap-size.")

    n_variants = counts.shape[1]
    partition = max(2, min(partition_size, n_variants))
    overlap = max(1, min(overlap_size, partition - 1))

    return StepwiseBFGSCompositionEstimator(counts, partition_size=partition, overlap_size=overlap)


def ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parameters_payload(
    estimator_name: str,
    grouping: str,
    pivot: Optional[str],
    variant_names,
    growth,
    intercept,
):
    variants = []
    for idx, name in enumerate(variant_names):
        variants.append(
            {
                "variant": str(name),
                "fitness": float(growth[idx]),
                "intercept": float(intercept[idx]),
            }
        )
    return {
        "estimator": estimator_name,
        "grouping_column": grouping,
        "pivot_variant": pivot,
        "variants": variants,
    }


def frequencies_payload(
    estimator_name: str,
    grouping: str,
    pivot: Optional[str],
    variant_names,
    dates,
    frequencies,
):
    series = []
    for col_idx, name in enumerate(variant_names):
        series.append(
            {
                "variant": str(name),
                "frequencies": [
                    {"date": date.isoformat(), "frequency": float(frequencies[row_idx][col_idx])}
                    for row_idx, date in enumerate(dates)
                ],
            }
        )
    return {
        "estimator": estimator_name,
        "grouping_column": grouping,
        "pivot_variant": pivot,
        "variants": series,
    }


def calculate_raw_frequencies(csv_path, grouping_col):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df_counts = df.groupby(["date", grouping_col])["count"].sum().unstack(fill_value=0)
    df_freqs = df_counts.div(df_counts.sum(axis=1), axis=0)
    return df_freqs


def to_nextstrain_format(
    variant_names,
    dates,
    frequencies,
    growth_rates,
    raw_frequencies,
    locations=None,
    pivot_variant=None,
    variant_display_names=None,
    updated=None,
):
    if locations is None:
        locations = ["default"]

    if variant_display_names is None:
        variant_display_names = {name: name for name in variant_names}

    if updated is None:
        updated = pd.to_datetime("today").strftime("%Y-%m-%d")

    metadata = {
        "ps": ["median"],
        "sites": ["freq", "ga", "daily_raw_freq"],
        "location": locations,
        "dates": [date.strftime("%Y-%m-%d") for date in dates],
        "variants": variant_names.tolist(),
        "variantDisplayNames": variant_display_names,
        "pivot": pivot_variant,
        "updated": updated,
    }

    data = []
    for location in locations:
        for i, variant in enumerate(variant_names):
            for j, date in enumerate(dates):
                data.append({
                    "location": location,
                    "site": "freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": frequencies[j, i],
                    "ps": "median",
                })
            data.append({
                "location": location,
                "site": "ga",
                "variant": variant,
                "date": dates[-1].strftime("%Y-%m-%d"),
                "value": np.exp(growth_rates[i]),
                "ps": "median",
            })
        for date, row in raw_frequencies.iterrows():
            for variant in raw_frequencies.columns:
                data.append({
                    "location": location,
                    "site": "daily_raw_freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": row[variant],
                    "ps": "median",
                })


    return {"metadata": metadata, "data": data}


def main() -> None:
    args = parse_args()
    counts_df = load_counts(args.input_csv)

    if args.grouping_column not in counts_df.columns:
        raise ValueError(
            f"Column '{args.grouping_column}' not found in {args.input_csv}. "
            "Use --grouping-column to pick an existing column."
        )

    data = preprocess_covid_data(counts_df, grouping_col=args.grouping_column)

    if args.pivot_variant:
        if args.pivot_variant not in set(data["variant_names"]):
            raise ValueError(
                f"Pivot variant '{args.pivot_variant}' not found after preprocessing."
            )
        data = choose_pivot_variant(data, pivot=args.pivot_variant)

    estimator = build_estimator(
        args.estimator, data["counts"], args.partition_size, args.overlap_size
    )
    estimator.fit()
    results = estimator.get_results()

    if args.output_format == "nextstrain":
        raw_frequencies = calculate_raw_frequencies(args.input_csv, args.grouping_column)
        nextstrain_json = to_nextstrain_format(
            variant_names=data["variant_names"],
            dates=data["counts_df_pivot"].index.to_pydatetime(),
            frequencies=results["composition_estimate"],
            growth_rates=results["growth_rate_estimate"],
            raw_frequencies=raw_frequencies,
            pivot_variant=args.pivot_variant,
        )
        output_path = Path(args.output)
        write_json(output_path, nextstrain_json)
    else:
        params_json = parameters_payload(
            args.estimator,
            args.grouping_column,
            args.pivot_variant,
            data["variant_names"],
            results["growth_rate_estimate"],
            results["log_init_freq_estimate"],
        )

        freq_json = frequencies_payload(
            args.estimator,
            args.grouping_column,
            args.pivot_variant,
            data["variant_names"],
            data["counts_df_pivot"].index.to_pydatetime(),
            results["composition_estimate"],
        )

        params_path = Path(args.output)
        freq_path = Path(args.output.replace(".json", "_frequencies.json"))

        write_json(params_path, params_json)
        write_json(freq_path, freq_json)


if __name__ == "__main__":
    main()
