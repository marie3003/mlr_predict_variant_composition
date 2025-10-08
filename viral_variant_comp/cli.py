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

    # Calculate daily raw frequencies
    df_daily_freqs = df_counts.div(df_counts.sum(axis=1), axis=0)

    # Calculate weekly raw frequencies (7-day rolling mean)
    df_weekly_freqs = df_counts.rolling(window=7, min_periods=1).sum()
    df_weekly_freqs = df_weekly_freqs.div(df_weekly_freqs.sum(axis=1), axis=0)

    return df_daily_freqs, df_weekly_freqs


def to_nextstrain_format(
    variant_names,
    dates,
    frequencies,
    growth_rates,
    raw_frequencies,
    weekly_raw_frequencies,
    std_errors,
    locations=None,
    pivot_variant=None,
    variant_display_names=None,
    updated=None,
    csv_path=None,
    grouping_col=None,
):
    if locations is None:
        locations = ["default"]

    if variant_display_names is None:
        variant_display_names = {name: name for name in variant_names}

    if updated is None:
        updated = pd.to_datetime("today").strftime("%Y-%m-%d")

    metadata = {
        "ps": ["median", "HDI_50_upper", "HDI_50_lower", "HDI_80_upper", "HDI_80_lower", "HDI_95_upper", "HDI_95_lower"],
        "sites": ["freq", "ga", "daily_raw_freq", "weekly_raw_freq"],
        "location": locations,
        "dates": [date.strftime("%Y-%m-%d") for date in dates],
        "variants": variant_names.tolist(),
        "variantDisplayNames": variant_display_names,
        "pivot": pivot_variant or (variant_names[0] if len(variant_names) > 0 else None),
        "updated": updated,
        "forecast_dates": [date.strftime("%Y-%m-%d") for date in dates][-7:],  # Last 7 dates as forecast dates as an example
        "variantColors": {name: f"#{hash(name) % 0xFFFFFF:06x}" for name in variant_names},  # Generate simple colors for each variant
    }

    n_variants = len(variant_names)
    # The first variant is the pivot, so its std errors are 0
    growth_rate_std_errors_full = np.zeros(n_variants)
    log_init_freq_std_errors_full = np.zeros(n_variants)

    growth_rate_std_errors_full[1:] = std_errors[:n_variants - 1]
    log_init_freq_std_errors_full[1:] = std_errors[n_variants - 1:]

    data = []
    for location in locations:
        for i, variant in enumerate(variant_names):
            # Frequencies
            for j, date in enumerate(dates):
                # Calculate frequency confidence intervals using parameter standard errors
                # Use the delta method approximation: if f = exp(a*t + b) / sum(exp(a*t + b)), then 
                # the derivative can be approximated using the parameter standard errors
                freq_value = frequencies[j, i]
                
                # For frequency confidence intervals, we'll use a simplified approach
                # based on the log_init_freq std errors
                freq_se = 0
                if i < len(log_init_freq_std_errors_full):  # Make sure we don't go out of bounds
                    # Calculate the standard error for the frequency using the parameter SEs
                    # This is a simplified approach using the std error of the initial log frequency
                    # For more accuracy, we would need to compute the full Jacobian and apply the delta method
                    freq_se = log_init_freq_std_errors_full[i] * freq_value  # Approximation
                    t = j  # Current time point (starting from 0)
                    
                    # Calculate growth contribution to variance
                    if i < len(growth_rate_std_errors_full):
                        growth_se_contrib = growth_rate_std_errors_full[i] * t * freq_value * (1 - freq_value)  # More appropriate formula
                        freq_se = np.sqrt(freq_se**2 + growth_se_contrib**2)  # Combined standard error
                else:
                    freq_se = 0  # For safety

                # Calculate confidence intervals - add small epsilon to avoid zero-width intervals
                if freq_se == 0:
                    freq_se = 1e-6  # Small value to ensure we have some CI

                freq_lower_50 = max(0, freq_value - 0.67 * freq_se)
                freq_upper_50 = min(1, freq_value + 0.67 * freq_se)
                freq_lower_80 = max(0, freq_value - 1.28 * freq_se)
                freq_upper_80 = min(1, freq_value + 1.28 * freq_se)
                freq_lower_95 = max(0, freq_value - 1.96 * freq_se)
                freq_upper_95 = min(1, freq_value + 1.96 * freq_se)

                data.append({
                    "location": location,
                    "site": "freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": freq_value,
                    "ps": "median",
                })
                data.append({
                    "location": location,
                    "site": "freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": freq_lower_50,
                    "ps": "HDI_50_lower",
                })
                data.append({
                    "location": location,
                    "site": "freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": freq_upper_50,
                    "ps": "HDI_50_upper",
                })
                data.append({
                    "location": location,
                    "site": "freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": freq_lower_80,
                    "ps": "HDI_80_lower",
                })
                data.append({
                    "location": location,
                    "site": "freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": freq_upper_80,
                    "ps": "HDI_80_upper",
                })
                data.append({
                    "location": location,
                    "site": "freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": freq_lower_95,
                    "ps": "HDI_95_lower",
                })
                data.append({
                    "location": location,
                    "site": "freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": freq_upper_95,
                    "ps": "HDI_95_upper",
                })

            # Growth Advantage
            data.append({
                "location": location,
                "site": "ga",
                "variant": variant,
                "date": dates[-1].strftime("%Y-%m-%d"),
                "value": np.exp(growth_rates[i]),
                "ps": "median",
            })
            # HDI for growth advantage (assuming normal distribution of growth_rate)
            for hdi_level, multiplier in [(0.5, 0.67), (0.8, 1.28), (0.95, 1.96)]:
                lower_bound = np.exp(growth_rates[i] - multiplier * growth_rate_std_errors_full[i])
                upper_bound = np.exp(growth_rates[i] + multiplier * growth_rate_std_errors_full[i])
                data.append({
                    "location": location,
                    "site": "ga",
                    "variant": variant,
                    "date": dates[-1].strftime("%Y-%m-%d"),
                    "value": lower_bound,
                    "ps": f"HDI_{int(hdi_level*100)}_lower",
                })
                data.append({
                    "location": location,
                    "site": "ga",
                    "variant": variant,
                    "date": dates[-1].strftime("%Y-%m-%d"),
                    "value": upper_bound,
                    "ps": f"HDI_{int(hdi_level*100)}_upper",
                })
        # Process daily raw frequencies (empirical frequencies from raw counts)
        # These should follow the real Nextstrain format without ps field
        df_counts = pd.read_csv(csv_path)
        df_counts["date"] = pd.to_datetime(df_counts["date"])
        df_counts_pivot = df_counts.groupby(["date", grouping_col])["count"].sum().unstack(fill_value=0)
        
        for date, row in raw_frequencies.iterrows():
            for variant in raw_frequencies.columns:
                raw_freq_value = row[variant]
                
                data.append({
                    "location": location,
                    "site": "daily_raw_freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": raw_freq_value,
                })

        # Process weekly raw frequencies (empirical frequencies from raw counts)
        for date, row in weekly_raw_frequencies.iterrows():
            for variant in weekly_raw_frequencies.columns:
                raw_freq_value = row[variant]
                
                data.append({
                    "location": location,
                    "site": "weekly_raw_freq",
                    "variant": variant,
                    "date": date.strftime("%Y-%m-%d"),
                    "value": raw_freq_value,
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
        df_daily_freqs, df_weekly_freqs = calculate_raw_frequencies(args.input_csv, args.grouping_column)
        nextstrain_json = to_nextstrain_format(
            variant_names=data["variant_names"],
            dates=data["counts_df_pivot"].index.to_pydatetime(),
            frequencies=results["composition_estimate"],
            growth_rates=results["growth_rate_estimate"],
            raw_frequencies=df_daily_freqs,
            weekly_raw_frequencies=df_weekly_freqs,
            std_errors=results["std_errors"],
            pivot_variant=args.pivot_variant,
            csv_path=args.input_csv,
            grouping_col=args.grouping_column,
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
