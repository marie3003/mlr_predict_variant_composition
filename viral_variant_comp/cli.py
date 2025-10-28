import argparse
import datetime
import io
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Index):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

# areguments that are optional start with --
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit viral variant growth rates and frequencies from lineage count data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Estimate command
    parser_estimate = subparsers.add_parser("estimate", help="Estimate model parameters.")
    parser_estimate.add_argument(
        "input_csv",
        help="Path to the lineage count CSV. Expected columns: date, count, grouping column.",
    )
    parser_estimate.add_argument(
        "--grouping-column",
        default="nextstrainClade",
        help="Column name used to group sequences into variants (default: nextstrainClade).",
    )
    parser_estimate.add_argument(
        "--estimator",
        choices=("bfgs", "stepwise"),
        default="stepwise",
        help="Estimator to use for the multinomial logistic regression fit (default: stepwise).",
    )
    parser_estimate.add_argument(
        "--partition-size",
        type=int,
        default=12,
        help="Partition size for the stepwise estimator; ignored when using plain BFGS.",
    )
    parser_estimate.add_argument(
        "--overlap-size",
        type=int,
        default=6,
        help="Overlap size for the stepwise estimator; ignored when using plain BFGS.",
    )
    parser_estimate.add_argument(
        "--pivot-variant",
        help="Optional variant name to use as the fixed pivot (first column) in the counts matrix.",
    )
    parser_estimate.add_argument(
        "--output",
        default="estimates.json",
        help="Path to write the estimated parameters and hessian (default: estimates.json).",
    )
    parser_estimate.add_argument(
        "--location-column",
        help="Optional: Column name for location data. If provided, run estimation for each location separately.",
    )

    # Visualize command
    parser_visualize = subparsers.add_parser("visualize", help="Generate visualization data.")
    parser_visualize.add_argument(
        "input_json",
        help="Path to the JSON file with estimated parameters and hessian.",
    )
    parser_visualize.add_argument(
        "--output",
        default="results.json",
        help="Path to write the results (default: results.json).",
    )
    parser_visualize.add_argument(
        "--location-name",
        help="Name for the location if not running by location.",
    )
    return parser.parse_args()


def load_counts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"date", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}")
    return df

def is_informative_dataset(df, grouping_col="nextstrainClade", min_total=50, min_overlap_days=2):
    total_counts = df["count"].sum()
    if total_counts < min_total:
        return False

    variant_totals = df.groupby(grouping_col)["count"].sum()
    n_nonzero = (variant_totals > 0).sum()
    if n_nonzero < 2:
        return False

    pivot = df.pivot_table(index="date", columns=grouping_col, values="count", aggfunc="sum", fill_value=0)
    n_overlap_days = ((pivot > 0).sum(axis=1) >= 2).sum()
    if n_overlap_days < min_overlap_days:
        return False

    return True


def build_estimator(
    name: str, counts, partition_size: Optional[int], overlap_size: Optional[int]
) -> BaseCompositionEstimator:
    if name == "bfgs":
        return BFGSCompositionEstimator(counts)

    if partition_size is None or overlap_size is None:
        raise ValueError("Stepwise estimator requires --partition-size and --overlap-size.")

    return StepwiseBFGSCompositionEstimator(counts, partition_size=partition_size, overlap_size=overlap_size)


def ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, cls=NumpyEncoder))


def calculate_raw_frequencies(df: pd.DataFrame, grouping_col: str):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df_counts = df.groupby(["date", grouping_col])["count"].sum().unstack(fill_value=0).sort_index()

    # Ignore recombinant nextstrain clades
    if grouping_col == "nextstrainClade" and "recombinant" in df_counts.columns:
        df_counts = df_counts.drop("recombinant", axis=1)

    daily_totals = df_counts.sum(axis=1)
    df_daily_freqs = df_counts.div(daily_totals.replace(0, np.nan), axis=0).fillna(0)

    # ensures that all days are present
    df_counts_daily = df_counts.asfreq("D", fill_value=0)
    df_weekly_counts = df_counts_daily.rolling(window=7, min_periods=1).sum()
    weekly_totals = df_weekly_counts.sum(axis=1)

    df_weekly_freqs = (
        df_weekly_counts
        .div(weekly_totals.replace(0, np.nan), axis=0)
        .fillna(0)
    )
    return df_daily_freqs, df_weekly_freqs


def generate_nextstrain_data_entries(
    location,
    variant_names,
    dates,
    frequencies,
    growth_rates,
    raw_frequencies,
    weekly_raw_frequencies,
    std_errors,
):
    data = []
    n_variants = len(variant_names)
    growth_rate_std_errors_full = np.zeros(n_variants)
    log_init_freq_std_errors_full = np.zeros(n_variants)

    growth_rate_std_errors_full[1:] = std_errors[:n_variants - 1]
    log_init_freq_std_errors_full[1:] = std_errors[n_variants - 1:]

    for i, variant in enumerate(variant_names):
        for j, date in enumerate(dates):
            freq_value = frequencies[j, i]
            freq_se = 0
            if i < len(log_init_freq_std_errors_full):
                freq_se = log_init_freq_std_errors_full[i] * freq_value
                t = j
                if i < len(growth_rate_std_errors_full):
                    growth_se_contrib = growth_rate_std_errors_full[i] * t * freq_value * (1 - freq_value)
                    freq_se = np.sqrt(freq_se**2 + growth_se_contrib**2)
            else:
                freq_se = 0

            if freq_se == 0:
                freq_se = 1e-6

            freq_lower_50 = np.clip(freq_value - 0.67 * freq_se, 0, 1)
            freq_upper_50 = np.clip(freq_value + 0.67 * freq_se, 0, 1)
            freq_lower_80 = np.clip(freq_value - 1.28 * freq_se, 0, 1)
            freq_upper_80 = np.clip(freq_value + 1.28 * freq_se, 0, 1)
            freq_lower_95 = np.clip(freq_value - 1.96 * freq_se, 0, 1)
            freq_upper_95 = np.clip(freq_value + 1.96 * freq_se, 0, 1)

            data.append({"location": location, "site": "freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": freq_value, "ps": "median"})
            data.append({"location": location, "site": "freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": freq_lower_50, "ps": "HDI_50_lower"})
            data.append({"location": location, "site": "freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": freq_upper_50, "ps": "HDI_50_upper"})
            data.append({"location": location, "site": "freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": freq_lower_80, "ps": "HDI_80_lower"})
            data.append({"location": location, "site": "freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": freq_upper_80, "ps": "HDI_80_upper"})
            data.append({"location": location, "site": "freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": freq_lower_95, "ps": "HDI_95_lower"})
            data.append({"location": location, "site": "freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": freq_upper_95, "ps": "HDI_95_upper"})

        data.append({"location": location, "site": "ga", "variant": variant, "date": dates[-1].strftime("%Y-%m-%d"), "value": np.exp(growth_rates[i]), "ps": "median"})
        for hdi_level, multiplier in [(0.5, 0.67), (0.8, 1.28), (0.95, 1.96)]:
            lower_log_val = growth_rates[i] - multiplier * growth_rate_std_errors_full[i]
            upper_log_val = growth_rates[i] + multiplier * growth_rate_std_errors_full[i]
            lower_log_val = np.clip(lower_log_val, -700, 700)
            upper_log_val = np.clip(upper_log_val, -700, 700)
            lower_bound = np.exp(lower_log_val)
            upper_bound = np.exp(upper_log_val)
            data.append({"location": location, "site": "ga", "variant": variant, "date": dates[-1].strftime("%Y-%m-%d"), "value": lower_bound, "ps": f"HDI_{int(hdi_level*100)}_lower"})
            data.append({"location": location, "site": "ga", "variant": variant, "date": dates[-1].strftime("%Y-%m-%d"), "value": upper_bound, "ps": f"HDI_{int(hdi_level*100)}_upper"})

    for date, row in raw_frequencies.iterrows():
        for variant in raw_frequencies.columns:
            raw_freq_value = row[variant]
            data.append({"location": location, "site": "daily_raw_freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": raw_freq_value, "ps": "median"})

    for date, row in weekly_raw_frequencies.iterrows():
        for variant in weekly_raw_frequencies.columns:
            raw_freq_value = row[variant]
            data.append({"location": location, "site": "weekly_raw_freq", "variant": variant, "date": date.strftime("%Y-%m-%d"), "value": raw_freq_value, "ps": "median"})

    return data



def run_estimation_for_df(
    df_subset: pd.DataFrame,
    location_label: str,
    args,
) -> dict:
    """
    Helper function: run preprocessing, pivot selection, model fitting,
    and result packaging for a single location (country).

    Returns a dict with all estimated parameters and metadata.
    """
    print(f"Estimating for location: {location_label}")

    data = preprocess_covid_data(df_subset, grouping_col=args.grouping_column)

    if args.pivot_variant:
        if args.pivot_variant not in set(data["variant_names"]):
            print(
                f"Pivot variant '{args.pivot_variant}' not found for location {location_label}. "
                "Choosing a pivot automatically."
            )
        else:
            # only reorder data if a pivot variant is specified
            data = choose_pivot_variant(data, pivot=args.pivot_variant)

    # build estimator (BFGS or stepwise)
    estimator = build_estimator(
        args.estimator,
        data["counts"],
        args.partition_size,
        args.overlap_size,
    )
    estimator.fit()
    results = estimator.get_results()

    # enrich results with metadata
    results["variant_names"] = data["variant_names"]
    results["dates"] = [
        d.strftime("%Y-%m-%d")
        if isinstance(d, (datetime.date, datetime.datetime))
        else str(d)
        for d in data["dates"]
    ]
    results["pivot_variant"] = data["variant_names"][0]

    # raw frequencies from counts
    df_daily_freqs, df_weekly_freqs = calculate_raw_frequencies(
        df_subset,
        args.grouping_column,
    )
    results["daily_raw_freqs"] = df_daily_freqs.to_json(orient="split")
    results["weekly_raw_freqs"] = df_weekly_freqs.to_json(orient="split")

    return results



def estimate_command(args):
    """
    Run the estimator and write a JSON file that ALWAYS has the shape:
    {
        "<country_or_location_name>": {...},
        "<another_location>": {...}
    }

    Even if there is only one location, it is wrapped under a key.
    """
    counts_df = load_counts(args.input_csv)

    # sanity check for grouping column
    if args.grouping_column not in counts_df.columns:
        raise ValueError(
            f"Column '{args.grouping_column}' not found in {args.input_csv}. "
            "Use --grouping-column to pick an existing column."
        )

    all_results = {}
    if args.location_column:
        if args.location_column not in counts_df.columns:
            raise ValueError(
                f"Location column '{args.location_column}' not found in {args.input_csv}."
            )

        for location_value in counts_df[args.location_column].unique():
            loc_df = counts_df[counts_df[args.location_column] == location_value]
            if not is_informative_dataset(loc_df, args.grouping_column):
                print(f"Skipping {location_value} (insufficient overlap or data).")
                continue
            results = run_estimation_for_df(loc_df, str(location_value), args)
            all_results[str(location_value)] = results
        
        # After looping, if we skipped everything → error
        if not all_results:
            raise ValueError(
                "No locations had sufficient variants, counts and/or overlap between variants."
                "Cannot perform estimation."
            )

    else:
        if not is_informative_dataset(counts_df, args.grouping_column):
            raise ValueError(
                "Dataset contains insufficient variants, counts or overlap between variants"
            )
        results = run_estimation_for_df(counts_df, "default", args)
        all_results["default"] = results

    # Write consistent {location: {...}} JSON
    output_path = Path(args.output)
    write_json(output_path, all_results)



def visualize_command(args):

    """
    Read the standardized estimates.json (which is ALWAYS {location: {...}, ...})
    and build a Nextstrain-style JSON blob.
    """

    with open(args.input_json, "r") as f:
        results_data = json.load(f)

    location_results = {
        loc: res
        for (loc, res) in results_data.items()
        if isinstance(res, dict) and "growth_rate_estimate" in res
    }

    if not location_results:
        raise ValueError(
            "No valid location entries found in input_json. "
            "Expected a dict {location: {... 'growth_rate_estimate' ...}}."
        )

    final_data_entries = []
    all_variants = set()
    all_dates = set()
    discovered_locations = set()

    for original_location_name, res in location_results.items():
        display_location = args.location_name or original_location_name
        discovered_locations.add(display_location)

        numeric_keys = [
            "growth_rate_estimate",
            "log_init_freq_estimate",
            "composition_estimate",
            "hessian",
            "hess_inv",
            "std_errors",
        ]
        for key in numeric_keys:
            if key in res:
                res[key] = np.array(res[key])

        dates = [pd.to_datetime(d) for d in res["dates"]]
        all_dates.update(dates)

        variant_names = np.array(res["variant_names"])
        all_variants.update(list(variant_names))

        df_daily_freqs = pd.read_json(io.StringIO(res["daily_raw_freqs"]), orient="split")
        df_weekly_freqs = pd.read_json(io.StringIO(res["weekly_raw_freqs"]), orient="split")

        location_data_entries = generate_nextstrain_data_entries(
            location=display_location,
            variant_names=variant_names,
            dates=dates,
            frequencies=res["composition_estimate"],
            growth_rates=res["growth_rate_estimate"],
            raw_frequencies=df_daily_freqs,
            weekly_raw_frequencies=df_weekly_freqs,
            std_errors=res["std_errors"],
        )

        final_data_entries.extend(location_data_entries)

    updated = pd.to_datetime("today").strftime("%Y-%m-%d")
    first_loc_key = next(iter(location_results.keys()))
    pivot_variant = location_results[first_loc_key].get("pivot_variant", None)

    sorted_variants = sorted(list(all_variants))
    sorted_dates = sorted(list(all_dates))
    sorted_locations = sorted(list(discovered_locations))

    metadata = {
        "ps": [
            "median",
            "HDI_50_upper",
            "HDI_50_lower",
            "HDI_80_upper",
            "HDI_80_lower",
            "HDI_95_upper",
            "HDI_95_lower",
        ],
        "sites": ["freq", "ga", "daily_raw_freq", "weekly_raw_freq"],
        "location": sorted_locations,
        "dates": [date.strftime("%Y-%m-%d") for date in sorted_dates],
        "variants": sorted_variants,
        "variantDisplayNames": [[name, name] for name in sorted_variants],
        "pivot": pivot_variant,
        "updated": updated,
        "variantColors": [
            [name, f"#{hash(name) % 0xFFFFFF:06x}"] for name in sorted_variants
        ],
    }

    final_json = {"metadata": metadata, "data": final_data_entries}

    output_path = Path(args.output)
    write_json(output_path, final_json)



def main() -> None:
    args = parse_args()
    if args.command == "estimate":
        estimate_command(args)
    elif args.command == "visualize":
        visualize_command(args)


if __name__ == "__main__":
    main()
