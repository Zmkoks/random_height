from typing import Optional, List, Tuple, Dict, Union
import pandas as pd
import numpy as np
import random
from difflib import get_close_matches
from scipy.interpolate import interp1d
from scipy.stats import linregress
import ast
import unicodedata


# Load datasets (assume already loaded outside this snippet)
# height_chart_df, all_countries_df, non_countries_df

# Paths to your CSV files
path_all_countries = "all_countries_nation_height.csv"
path_non_countries = "non_countries.csv"
path_height_chart = "height_chart_WHO_CDC.csv"

# Load all_countries_nation_height.csv
all_countries_df = pd.read_csv(path_all_countries, encoding="ISO-8859-1")

# Load non_countries.csv (broader regional terms, urban/rural distinctions)
non_countries_df = pd.read_csv(path_non_countries, encoding="ISO-8859-1")


# Load WHO/CDC percentile chart
height_chart_df = pd.read_csv(path_height_chart)

# Constants
BRITISH_AVG_MALE = 177
BRITISH_AVG_FEMALE = 164

SIZE_CLASS_TO_PERCENTILE_COL = {
    "XXS": "P3", "XS": "P5", "S": "P10", "SM": "P25", "M": "P50",
    "ML": "P75", "LM": "P90", "L": "P90", "XL": "P95", "XXL": "P97"
}

PERCENTILE_COLUMNS = [col for col in height_chart_df.columns if col.startswith("P")]
PERCENTILE_VALUES = [int(col[1:]) for col in PERCENTILE_COLUMNS]

# Height curve generator (interpolated or extrapolated)
def get_virtual_percentile_curve(sex: str, target_percentile: float) -> Dict[int, float]:
    """
    Returns a dictionary of estimated heights (in cm) from birth to 20 years
    at a given growth percentile, based on WHO/CDC data.

    ✅ What it does:
    - Uses real WHO/CDC child growth reference data
    - Handles both interpolation (normal range) and extrapolation (extreme short/tall)
    - Returns height in cm at each standard milestone: birth, 3mo, 6mo, ..., 240mo (20 years)

    📥 Parameters:
    - sex (str): "male" or "female"
    - target_percentile (float): the desired growth percentile (e.g. 25, 50, 85)

    📤 Returns:
    - Dictionary: {age_in_months: height_in_cm}
    """

    # Convert to WHO/CDC dataset's internal coding
    sex_code = 1 if sex.lower() == "male" else 2

    # Filter WHO/CDC dataset for this sex
    df = height_chart_df[height_chart_df["Sex"] == sex_code]

    curve = {}  # Final output

    for _, row in df.iterrows():
        agemos = int(row["Agemos"])  # Age in months
        heights = [row[col] for col in PERCENTILE_COLUMNS]  # Extract height values by percentile

        # Interpolate if within WHO's known percentiles
        if min(PERCENTILE_VALUES) <= target_percentile <= max(PERCENTILE_VALUES):
            interp_func = interp1d(PERCENTILE_VALUES, heights, kind='linear')
            height = float(interp_func(target_percentile))

        # Extrapolate for extreme values below P3 or above P97
        elif target_percentile < min(PERCENTILE_VALUES):
            slope, intercept, *_ = linregress(PERCENTILE_VALUES[:2], heights[:2])
            height = slope * target_percentile + intercept

        elif target_percentile > max(PERCENTILE_VALUES):
            slope, intercept, *_ = linregress(PERCENTILE_VALUES[-2:], heights[-2:])
            height = slope * target_percentile + intercept

        # Round and store result
        curve[agemos] = round(height, 2)

    return curve

# Height modifiers
def apply_growth_modifiers(
    base_heights: Dict[str, float],
    dips: List[Tuple[float, float, str]] = [],
    spurts: List[Tuple[float, float, str]] = [],
    drift: bool = True
) -> Dict[str, float]:
    """
    Applies dips, spurts, and optional drift to the base height curve.

    ✅ What it does:
    - Subtracts from height during dip windows (e.g., illness, trauma)
    - Adds to height during spurt windows (e.g., recovery, puberty)
    - Applies small, normally distributed drift for realism
    - Returns a modified dictionary of {label: height}

    📥 Parameters:
    - base_heights: dict with age labels ('birth', '3 mo', '1 year', etc.) and base heights
    - dips: list of (start_age, end_age, label) events that reduce height
    - spurts: list of (start_age, end_age, label) events that boost height
    - drift: whether to apply subtle random variation (default: True)

    🧪 Notes:
    - Dips apply ~–0.3cm during affected range
    - Spurts apply ~+0.5cm
    - Drift is Gaussian noise: mean 0, std dev 0.2cm

    📤 Output:
    - A dict like the input, but with modified heights
    """

    result = {}

    for label, height in base_heights.items():
        if label == "Name":
            result[label] = height
            continue

        # --- Convert label to decimal age ---
        if label == "birth":
            age = 0.0
        elif "mo" in label:
            age = float(label.split()[0])
        else:
            age = float(label.split()[0])

        # --- Apply dip/spurt effects ---
        mod = 0.0

        for start, end, _ in dips:
            if start <= age <= end:
                mod -= 0.3

        for start, end, _ in spurts:
            if start <= age <= end:
                mod += 0.5

        # --- Apply drift ---
        if drift:
            mod += np.random.normal(0, 0.2)

        result[label] = round(height + mod, 1)

    return result

# Nationality percentile shift

def normalise_name(name: str) -> str:
    """
    Standardises strings by removing special dashes and accents,
    converting to lowercase, and stripping whitespace.
    """
    return unicodedata.normalize("NFKD", str(name)).replace("–", "-").replace("—", "-").replace("−", "-").lower().strip()

def get_nationality_percentile_shift(nationality: str, sex: str) -> Union[float, str]:
    """
    Looks up the average height for a nationality and returns a percentile shift
    (positive or negative) compared to the British average.

    ✅ What it does:
    - Matches input nationality to known country names, nationality terms, or codes
    - Compares that country's average height to the British average
    - Converts the cm difference into a percentile shift for the WHO/CDC chart

    📥 Parameters:
    - nationality (str): e.g. "Dutch", "Japan", "NL", "FRA"
    - sex (str): "male" or "female"

    📤 Returns:
    - A float indicating the number of percentiles to shift
    - OR a string error message suggesting close matches if nationality is invalid
    """

    sex = sex.lower()
    column = "female_avg" if sex == "female" else "male_avg"
    matched_row = None
    nat_lower = normalise_name(nationality)

    # --- Match in all_countries_df by code, name, or nationality ---
    for df in [
        all_countries_df[all_countries_df["alpha_2"].str.lower() == nat_lower],
        all_countries_df[all_countries_df["alpha_3"].str.lower() == nat_lower],
        all_countries_df[all_countries_df["country"].apply(lambda x: normalise_name(x) == nat_lower)],
        all_countries_df[all_countries_df["nationality"].fillna("").str.lower().str.contains(nat_lower)],
    ]:
        if not df.empty:
            matched_row = df.iloc[0]
            break

    # --- Fallback to non_countries_df full match ---
    if matched_row is None:
        match = non_countries_df[non_countries_df["country"].apply(lambda x: normalise_name(x) == nat_lower)]
        if not match.empty:
            matched_row = match.iloc[0]

    # --- Partial match in non_countries_df ---
    if matched_row is None:
        match = non_countries_df[non_countries_df["country"].apply(lambda x: nat_lower in normalise_name(x))]
        if not match.empty:
            matched_row = match.iloc[0]
        else:
            # Suggest possible matches
            names = (
                non_countries_df["country"].tolist()
                + all_countries_df["country"].dropna().tolist()
                + all_countries_df["nationality"].dropna().str.split(".").explode().tolist()
            )
            suggestions = get_close_matches(nationality, names, n=3, cutoff=0.6)
            return f"This nationality doesn't exist in our database. Did you mean: {', '.join(suggestions)}?"

    # --- Calculate shift ---
    avg_height = matched_row[column]
    british_avg = BRITISH_AVG_FEMALE if sex == "female" else BRITISH_AVG_MALE
    cm_diff = avg_height - british_avg

    shift = cm_diff / (0.75 if sex == "female" else 0.9)  # Rough cm-per-percentile approximation
    return round(shift, 1)


# Random dip/spurt generator
def generate_random_modifiers(
    add_random_dips: bool = True,
    add_random_spurts: bool = True,
    dip_count_range: Tuple[int, int] = (1, 2),
    spurt_count_range: Tuple[int, int] = (1, 2),
    dip_labels: List[str] = None,
    spurt_labels: List[str] = None,
    min_age: float = 1.0,
    max_age: float = 18.0
) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
    """
    Randomly generates dips and spurts to apply to a character's growth timeline.

    ✅ What it does:
    - Simulates unexpected life events that influence growth.
    - Dips: illness, stress, trauma — cause temporary height reduction.
    - Spurts: recovery, environment change, maturation — cause temporary height boosts.
    - Returns both as lists of (start_age, end_age, description).

    📥 Parameters:
    - add_random_dips: whether to generate dips at all
    - add_random_spurts: whether to generate spurts at all
    - dip_count_range: how many dips to create (e.g., (1, 2))
    - spurt_count_range: how many spurts to create (e.g., (1, 2))
    - dip_labels: list of custom dip descriptions (optional)
    - spurt_labels: list of custom spurt descriptions (optional)
    - min_age: youngest possible event onset (in years)
    - max_age: oldest possible event onset (in years)

    📤 Returns:
    - A tuple (dips, spurts), each a list of (start_age, end_age, label)
    """

    # Default reason pools
    dip_labels = dip_labels or [
        "illness", "malnourished", "stress", "loss", "injury", "neglect", "bullying", "abuse"
    ]

    spurt_labels = spurt_labels or [
        "growth spurt", "recovery", "relocation", "surge", "matured", "stable home", "good nutrition"
    ]

    dips = []
    spurts = []

    # --- Dips ---
    if add_random_dips:
        for _ in range(random.randint(*dip_count_range)):
            start = round(random.uniform(min_age, max_age - 0.5), 1)
            duration = round(random.uniform(0.3, 1.2), 1)
            end = round(start + duration, 1)
            label = random.choice(dip_labels)
            dips.append((start, end, label))

    # --- Spurts ---
    if add_random_spurts:
        for _ in range(random.randint(*spurt_count_range)):
            start = round(random.uniform(min_age, max_age - 0.5), 1)
            duration = round(random.uniform(0.3, 1.5), 1)
            end = round(start + duration, 1)
            label = random.choice(spurt_labels)
            spurts.append((start, end, label))

    return dips, spurts

# Main generator
def generate_growth_with_nationality(
    name: str,
    sex: str,
    size_class: str,
    nationality: Optional[str] = "British",
    dips: Optional[List[Tuple[float, float, str]]] = None,
    spurts: Optional[List[Tuple[float, float, str]]] = None,
    drift: bool = True,
    bloom_style: Optional[str] = None,
    birth_timing: Optional[str] = None,
    add_random_dips: bool = False,
    add_random_spurts: bool = False
) -> Union[Dict[str, float], str]:
    """
    Creates a detailed growth chart for a single character.

    ✅ What it does:
    - Looks up the height percentile curve (birth to 20 yrs)
    - Adjusts based on nationality (+/- percentile shift)
    - Inserts birth-related and puberty-related growth modifiers
    - Optionally includes life-event dips and spurts
    - Adds subtle drift for realism
    - Outputs a dict with height at each stage and a comment summary

    📥 Inputs:
    - name: character’s name
    - sex: 'male' or 'female'
    - size_class: XXS–XXL (mapped to base percentile)
    - nationality: country/nationality or code (default = British)
    - dips / spurts: list of (start_age, end_age, label) or None
    - drift: apply small random variation to heights (default: True)
    - bloom_style: puberty type ('early', 'standard', 'late') or None
    - birth_timing: 'premature', 'normal', 'late' or None
    - add_random_dips/spurts: if True, appends new random events

    📤 Output:
    - Dict with height per age label and a 'comment' with modifiers
    """

    # Initialise dips/spurts if empty
    dips = dips or []
    spurts = spurts or []

    # Default: British base
    resolved_nation = "British"
    alpha_code = "GB"
    shift = 0.0

    # --- Nationality & percentile shift ---
    if nationality and nationality.lower() != "british":
        shift_result = get_nationality_percentile_shift(nationality, sex)
        if isinstance(shift_result, str):  # Error string
            return shift_result
        shift = shift_result

        # Match country and code for summary
        nat_lower = nationality.lower()
        for df in [
            all_countries_df[all_countries_df["alpha_2"].str.lower() == nat_lower],
            all_countries_df[all_countries_df["alpha_3"].str.lower() == nat_lower],
            all_countries_df[all_countries_df["country"].str.lower() == nat_lower],
            all_countries_df[all_countries_df["nationality"].fillna("").str.lower().str.contains(nat_lower)],
        ]:
            if not df.empty:
                resolved_nation = df.iloc[0]["country"]
                alpha_code = df.iloc[0]["alpha_2"]
                break

    # --- Add random life events if requested ---
    if add_random_dips or add_random_spurts:
        extra_dips, extra_spurts = generate_random_modifiers(
            add_random_dips=add_random_dips,
            add_random_spurts=add_random_spurts
        )
        dips.extend(extra_dips)
        spurts.extend(extra_spurts)

    # --- Get percentile curve ---
    base_percentile = int(SIZE_CLASS_TO_PERCENTILE_COL[size_class.upper()][1:])
    final_percentile = max(1, min(99, base_percentile + shift))  # Keep within bounds
    virtual_curve = get_virtual_percentile_curve(sex, final_percentile)

    # --- Format growth data by label ---
    growth_dict = {"Name": name}
    for agemos, height in virtual_curve.items():
        label = (
            "birth" if agemos == 0 else
            f"{int(agemos / 12)} year" if agemos % 12 == 0 else
            f"{int(agemos / 12)}.{int((agemos % 12) // 3) * 3} mo"
        )
        growth_dict[label] = height

    # --- Puberty spurt based on style ---
    if bloom_style is None:
        bloom_style = random.choices(["early", "standard", "late"], weights=[0.2, 0.6, 0.2])[0]

    if bloom_style == "early":
        spurts.append((10, 11.5, "puberty (early)"))
    elif bloom_style == "standard":
        spurts.append((12, 13.5, "puberty"))
    elif bloom_style == "late":
        spurts.append((14, 15.5, "puberty (late)"))

    # --- Birth impact (dip or spurt) ---
    if birth_timing is None:
        birth_timing = random.choices(["premature", "normal", "late"], weights=[0.1, 0.8, 0.1])[0]

    if birth_timing == "premature":
        dips.append((0, 0.75, "premature"))
    elif birth_timing == "late":
        spurts.append((0, 0.25, "late birth spurt"))

    # --- Apply modifiers ---
    result = apply_growth_modifiers(growth_dict, dips=dips, spurts=spurts, drift=drift)

    # --- Generate comment string ---
    dip_str = "; ".join([f"{s}-{e} ({lbl})" for s, e, lbl in dips]) if dips else "None"
    spurt_str = "; ".join([f"{s}-{e} ({lbl})" for s, e, lbl in spurts]) if spurts else "None"

    result["comment"] = (
        f"Nationality: {resolved_nation} ({alpha_code}), shift: {shift:+.1f} percentiles | "
        f"Dips: {dip_str} | Spurts: {spurt_str}"
    )

    return result


import csv

def detect_csv_delimiter(file_path: str, sample_lines: int = 5) -> str:
    """
    Detects whether a CSV file uses a comma or semicolon delimiter.
    Defaults to comma if unsure.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = ''.join([next(f) for _ in range(sample_lines)])
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter

def read_character_input_csv_with_validation(path: str) -> pd.DataFrame:
    """
    Reads and validates the character input CSV with automatic delimiter detection.
    """

    delimiter = detect_csv_delimiter(path)

    # Read the file using detected delimiter
    df = pd.read_csv(path, sep=delimiter)

    # Valid values
    valid_sexes = {"male", "female"}
    valid_size_classes = {"XXS", "XS", "S", "SM", "M", "ML", "LM", "L", "XL", "XXL"}
    valid_birth_styles = {"premature", "normal", "late"}
    valid_growth_styles = {"early", "standard", "late"}

    errors = []

    # Normalise
    df["sex"] = df["sex"].str.lower().fillna("female")
    df["size_class"] = df["size_class"].str.upper().fillna("M")

    for col in ["birth", "growth_style", "nation"]:
        df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) and x.strip().lower() != "none" else None)

    # Parse dip
    df["dip"] = df["dip"].apply(lambda val: ast.literal_eval(val) if pd.notna(val) and val != "None" else None)

    # Convert boolean flags
    df["add_random_dips"] = df["add_random_dips"].apply(lambda x: str(x).strip().lower() == "true")
    df["add_random_spurts"] = df["add_random_spurts"].apply(lambda x: str(x).strip().lower() == "true")

    for idx, row in df.iterrows():
        name = row.get("name", f"<row {idx+1}>")
        if row["sex"] not in valid_sexes:
            errors.append(f"{name}: Invalid sex '{row['sex']}'")
        if row["size_class"] not in valid_size_classes:
            errors.append(f"{name}: Invalid size_class '{row['size_class']}'")
        if row["birth"] is not None and row["birth"] not in valid_birth_styles:
            errors.append(f"{name}: Invalid birth '{row['birth']}'")
        if row["growth_style"] is not None and row["growth_style"] not in valid_growth_styles:
            errors.append(f"{name}: Invalid growth_style '{row['growth_style']}'")

    if errors:
        print("⚠️ Validation Errors Found:")
        for e in errors:
            print(" -", e)

    return df


def main(
    input_path: str = "/mnt/data/character_growth_template.csv",
    output_path: str = "/mnt/data/character_growth_output.csv"
):
    """
    Main function to run the character growth generator pipeline.

    ✅ What it does:
    - Loads character metadata from a CSV
    - Validates and normalises inputs
    - Generates growth curves with randomised or defined modifiers
    - Saves the output as a single flat CSV file

    📥 Inputs:
    - input_path: path to character input CSV (template format)
    - output_path: where to save the final growth data

    📤 Output:
    - A CSV with:
        - Name
        - Height at each key age
        - A 'comment' field explaining modifiers

    🛑 Any invalid characters are skipped, but warnings are printed.
    """

    df = read_character_input_csv_with_validation(input_path)

    all_outputs = []

    for idx, row in df.iterrows():
        try:
            result = generate_growth_with_nationality(
                name=row["name"],
                sex=row["sex"],
                size_class=row["size_class"],
                nationality=row.get("nation", "British"),
                dips=row.get("dip", None),
                bloom_style=row.get("growth_style", None),
                birth_timing=row.get("birth", None),
                add_random_dips=row.get("add_random_dips", False),
                add_random_spurts=row.get("add_random_spurts", False),
                spurts=None,  # can be expanded later
                drift=True
            )

            if isinstance(result, str):  # if result is an error message
                print(f"⚠️ Skipped {row['name']}: {result}")
                continue

            all_outputs.append(result)

        except Exception as e:
            print(f"❌ Error processing {row.get('name', f'row {idx+1}')}: {e}")

    # Export final output
    if all_outputs:
        output_df = pd.DataFrame(all_outputs)
        output_df.to_csv(output_path, index=False)
        print(f"✅ Exported {len(all_outputs)} characters to: {output_path}")
    else:
        print("⚠️ No valid characters were processed. Nothing to export.")
