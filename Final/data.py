# data_fixed.py
import pandas as pd
import ast
import math

BIG_PUBLISHERS = {
    "Sony Interactive Entertainment",
    "Nintendo",
    "Ubisoft",
    "Electronic Arts",
    "EA",
    "Activision",
    "Bethesda Softworks",
    "Rockstar Games",
    "Take-Two Interactive",
    "Square Enix",
    "Capcom",
    "Bandai Namco",
}

def parse_list_column(val):
    """
    Robust parser for CSV cells that might contain:
      - a Python-style list string: "['A', 'B']"
      - a plain comma-separated string: "A, B"
      - a single value: "A"
      - empty / NaN -> []
    Returns a list of stripped strings.
    """
    # NaN / None
    if val is None:
        return []
    # Pandas NaN check
    if isinstance(val, float) and math.isnan(val):
        return []
    # If it's already a list
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip() != ""]
    # If it's not a string, coerce to str then continue
    if not isinstance(val, str):
        val = str(val)

    s = val.strip()
    if s == "":
        return []

    # If it looks like a python list literal, try ast.literal_eval
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip() != ""]
        except Exception:
            # fall through to comma-splitting
            pass

    # Remove surrounding quotes if the whole string is wrapped in quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # If the string contains commas, split on commas
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        # remove empty entries
        return [p for p in parts if p != ""]
    # Otherwise single token
    return [s] if s != "" else []


def estimate_studio_size(dev_game_count, publisher_name):
    publisher_name = publisher_name or ""
    try:
        dev_game_count = int(dev_game_count)
    except Exception:
        dev_game_count = 0

    if dev_game_count > 15 or publisher_name in BIG_PUBLISHERS:
        return "large"
    if 5 < dev_game_count <= 15:
        return "medium"
    return "small"


def estimate_funding(studio_size, has_mult):
    base = {"small": 1, "medium": 2, "large": 3}.get(studio_size, 1)
    if has_mult:
        base += 1
    if base <= 2:
        return "low"
    elif base == 3:
        return "medium"
    return "high"


def estimate_dev_time(studio_size, funding, game_scope):
    score = 0
    score += {"small": 3, "medium": 2, "large": 1}.get(studio_size, 2)
    score += {"low": 3, "medium": 2, "high": 1}.get(funding, 2)

    if game_scope > 20:
        score += 2
    elif game_scope > 10:
        score += 1

    if score <= 3:
        return "short"
    if score <= 5:
        return "medium"
    return "long"


def main(csv_path="rawg_games.csv", output_path="games.csv"):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)  # read everything as string to avoid surprises

    # Ensure columns exist
    for col in ["genres", "type", "developer", "publisher", "user_rating", "metacritic_score"]:
        if col not in df.columns:
            df[col] = None

    # Parse list-like columns robustly
    df["genres_parsed"] = df["genres"].apply(parse_list_column)
    df["type_parsed"] = df["type"].apply(parse_list_column)
    df["developer_parsed"] = df["developer"].apply(parse_list_column)

    # compute dev_game_count: how many times each developer appears across dataset
    dev_game_counts = {}
    for devs in df["developer_parsed"]:
        for dev in devs:
            dev = dev.strip()
            if dev == "":
                continue
            dev_game_counts[dev] = dev_game_counts.get(dev, 0) + 1

    def sum_dev_counts(devs):
        if not isinstance(devs, list):
            return 0
        s = sum(dev_game_counts.get(dev.strip(), 0) for dev in devs if dev.strip() != "")
        return int(s)

    df["dev_game_count"] = df["developer_parsed"].apply(sum_dev_counts)

    # Derived features
    # has_mult: check if any type token contains 'multiplayer' (case-insensitive)
    df["has_mult"] = df["type_parsed"].apply(
        lambda toks: any("multiplayer" in t.lower() for t in toks) if isinstance(toks, list) else False
    )

    # game_scope: number of tags + genres; since we may not have tags, use genres + type lengths
    df["game_scope"] = df.apply(
        lambda r: (len(r["genres_parsed"]) if isinstance(r["genres_parsed"], list) else 0)
                  + (len(r["type_parsed"]) if isinstance(r["type_parsed"], list) else 0),
        axis=1
    )

    # Compute heuristics
    df["studio_size"] = df.apply(
        lambda r: estimate_studio_size(r["dev_game_count"], r.get("publisher", "")),
        axis=1
    )

    df["funding_estimate"] = df.apply(
        lambda r: estimate_funding(r["studio_size"], r["has_mult"]),
        axis=1
    )

    df["dev_time_estimate"] = df.apply(
        lambda r: estimate_dev_time(r["studio_size"], r["funding_estimate"], r["game_scope"]),
        axis=1
    )

    # Optional: keep human-readable columns; convert parsed lists back to comma-strings for CSV
    df["genres_clean"] = df["genres_parsed"].apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
    df["type_clean"] = df["type_parsed"].apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
    df["developer_clean"] = df["developer_parsed"].apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")

    # Build output columns and order
    out_cols = [
        "title",
        "genres_clean", "type_clean", "developer_clean", "publisher",
        "user_rating", "metacritic_score",
        "dev_game_count", "has_mult", "game_scope",
        "studio_size", "funding_estimate", "dev_time_estimate"
    ]

    # Ensure all out_cols exist (create empty if not)
    for c in out_cols:
        if c not in df.columns:
            df[c] = ""

    df_out = df[out_cols].copy()
    # Rename cleaned columns back to original names if you prefer:
    df_out = df_out.rename(columns={
        "genres_clean": "genres",
        "type_clean": "type",
        "developer_clean": "developer"
    })

    print(f"Saving enhanced dataset to: {output_path}")
    df_out.to_csv(output_path, index=False)
    print("Done! Rows:", len(df_out))


if __name__ == "__main__":
    main()
