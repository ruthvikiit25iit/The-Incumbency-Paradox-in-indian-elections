from pathlib import Path
import pandas as pd

INPUT_FILE = Path("filtered_data/corrected/updated_general_elections.csv")

EXISTING_INCUMBENTS_FILE = Path("filtered_data/corrected/incumbents.csv")

OUTPUT_FILE = Path("filtered_data/corrected/non_incumbents.csv")


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "incumbent" not in df.columns:
        raise KeyError("The input CSV does not contain an 'incumbent' column.")

    incumbent_values = df["incumbent"].astype("string").str.strip().str.lower()
    incumbent_mask = incumbent_values.eq("true")
    non_incumbent_mask = incumbent_values.eq("false")

    non_incumbents = df.loc[non_incumbent_mask].drop(columns=["incumbent"]).copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    non_incumbents.to_csv(OUTPUT_FILE, index=False)

    incumbent_count = int(incumbent_mask.sum())
    unclassified_count = len(df) - incumbent_count - len(non_incumbents)


    if EXISTING_INCUMBENTS_FILE.is_file():
        existing_incumbent_count = len(pd.read_csv(EXISTING_INCUMBENTS_FILE))
        if existing_incumbent_count == incumbent_count:
            print("Consistency check: existing incumbents.csv row count matches.")
        else:
            print(
                "WARNING: existing incumbents.csv has "
                f"{existing_incumbent_count} rows, but updated_general_elections.csv "
                f"contains {incumbent_count} incumbent rows. Check that both files "
                "were generated from the same preprocessing run."
            )
    else:
        print("Note: incumbents.csv was not found, so the consistency check was skipped.")


if __name__ == "__main__":
    main()
