


import sys
from pathlib import Path
import pandas as pd



COLUMNS_TO_REMOVE = [
    "month",
    "sub_region",
    "candidate_sex",
    "candidate_type",
    "education_level",
    "poll_type",
    "deposit_lost"
]


def to_bool(series):


    values = series.astype(str).str.strip().str.lower()

    return values.map({
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False
    })


def clean_split_file(data):


    remove = [
        column for column in COLUMNS_TO_REMOVE
        if column in data.columns
    ]

    for column in data.columns:

        if "profession" in column.lower():
            remove.append(column)

    return data.drop(
        columns=list(set(remove)),
        errors="ignore"
    ).copy()


def same_seat(previous, current, seat_columns):

    if not seat_columns:
        return False

    for column in seat_columns:

        if pd.isna(previous[column]) or pd.isna(current[column]):
            return False

        if str(previous[column]).strip() != str(current[column]).strip():
            return False

    return True



def add_previous_win_history(data):



    required = [
        "candidate_id",
        "year",
        "position",
        "poll_type",
        "incumbent",
        "margin_percentage"
    ]

    for column in required:

        if column not in data.columns:
            raise ValueError("Required column missing: " + column)

    df = data.copy()


    df["_year"] = pd.to_numeric(
        df["year"], errors="coerce"
    )

    if "month" in df.columns:

        df["_month"] = pd.to_numeric(
            df["month"], errors="coerce"
        )

    else:

        df["_month"] = pd.NA

    df["_position"] = pd.to_numeric(
        df["position"], errors="coerce"
    )

    df["_poll_type"] = pd.to_numeric(
        df["poll_type"], errors="coerce"
    )

    df["_margin"] = pd.to_numeric(
        df["margin_percentage"], errors="coerce"
    )

    df["_incumbent"] = to_bool(df["incumbent"])


    seat_columns = [
        column
        for column in [
            "state_acronym",
            "constituency_number",
            "delimitation_id"
        ]
        if column in df.columns
    ]


    new_columns = [
        "last_win_year",
        "last_win_month",
        "last_win_poll_type",
        "last_win_type",
        "last_won_margin_percentage",
        "last_win_party",
        "last_win_constituency_name",
        "last_win_same_constituency",
        "last_win_match_status"
    ]

    for column in new_columns:
        df[column] = pd.NA


    candidate_history = {}

    for candidate_id, group in df[
        df["candidate_id"].notna()
    ].groupby("candidate_id"):

        candidate_history[candidate_id] = group


    incumbent_indices = df.index[
        df["_incumbent"] == True
    ]

    for index in incumbent_indices:

        current = df.loc[index]

        candidate_id = current["candidate_id"]

        current_year = current["_year"]

        current_month = current["_month"]


        if pd.isna(current_year):

            df.at[index, "last_win_match_status"] = (
                "MISSING_CURRENT_YEAR"
            )

            continue

        history = candidate_history.get(candidate_id)

        if history is None:

            df.at[index, "last_win_match_status"] = (
                "NO_CANDIDATE_HISTORY"
            )

            continue


        wins = history[
            history["_position"] == 1
        ].copy()


        earlier_year = (
            wins["_year"] < current_year
        )

        earlier_month = (
            (wins["_year"] == current_year)
            & wins["_month"].notna()
            & pd.notna(current_month)
            & (wins["_month"] < current_month)
        )

        wins = wins[
            earlier_year | earlier_month
        ].copy()

        if wins.empty:

            df.at[index, "last_win_match_status"] = (
                "NO_EARLIER_WIN_FOUND"
            )

            continue


        latest_year = wins["_year"].max()

        latest_year_wins = wins[
            wins["_year"] == latest_year
        ].copy()


        if (
            len(latest_year_wins) > 1
            and latest_year_wins["_month"].isna().any()
        ):

            df.at[index, "last_win_match_status"] = (
                "AMBIGUOUS_DATE_MONTH"
            )

            continue


        latest_month = latest_year_wins["_month"].max()

        if pd.notna(latest_month):

            latest_event = latest_year_wins[
                latest_year_wins["_month"] == latest_month
            ].copy()

        else:

            latest_event = latest_year_wins.copy()


        same_seat_indices = []

        for win_index, win_row in latest_event.iterrows():

            if same_seat(win_row, current, seat_columns):

                same_seat_indices.append(win_index)

        if len(same_seat_indices) == 1:

            chosen = latest_event.loc[
                same_seat_indices[0]
            ]

            ambiguous = False

        elif len(latest_event) == 1:

            chosen = latest_event.iloc[0]

            ambiguous = False

        else:


            chosen = latest_event.iloc[0]

            ambiguous = True


        df.at[index, "last_win_year"] = chosen["year"]

        df.at[index, "last_win_month"] = chosen.get(
            "month", pd.NA
        )

        df.at[index, "last_win_poll_type"] = chosen[
            "_poll_type"
        ]


        if chosen["_poll_type"] == 0:

            df.at[index, "last_win_type"] = (
                "GENERAL_ELECTION"
            )

        else:

            df.at[index, "last_win_type"] = (
                "BY_ELECTION"
            )

        if "party" in chosen.index:

            df.at[index, "last_win_party"] = (
                chosen["party"]
            )

        if "constituency_name" in chosen.index:

            df.at[index, "last_win_constituency_name"] = (
                chosen["constituency_name"]
            )

        df.at[index, "last_win_same_constituency"] = (
            same_seat(chosen, current, seat_columns)
        )


        if ambiguous:

            df.at[index, "last_win_match_status"] = (
                "AMBIGUOUS_MULTIPLE_WINS_SAME_EVENT"
            )

        elif pd.isna(chosen["_margin"]):

            df.at[index, "last_win_match_status"] = (
                "PRIOR_WIN_FOUND_BUT_MARGIN_MISSING"
            )

        else:

            df.at[index, "last_won_margin_percentage"] = (
                chosen["_margin"]
            )

            if pd.isna(current_month):

                df.at[index, "last_win_match_status"] = (
                    "MATCHED_MONTH_UNAVAILABLE"
                )

            else:

                df.at[index, "last_win_match_status"] = (
                    "MATCHED"
                )


    df = df.drop(
        columns=[
            "_year",
            "_month",
            "_position",
            "_poll_type",
            "_margin",
            "_incumbent"
        ]
    )

    return df



def main():


    input_file = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "srcs/general_elections.csv"
    )

    output_folder = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else "filtered_data/corrected"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    if not input_file.exists():

        raise FileNotFoundError(
            "Input file not found: " + str(input_file)
        )


    df = pd.read_csv(
        input_file,
        low_memory=False
    )




    if "deposit_lost" not in df.columns:
        raise ValueError("deposit_lost column is missing")

    deposit_lost_flag = to_bool(df["deposit_lost"])

    deposit_lost_rows_removed = int(
        (deposit_lost_flag == True).sum()
    )

    df = df[deposit_lost_flag != True].copy()


    if "poll_type" not in df.columns:

        raise ValueError(
            "poll_type column is missing"
        )

    poll_type = pd.to_numeric(
        df["poll_type"],
        errors="coerce"
    )

    general = df[
        poll_type == 0
    ].copy()

    other_polls = df[
        poll_type.notna()
        & (poll_type != 0)
    ].copy()

    unknown_polls = df[
        poll_type.isna()
    ].copy()


    clean_split_file(general).to_csv(
        output_folder / "general_elections.csv",
        index=False
    )


    clean_split_file(other_polls).to_csv(
        output_folder / "other_poll_types.csv",
        index=False
    )


    if not unknown_polls.empty:

        unknown_polls.to_csv(
            output_folder / "unknown_poll_type_rows.csv",
            index=False
        )



    enriched = add_previous_win_history(df)


    enriched = enriched.drop(
        columns=["deposit_lost"],
        errors="ignore"
    )

    incumbent_flag = to_bool(
        enriched["incumbent"]
    )

    all_incumbents = enriched[
        incumbent_flag == True
    ].copy()

    all_incumbents.to_csv(
        output_folder / "all_poll_type_incumbents_with_history.csv",
        index=False
    )

    enriched_poll_type = pd.to_numeric(
        enriched["poll_type"],
        errors="coerce"
    )

    updated_general = enriched[
        enriched_poll_type == 0
    ].copy()

    updated_general.to_csv(
        output_folder / "updated_general_elections.csv",
        index=False
    )


    general_incumbents = updated_general[
        to_bool(updated_general["incumbent"]) == True
    ].copy()

    general_incumbents.to_csv(
        output_folder / "incumbents.csv",
        index=False
    )

    complete_margin = general_incumbents[
        general_incumbents[
            "last_won_margin_percentage"
        ].notna()
    ].copy()

    complete_margin.to_csv(
        output_folder /
        "general_election_incumbents_with_last_win_margin.csv",
        index=False
    )

    unmatched = general_incumbents[
        general_incumbents[
            "last_won_margin_percentage"
        ].isna()
    ].copy()

    unmatched.to_csv(
        output_folder /
        "general_election_incumbents_unmatched_history.csv",
        index=False
    )


    report = []

    report.append(
        "INCUMBENCY PROJECT - DATA REBUILD REPORT"
    )

    report.append("=" * 55)

    report.append(
        "Rows remaining after deposit_lost filter: " + str(len(df))
    )

    report.append(
        "Rows removed because deposit_lost == True: "
        + str(deposit_lost_rows_removed)
    )

    report.append(
        "General election rows after deposit filter: " + str(len(general))
    )

    report.append(
        "Other poll type rows: " + str(len(other_polls))
    )

    report.append(
        "Unknown poll type rows: " + str(len(unknown_polls))
    )

    report.append("")

    report.append("INCUMBENT DATASETS")

    report.append(
        "All poll-type incumbents: " + str(len(all_incumbents))
    )

    report.append(
        "General-election incumbents: " + str(len(general_incumbents))
    )

    report.append(
        "With known previous-win margin: " + str(len(complete_margin))
    )

    report.append(
        "Without known previous-win margin: " + str(len(unmatched))
    )

    report.append("")

    report.append("LAST WIN TYPE")

    report.append(
        general_incumbents[
            "last_win_type"
        ].value_counts(dropna=False).to_string()
    )

    report.append("")

    report.append("MATCH STATUS")

    report.append(
        general_incumbents[
            "last_win_match_status"
        ].value_counts(dropna=False).to_string()
    )

    report.append("")

    report.append(
        "Missing margins in complete file: "
        + str(
            complete_margin[
                "last_won_margin_percentage"
            ].isna().sum()
        )
    )

    report.append("")

    report.append("IMPORTANT NOTES")

    report.append(
        "1. Historical matching uses all poll types."
    )

    report.append(
        "2. Main analysis contains general elections only."
    )

    report.append(
        "3. last_win_type describes the previous victory."
    )

    report.append(
        "4. A by-election winner can be a genuine incumbent "
        "at a later general election."
    )

    report.append(
        "5. Missing margins are never replaced with zero."
    )

    report.append(
        "6. Rows with deposit_lost == True are removed from all poll types "
        "before splitting and historical matching. The deposit_lost column "
        "is removed from enriched output files."
    )

    report.append(
        "7. The complete-margin file is a selected subset. "
        "Unmatched rows are preserved separately."
    )

    report_path = (
        output_folder / "historical_matching_report.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )




if __name__ == "__main__":
    main()
