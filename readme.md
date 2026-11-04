# The Incumbency Paradox in Indian Elections

This project analyzes Indian state election results to compare incumbent and non-incumbent candidates. Its scripts prepare election data and study win rates, victory margins, previous performance, and close elections.

## Setup

Run these commands from the repository folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use `.\.venv\Scripts\python.exe` to run the commands below. Most analyses read `data_files/extended.csv`; scripts that accept an input file are noted.

## Scripts

### Prepare data

```powershell
.\.venv\Scripts\python.exe scripts/extended.py input.csv output.csv
.\.venv\Scripts\python.exe scripts/updated_dataset.py input.csv output.csv
```

- `extended.py` keeps regular elections and adds signed margins, candidate contest history, and previous party vote share.
- `updated_dataset.py` filters regular elections and adds signed victory margins.

### Analyze election results

```powershell
.\.venv\Scripts\python.exe scripts/stage1.py
.\.venv\Scripts\python.exe scripts/stage1.py input.csv
.\.venv\Scripts\python.exe scripts/victory-margin.py
.\.venv\Scripts\python.exe scripts/year-wise.py input.csv
.\.venv\Scripts\python.exe scripts/close_election_analysis.py
```

- `stage1.py` compares incumbent and non-incumbent win rates with confidence intervals and a chart. Defaults to `data_files/extended.csv`; an input CSV can be supplied.
- `victory-margin.py` summarizes winners' victory margins by incumbency status. Defaults to `data_files/extended.csv` and creates charts.
- `year-wise.py` calculates yearly win rates and their difference; requires an input CSV.
- `close_election_analysis.py` compares subsequent outcomes for candidates who narrowly won or lost; writes its numerical report to standard output and creates charts.

### Regression analyses

```powershell
.\.venv\Scripts\python.exe scripts/simple_regression.py input.csv
.\.venv\Scripts\python.exe scripts/previous_margin_regression.py
.\.venv\Scripts\python.exe scripts/multiple_regression.py
.\.venv\Scripts\python.exe scripts/regression_diagnostic.py
```

- `simple_regression.py` relates previous signed margins to current margins; requires an input CSV.
- `previous_margin_regression.py` estimates the relationship between a candidate's previous and subsequent signed margins.
- `multiple_regression.py` models the subsequent signed margin using incumbency, previous candidate margin, and previous party vote share. Numerical results print to standard output.
- `regression_diagnostic.py` creates a residuals-versus-fitted plot for the multiple regression.

Analysis charts are saved in `visualisations/`; some scripts also write CSV outputs under `data_files/`.
