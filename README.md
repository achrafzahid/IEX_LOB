# IEX_LOB

IEX_LOB is a small ETL pipeline that downloads IEX market data, parses it into per-tick CSVs,
constructs second-by-second Limit Order Book (LOB) snapshots, normalizes them, and saves
the resulting tensors for machine learning workloads (DeepLOB-style).

## Project Structure

- `extract.py` — Downloads and parses raw IEX pcap data into parsed CSVs using `iex_cppparser`.
- `transform.py` — Cleans parsed CSVs, constructs 100-second rolling LOB snapshots for each ticker,
	normalizes prices and volumes, and returns NumPy tensors ready for saving.
- `load.py` — Converts normalized tensors to PyTorch tensors and saves them as `.pt` files.
- `main.py` — Example runner that sequences extraction, transformation, and saving across dates.

## Directories and Outputs

- `../pcap` — raw downloaded pcap files (created/used by `extract.py`).
- `../parsed` — intermediate parsed CSV files (output from parser).
- `../tensors` — final normalized tensors saved per ticker and date as `TICKER_YYYY-MM-DD.pt`.

## Dependencies

- Python 3.8+
- pandas, numpy, torch
- iex_cppparser (project uses `parse_dates` and `compile_cpp` from this package)

Install typical dependencies with pip, e.g.:

```bash
pip install pandas numpy torch
# iex_cppparser is a project-specific parser — install or build as required.
```

## Usage

Run the example pipeline (will attempt to download data, parse, transform, and save tensors):

```bash
python3 main.py
```

`main.py` shows the intended flow: for each date it calls `extract_day()` to ensure parsed CSVs
exist, uses `cleanday()` to filter records and split by ticker, `build_and_save_deeplob_tensors()`
to build normalized LOB tensors, then `load_tensor()` to persist them.

## Notes and Recommendations

- The parser supports a `split=True` mode (see `extract.py` comments) which is recommended on
	low-memory machines to avoid large in-memory allocations when parsing many pcaps.
- `transform.py` builds 100-second windows of top-10 asks and bids (40 features per timestep).
	Prices are mid-price centered and volumes are z-score normalized per-day.
- The code expects specific column names from the parser (e.g., `Exchange Timestamp`, `Buy_Ask Flag`).
	If your parser version uses different names, adjust `transform.py` accordingly.

## Single-date Example

To run the pipeline for a single date (no loop), you can call the functions directly. Example script:

```python
from extract import extract_day
from transform import return_csv_path, cleanday, build_and_save_deeplob_tensors
from load import load_tensor, remove_csv, remove_pcap

date = "2024-08-05"
extract_day(date)                  # ensure parsed CSV exists for the date
csv_path = return_csv_path(date)
day_dict = cleanday(csv_path)      # get per-ticker DataFrames
for ticker, df in day_dict.items():
		tensor = build_and_save_deeplob_tensors(df=df, ticker=ticker, date_str=date)
		if tensor is not None:
				load_tensor(normalized_tensor=tensor, ticker=ticker, date_str=date)
remove_csv(csv_path)
remove_pcap()
```

Save that as `run_single.py` and run `python3 run_single.py`.

## Adapting to Different Input Data

- Column names: If your parsed CSV uses different column names, update `transform.py` to map
	the parser's columns to the expected names (e.g., `Exchange Timestamp`, `Buy_Ask Flag`,
	`Price`, `Size`, `Symbol`).
- Different tickers: `cleanday()` filters for `AAPL`, `NVDA`, and `SPY` as examples. Change or
	extend this list to process additional tickers.
- Window length & depth: `transform.py` builds 100-second windows of top-10 levels. To change
	the snapshot horizon or depth, modify the `deque(maxlen=100)` and the top-N logic.
- Normalization: Current normalization centers prices by mid-price and z-scores volumes per day.
	If you prefer log-returns, min-max scaling, or per-snapshot normalization, edit the
	normalization section in `build_and_save_deeplob_tensors()`.
- Missing data: The transformer pads missing levels with zeros. If you have alternative
	padding or imputation strategies, implement them before snapshotting.

