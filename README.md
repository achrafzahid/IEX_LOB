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

If you'd like, I can also:
- add a more detailed example showing how to run for a single date
- create a `requirements.txt` or `pyproject.toml`
- add automated tests for the transformer

