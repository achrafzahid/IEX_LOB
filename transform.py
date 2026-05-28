from collections import deque
import os
import glob
import gc
import pandas as pd
import numpy as np
import torch
from iex_cppparser import parse_dates

# --- Configurations ---
DOWNLOAD_DIR = "../pcap"
PARSED_DIR = "../parsed"
TENSORS_DIR = "../tensors"

# Ensure directories exist
for d in [DOWNLOAD_DIR, PARSED_DIR, TENSORS_DIR]:
    os.makedirs(d, exist_ok=True)


def return_csv_path(current_date :str) :
    search_pattern = os.path.join(PARSED_DIR, f"*{current_date.replace('-', '')}*_prl.csv")
    prl_files = glob.glob(search_pattern)
    if not prl_files:
        print(f"No parsed data found for {current_date}. Skipping.")
        return None
    print(f"csv file is {prl_files[0]}")
    return prl_files[0]


def cleanday(csv_path : str) :
    df = pd.read_csv(csv_path)

    df = df[df["Record Type"] == 'R']
    df = df[df["Event Flag"] == 1]
    df = df[df["Size"] != 0]
    df.drop(columns =["Event Flag","Record Type","Packet Capture Time","Send Time","Tick Type"],inplace=True)
    AAPL_df = df[df["Symbol"] == "AAPL"]
    AAPL_df.drop(columns=["Symbol"],inplace=True)
    NVDA_df = df[df["Symbol"] == "NVDA"]
    NVDA_df.drop(columns=["Symbol"],inplace=True)
    SPY_df = df[df["Symbol"] == "SPY"]
    SPY_df.drop(columns=["Symbol"],inplace=True)
    return {"AAPL":AAPL_df, "NVDA":NVDA_df, "SPY":SPY_df}



def build_and_save_deeplob_tensors(df : pd.DataFrame, ticker, date_str):
    """
    Fully automated LOB constructor, snapshotter, and normalizer for DeepLOB.
    Takes a raw dataframe for a single ticker and outputs a normalized .pt tensor.
    """
    print(f"[{ticker}] Starting processing for {date_str}...")

    # 1. CLEAN UP DATAFRAME
    # Strip spaces from column names to avoid " Buy_Ask Flag" errors
    df.columns = df.columns.str.strip()
    
    # Ensure it's sorted perfectly by time
    df["Timestamp"] = df["Exchange Timestamp"]
    df["Flag"]= df["Buy_Ask Flag"]  
    df.drop(columns=["Exchange Timestamp","Buy_Ask Flag"],inplace=True)
    df = df.sort_values(by="Timestamp")
    # Create an exact 1-second floor to trigger snapshots
    df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='ns', utc=True)
    df['Second'] = df['Datetime'].dt.floor('s')
    
    # 2. INITIALIZE LOB MEMORY AND WINDOW
    bids_book = {}
    asks_book = {}
    sliding_window = deque(maxlen=100)
    daily_matrices = []
    
    current_second = None
    
    # 3. TICK-BY-TICK ENGINE
    for row in df.itertuples():
        price = float(row.Price)
        volume = float(row.Size)
        # Handle variations in IEX column names safely
        if hasattr(row, 'Flag'):
            is_bid = (row.Flag == 1)
        else:
            raise ValueError("Could not find a valid Bid/Ask flag column.")
            
        # --- Time Jump & Forward-Fill Logic ---
        if current_second is None:
            current_second = row.Second
            
        # If the market advanced by 1 or more seconds, take snapshots to catch up
        while current_second < row.Second:
            # Extract Top 10 Bids (Descending) and Asks (Ascending)
            top_bids = sorted(bids_book.items(), key=lambda x: x[0], reverse=True)[:10]
            top_asks = sorted(asks_book.items(), key=lambda x: x[0])[:10]
            
            features = []
            for i in range(10):
                # Ask Price and Volume
                if i < len(top_asks):
                    features.extend([top_asks[i][0], top_asks[i][1]])
                else:
                    features.extend([0.0, 0.0]) # Pad empty levels
                    
                # Bid Price and Volume
                if i < len(top_bids):
                    features.extend([top_bids[i][0], top_bids[i][1]])
                else:
                    features.extend([0.0, 0.0]) # Pad empty levels
                    
            # Push snapshot to the 100-step rolling window
            sliding_window.append(features)
            
            # If the window is full, save it as a valid training matrix
            if len(sliding_window) == 100:
                daily_matrices.append(np.array(sliding_window))
                
            # Move clock forward by 1 second
            current_second += pd.Timedelta(seconds=1)
            
        # --- Update the Limit Order Book with current tick ---
        if is_bid:
            if volume == 0:
                bids_book.pop(price, None) # Order canceled or filled
            else:
                bids_book[price] = volume
        else:
            if volume == 0:
                asks_book.pop(price, None)
            else:
                asks_book[price] = volume

    # 4. PREPARE RAW TENSOR
    if not daily_matrices:
        print(f"[{ticker}] Warning: Not enough data to form a 100-second matrix.")
        return None
        
    raw_tensor = np.stack(daily_matrices) # Shape: (Total_Snapshots, 100, 40)
    normalized_tensor = np.zeros_like(raw_tensor)
    
    print(f"[{ticker}] Built {raw_tensor.shape[0]} matrices. Normalizing...")

    # 5. NORMALIZATION
    # Columns 0, 4, 8... are Ask Prices | Columns 2, 6, 10... are Bid Prices
    price_cols = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    vol_cols   = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]

    # Step A: Mid-Price Centering (Iterate over all generated matrices)
    for i in range(raw_tensor.shape[0]):
        matrix = raw_tensor[i]
        
        # Calculate Mid-Price = (Best Ask + Best Bid) / 2
        best_asks = matrix[:, 0]
        best_bids = matrix[:, 2]
        
        # Handle zero-padding edge cases
        mid_prices = np.where((best_asks > 0) & (best_bids > 0), 
                              (best_asks + best_bids) / 2.0, 
                              best_asks + best_bids)
        mid_prices = mid_prices.reshape(-1, 1)
        
        norm_matrix = np.copy(matrix)
        
        for c in price_cols:
            mask = norm_matrix[:, c] > 0 # Only subtract from actual prices, not zero-padded levels
            norm_matrix[mask, c] = norm_matrix[mask, c] - mid_prices[mask, 0]
            
        normalized_tensor[i] = norm_matrix

    # Step B: Volume Z-Score (Global across the entire day)
    for c in vol_cols:
        col_data = normalized_tensor[:, :, c]
        mean_v = np.mean(col_data)
        std_v = np.std(col_data)
        if std_v > 0:
            normalized_tensor[:, :, c] = (col_data - mean_v) / std_v
    return normalized_tensor
    
    
    