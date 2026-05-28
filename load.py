import torch
import os
import glob
TICKER = "SPY"
DOWNLOAD_DIR = "../pcap"
PARSED_DIR = "../parsed"
TENSORS_DIR = "../tensors"


def load_tensor(normalized_tensor, ticker, date_str) :
        # 6. SAVE TO DISK
    final_pt = torch.tensor(normalized_tensor, dtype=torch.float32)
    
    save_dir = os.path.join(TENSORS_DIR, ticker)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{ticker}_{date_str}.pt")
    
    torch.save(final_pt, file_path)
    print(f"[{ticker}] SUCCESS: Saved tensor of shape {tuple(final_pt.shape)} to {file_path}")
    
def remove_csv(csv_path) :
    os.remove(csv_path)
def remove_pcap() :
    pcap_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.pcap*"))
    for file_path in pcap_files:
        try:
            os.remove(file_path)
            print(f"Successfully deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")