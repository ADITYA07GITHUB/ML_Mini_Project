"""
NetGuard AI - Realistic Synthetic Demo Dataset Generator
=========================================================
Generates synthetic network flow records conforming exactly to the CIC-IDS2017
schema (78 flow features + Label column).

NOTE: This dataset is mathematically generated for offline demonstration and unit testing when raw gigabyte-scale CIC-IDS2017 PCAP/CSV downloads
are not available or impractical to host. It simulates statistical characteristics of
normal flows and distinct cyber-attack patterns (DDoS, PortScan, DoS Hulk, Bot).
"""

import os
import argparse
import numpy as np
import pandas as pd

# Core CIC-IDS2017 column names (cleaned standard naming)
FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

def generate_records(num_records: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate balanced realistic network flow records matching CIC-IDS2017 features."""
    rng = np.random.default_rng(seed)
    
    # Proportions: 60% BENIGN, 15% DDoS, 12% PortScan, 8% DoS Hulk, 5% Bot
    proportions = {"BENIGN": 0.60, "DDoS": 0.15, "PortScan": 0.12, "DoS Hulk": 0.08, "Bot": 0.05}
    
    records = []
    
    for label, prop in proportions.items():
        count = int(num_records * prop)
        if count == 0:
            count = 1
            
        for _ in range(count):
            row = {}
            if label == "BENIGN":
                # Normal web/service traffic: HTTP (80), HTTPS (443), DNS (53), SSH (22)
                dst_port = rng.choice([80, 443, 53, 22, 8080, rng.integers(1024, 65535)])
                duration = rng.uniform(500, 500000)
                tot_fwd_pkts = rng.integers(2, 35)
                tot_bwd_pkts = rng.integers(2, 45)
                fwd_len_mean = rng.uniform(40, 600)
                bwd_len_mean = rng.uniform(60, 1200)
                fwd_len_max = fwd_len_mean * rng.uniform(1.2, 2.5)
                bwd_len_max = bwd_len_mean * rng.uniform(1.2, 2.5)
                syn_flags = 1 if rng.random() < 0.2 else 0
                fin_flags = 1 if rng.random() < 0.15 else 0
                ack_flags = 1
                psh_flags = 1 if rng.random() < 0.3 else 0
                flow_pkts_s = (tot_fwd_pkts + tot_bwd_pkts) / (duration / 1e6 + 1e-5)
                flow_bytes_s = (tot_fwd_pkts * fwd_len_mean + tot_bwd_pkts * bwd_len_mean) / (duration / 1e6 + 1e-5)
                iat_mean = duration / max(tot_fwd_pkts + tot_bwd_pkts, 1)
                down_up_ratio = round(tot_bwd_pkts / max(tot_fwd_pkts, 1), 2)
                idle_mean = rng.uniform(0, 1000)
                active_mean = rng.uniform(0, 500)
                
            elif label == "DDoS":
                # High volume flood, massive packet rate, short duration bursts
                dst_port = rng.choice([80, 443, 8080])
                duration = rng.uniform(20, 2000)
                tot_fwd_pkts = rng.integers(50, 400)
                tot_bwd_pkts = rng.integers(0, 5)  # asymmetric
                fwd_len_mean = rng.uniform(20, 120)
                bwd_len_mean = rng.uniform(0, 40)
                fwd_len_max = fwd_len_mean * 1.5
                bwd_len_max = bwd_len_mean * 1.2
                syn_flags = 1
                fin_flags = 0
                ack_flags = 0 if rng.random() < 0.7 else 1
                psh_flags = 0
                flow_pkts_s = rng.uniform(50000, 500000)
                flow_bytes_s = flow_pkts_s * fwd_len_mean
                iat_mean = rng.uniform(1, 20)
                down_up_ratio = 0.05
                idle_mean = 0.0
                active_mean = duration
                
            elif label == "PortScan":
                # Rapid probing across varied ports, single SYN packets, no responses
                dst_port = rng.integers(1, 65535)
                duration = rng.uniform(10, 1500)
                tot_fwd_pkts = rng.integers(1, 3)
                tot_bwd_pkts = 0  # Unanswered probes
                fwd_len_mean = rng.uniform(20, 60)
                bwd_len_mean = 0.0
                fwd_len_max = fwd_len_mean
                bwd_len_max = 0.0
                syn_flags = 1
                fin_flags = 0
                ack_flags = 0
                psh_flags = 0
                flow_pkts_s = rng.uniform(1000, 15000)
                flow_bytes_s = flow_pkts_s * 40
                iat_mean = rng.uniform(2, 50)
                down_up_ratio = 0.0
                idle_mean = 0.0
                active_mean = 0.0
                
            elif label == "DoS Hulk":
                # HTTP request flood with obfuscated user-agents and parameters
                dst_port = 80
                duration = rng.uniform(5000, 80000)
                tot_fwd_pkts = rng.integers(15, 80)
                tot_bwd_pkts = rng.integers(1, 10)
                fwd_len_mean = rng.uniform(250, 600)
                bwd_len_mean = rng.uniform(50, 200)
                fwd_len_max = fwd_len_mean * 1.4
                bwd_len_max = bwd_len_mean * 1.5
                syn_flags = 1 if rng.random() < 0.1 else 0
                fin_flags = 0
                ack_flags = 1
                psh_flags = 1
                flow_pkts_s = rng.uniform(1500, 12000)
                flow_bytes_s = flow_pkts_s * fwd_len_mean
                iat_mean = rng.uniform(50, 400)
                down_up_ratio = 0.1
                idle_mean = rng.uniform(0, 100)
                active_mean = duration
                
            else:  # Bot
                # C2 beaconing, periodic short bursts, high idle periods
                dst_port = rng.choice([6667, 8088, 4444, 80])
                duration = rng.uniform(100000, 1000000)
                tot_fwd_pkts = rng.integers(4, 15)
                tot_bwd_pkts = rng.integers(3, 12)
                fwd_len_mean = rng.uniform(80, 250)
                bwd_len_mean = rng.uniform(100, 400)
                fwd_len_max = fwd_len_mean * 1.8
                bwd_len_max = bwd_len_mean * 1.8
                syn_flags = 0
                fin_flags = 0
                ack_flags = 1
                psh_flags = 1 if rng.random() < 0.5 else 0
                flow_pkts_s = rng.uniform(10, 200)
                flow_bytes_s = rng.uniform(200, 5000)
                iat_mean = rng.uniform(10000, 80000)
                down_up_ratio = 0.8
                idle_mean = rng.uniform(50000, 400000)
                active_mean = rng.uniform(1000, 10000)

            # Populate row conforming to CIC-IDS2017 feature definitions
            row["Destination Port"] = int(dst_port)
            row["Flow Duration"] = max(1.0, float(duration))
            row["Total Fwd Packets"] = int(tot_fwd_pkts)
            row["Total Backward Packets"] = int(tot_bwd_pkts)
            row["Total Length of Fwd Packets"] = float(tot_fwd_pkts * fwd_len_mean)
            row["Total Length of Bwd Packets"] = float(tot_bwd_pkts * bwd_len_mean)
            row["Fwd Packet Length Max"] = float(fwd_len_max)
            row["Fwd Packet Length Min"] = float(max(0, fwd_len_mean * 0.4))
            row["Fwd Packet Length Mean"] = float(fwd_len_mean)
            row["Fwd Packet Length Std"] = float(fwd_len_mean * 0.3)
            row["Bwd Packet Length Max"] = float(bwd_len_max)
            row["Bwd Packet Length Min"] = float(max(0, bwd_len_mean * 0.4))
            row["Bwd Packet Length Mean"] = float(bwd_len_mean)
            row["Bwd Packet Length Std"] = float(bwd_len_mean * 0.3)
            row["Flow Bytes/s"] = float(flow_bytes_s)
            row["Flow Packets/s"] = float(flow_pkts_s)
            row["Flow IAT Mean"] = float(iat_mean)
            row["Flow IAT Std"] = float(iat_mean * 0.4)
            row["Flow IAT Max"] = float(iat_mean * 2.0)
            row["Flow IAT Min"] = float(max(0.1, iat_mean * 0.1))
            row["Fwd IAT Total"] = float(iat_mean * tot_fwd_pkts)
            row["Fwd IAT Mean"] = float(iat_mean)
            row["Fwd IAT Std"] = float(iat_mean * 0.3)
            row["Fwd IAT Max"] = float(iat_mean * 1.8)
            row["Fwd IAT Min"] = float(max(0.1, iat_mean * 0.1))
            row["Bwd IAT Total"] = float(iat_mean * max(tot_bwd_pkts, 1))
            row["Bwd IAT Mean"] = float(iat_mean if tot_bwd_pkts > 0 else 0.0)
            row["Bwd IAT Std"] = float(iat_mean * 0.2 if tot_bwd_pkts > 0 else 0.0)
            row["Bwd IAT Max"] = float(iat_mean * 1.5 if tot_bwd_pkts > 0 else 0.0)
            row["Bwd IAT Min"] = float(max(0.0, iat_mean * 0.1) if tot_bwd_pkts > 0 else 0.0)
            row["Fwd PSH Flags"] = int(psh_flags)
            row["Bwd PSH Flags"] = 0
            row["Fwd URG Flags"] = 0
            row["Bwd URG Flags"] = 0
            row["Fwd Header Length"] = int(tot_fwd_pkts * 20)
            row["Bwd Header Length"] = int(tot_bwd_pkts * 20)
            row["Fwd Packets/s"] = float(tot_fwd_pkts / (duration / 1e6 + 1e-5))
            row["Bwd Packets/s"] = float(tot_bwd_pkts / (duration / 1e6 + 1e-5))
            row["Min Packet Length"] = float(min(row["Fwd Packet Length Min"], row["Bwd Packet Length Min"]))
            row["Max Packet Length"] = float(max(row["Fwd Packet Length Max"], row["Bwd Packet Length Max"]))
            avg_size = (row["Total Length of Fwd Packets"] + row["Total Length of Bwd Packets"]) / max(tot_fwd_pkts + tot_bwd_pkts, 1)
            row["Packet Length Mean"] = float(avg_size)
            row["Packet Length Std"] = float(avg_size * 0.35)
            row["Packet Length Variance"] = float((avg_size * 0.35) ** 2)
            row["FIN Flag Count"] = int(fin_flags)
            row["SYN Flag Count"] = int(syn_flags)
            row["RST Flag Count"] = 0
            row["PSH Flag Count"] = int(psh_flags)
            row["ACK Flag Count"] = int(ack_flags)
            row["URG Flag Count"] = 0
            row["CWE Flag Count"] = 0
            row["ECE Flag Count"] = 0
            row["Down/Up Ratio"] = float(down_up_ratio)
            row["Average Packet Size"] = float(avg_size)
            row["Avg Fwd Segment Size"] = float(fwd_len_mean)
            row["Avg Bwd Segment Size"] = float(bwd_len_mean)
            row["Init_Win_bytes_forward"] = int(rng.choice([8192, 29200, 65535, 14600]))
            row["Init_Win_bytes_backward"] = int(rng.choice([8192, 29200, 0, 14600]) if tot_bwd_pkts > 0 else -1)
            row["act_data_pkt_fwd"] = int(max(0, tot_fwd_pkts - 1))
            row["min_seg_size_forward"] = 20
            row["Active Mean"] = float(active_mean)
            row["Active Std"] = float(active_mean * 0.1)
            row["Active Max"] = float(active_mean * 1.2)
            row["Active Min"] = float(active_mean * 0.8)
            row["Idle Mean"] = float(idle_mean)
            row["Idle Std"] = float(idle_mean * 0.1)
            row["Idle Max"] = float(idle_mean * 1.2)
            row["Idle Min"] = float(idle_mean * 0.8)
            row["Label"] = label
            
            records.append(row)

    df = pd.DataFrame(records)
    
    # Shuffle records
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Introduce real-world CIC-IDS2017 quirks for testing preprocessor:
    # 0.5% rows have string 'Infinity' or NaN in Flow Bytes/s or Flow Packets/s
    quirk_indices = rng.choice(len(df), size=max(5, int(len(df) * 0.005)), replace=False)
    for idx in quirk_indices[:len(quirk_indices)//2]:
        df.loc[idx, "Flow Bytes/s"] = np.nan
    for idx in quirk_indices[len(quirk_indices)//2:]:
        df.loc[idx, "Flow Packets/s"] = np.inf
        
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CIC-IDS2017 demonstration data.")
    parser.add_argument("--count", type=int, default=6000, help="Number of records to generate for training/evaluation demo")
    parser.add_argument("--out-dir", type=str, default="data/demo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Generate main training/testing demo dataset
    demo_path = os.path.join(args.out_dir, "cicids2017_demo.csv")
    print(f"Generating {args.count} synthetic CIC-IDS2017 records...")
    df = generate_records(num_records=args.count, seed=args.seed)
    df.to_csv(demo_path, index=False)
    print(f"Saved demo dataset to {demo_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f"Class distribution:\n{df['Label'].value_counts()}")
    
    # Generate small sample test set for UI upload demonstrations
    sample_path = os.path.join(args.out_dir, "cicids2017_sample_test.csv")
    sample_df = generate_records(num_records=250, seed=args.seed + 100)
    sample_df.to_csv(sample_path, index=False)
    print(f"Saved sample test dataset to {sample_path} ({sample_df.shape[0]} rows)")

if __name__ == "__main__":
    main()
