# NetGuard AI - Data Directory

This directory stores datasets used for training, evaluating, and demonstrating the NetGuard AI platform.

## Directory Structure
- `demo/`: Contains demonstration datasets:
  - `cicids2017_demo.csv`: Mathematically generated 6,000-flow dataset adhering strictly to the CIC-IDS2017 schema with realistic distributions for Normal, DDoS, PortScan, DoS Hulk, and Bot flows. Used for reproducible out-of-the-box local training and university viva presentations.
  - `cicids2017_sample_test.csv`: 250-flow validation sample ready for drag-and-drop or 1-click testing in the web dashboard.
- `raw/` (Optional): Place downloaded raw CIC-IDS2017 CSV files here.

## How to Use Real CIC-IDS2017 Data
1. Download the CIC-IDS2017 dataset from the official University of New Brunswick (UNB) portal:
   https://www.unb.ca/cic/datasets/ids-2017.html
2. Place any day's CSV (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) into `data/raw/`.
3. Run the ingestion and cleaning script:
   ```bash
   python scripts/prepare_data.py --input data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv --output data/cicids2017_cleaned.csv --sample 15000
   ```
4. Train models using the prepared dataset:
   ```bash
   python -m ml.train --data data/cicids2017_cleaned.csv --mode binary
   ```

*Note: Raw gigabyte-scale CSVs are excluded from Git via `.gitignore`.*
