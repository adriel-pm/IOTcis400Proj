"""
Maps IoT-23 (Zeek conn.log.labeled) and CICIoT2023 (flow CSV) into ONE
common feature schema so a model trained on one can be evaluated on the
other. This alignment step is itself part of the project's contribution
(see proposal, Methodology section) because the two datasets do not
share a native schema.

Common schema (7 columns + label):
  duration        -- flow duration in seconds
  total_bytes      -- total bytes both directions
  total_pkts       -- total packets both directions
  avg_pkt_size      -- total_bytes / total_pkts
  is_tcp, is_udp, is_icmp  -- one-hot protocol
  label            -- 0 = benign, 1 = attack   (kept for evaluation only;
                       NOT given to the unsupervised models as a feature)
"""
import pandas as pd
import numpy as np

COMMON_COLUMNS = ["duration", "total_bytes", "total_pkts", "avg_pkt_size",
                   "is_tcp", "is_udp", "is_icmp", "label", "detail_label"]


def load_iot23(path):
    """Reads a real or synthetic IoT-23 conn.log.labeled file (Zeek TSV)."""
    import re
    with open(path) as f:
        header_line = None
        for line in f:
            if line.startswith("#fields"):
                header_line = re.split(r"\s+", line.strip())[1:]
                break
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=header_line,
                      na_values="-", engine="python")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    df["orig_bytes"] = pd.to_numeric(df["orig_bytes"], errors="coerce").fillna(0)
    df["resp_bytes"] = pd.to_numeric(df["resp_bytes"], errors="coerce").fillna(0)
    df["orig_pkts"] = pd.to_numeric(df["orig_pkts"], errors="coerce").fillna(0)
    df["resp_pkts"] = pd.to_numeric(df["resp_pkts"], errors="coerce").fillna(0)

    out = pd.DataFrame()
    out["duration"] = df["duration"]
    out["total_bytes"] = df["orig_bytes"] + df["resp_bytes"]
    out["total_pkts"] = df["orig_pkts"] + df["resp_pkts"]
    out["avg_pkt_size"] = out["total_bytes"] / out["total_pkts"].replace(0, np.nan)
    out["avg_pkt_size"] = out["avg_pkt_size"].fillna(0)
    out["is_tcp"] = (df["proto"].str.lower() == "tcp").astype(int)
    out["is_udp"] = (df["proto"].str.lower() == "udp").astype(int)
    out["is_icmp"] = (df["proto"].str.lower() == "icmp").astype(int)
    # IoT-23 uses 'label' column: "Benign" / "benign" vs "Malicious"
    out["label"] = (~df["label"].str.lower().str.contains("benign", na=False)).astype(int)

    # NEW: keep the specific attack name too (not just benign/attack).
    # IoT-23's 'detailed-label' column holds this; benign rows have "-" here.
    out["detail_label"] = df["detailed-label"].fillna("benign")

    out["source"] = "iot23"
    return out[COMMON_COLUMNS + ["source"]]


def load_ciciot2023(path):
    """Reads a real or synthetic CICIoT2023 part-*.csv file."""
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame()
    out["duration"] = pd.to_numeric(df["IAT"], errors="coerce").fillna(0)
    out["total_bytes"] = pd.to_numeric(df["Tot sum"], errors="coerce").fillna(0)
    out["total_pkts"] = pd.to_numeric(df["Number"], errors="coerce").fillna(0)
    out["avg_pkt_size"] = pd.to_numeric(df["AVG"], errors="coerce").fillna(0)
    out["is_tcp"] = pd.to_numeric(df["TCP"], errors="coerce").fillna(0).astype(int)
    out["is_udp"] = pd.to_numeric(df["UDP"], errors="coerce").fillna(0).astype(int)
    out["is_icmp"] = pd.to_numeric(df["ICMP"], errors="coerce").fillna(0).astype(int)
    out["label"] = (~df["Label"].str.lower().str.contains("benign", na=False)).astype(int)

    # NEW: CICIoT2023's own 'Label' column IS the detailed attack name
    # already (e.g. "DDoS-SYN_Flood") -- we just keep the raw string
    # before it gets binarized into 0/1 above.
    out["detail_label"] = df["Label"]

    out["source"] = "ciciot2023"
    return out[COMMON_COLUMNS + ["source"]]


if __name__ == "__main__":
    import glob

    iot23_files = glob.glob("data/iot23/*.log.labeled")
    iot23_frames = [load_iot23(f) for f in iot23_files]
    a = pd.concat(iot23_frames, ignore_index=True)

    ciciot_files = glob.glob("data/ciciot2023/Merged0*.csv")
    ciciot_frames = [load_ciciot2023(f) for f in ciciot_files]
    b = pd.concat(ciciot_frames, ignore_index=True)

    print("IoT-23 rows:", len(a), "| attack rate:", a["label"].mean().round(3))
    print(a.head(3).to_string())
    print()
    print("CICIoT2023 rows:", len(b), "| attack rate:", b["label"].mean().round(3))
    print(b.head(3).to_string())
    a.to_csv("output/features_iot23.csv", index=False)
    b.to_csv("output/features_ciciot2023.csv", index=False)
    print("\nSaved both feature tables to output/")