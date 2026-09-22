"""
Generates a small synthetic CSV in the SAME column layout as a real
CICIoT2023 part-*.csv file (46 flow-level features + label), just so we
can test the pipeline end-to-end before the real dataset is downloaded.
Replace this file's output with a real CICIoT2023 part file later --
the extractor script keys off these exact column names either way.
"""
import random
import csv

random.seed(7)

COLUMNS = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
    "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count", "HTTP", "HTTPS",
    "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
    "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
    "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight", "label"
]

def gen_row(benign=True):
    is_tcp = random.random() < 0.7
    proto = "TCP" if is_tcp else "UDP"
    number = random.randint(1, 30)
    avg = random.uniform(40, 400)
    tot = avg * number
    row = {c: 0 for c in COLUMNS}
    row["flow_duration"] = round(random.uniform(0.001, 5.0), 6)
    row["Header_Length"] = random.choice([20, 40])
    row["Protocol Type"] = 6 if is_tcp else 17
    row["Duration"] = row["flow_duration"]
    row["Number"] = number
    row["Tot sum"] = round(tot, 2)
    row["AVG"] = round(avg, 2)
    row["Min"] = round(avg * 0.6, 2)
    row["Max"] = round(avg * 1.4, 2)
    row["Std"] = round(avg * 0.15, 2)
    row["Tot size"] = round(tot, 2)
    row["Rate"] = round(number / max(row["flow_duration"], 1e-3), 2)
    row["TCP"] = 1 if is_tcp else 0
    row["UDP"] = 0 if is_tcp else 1
    if benign:
        row["label"] = "BenignTraffic"
        row["syn_flag_number"] = 1 if is_tcp else 0
        row["ack_flag_number"] = 1 if is_tcp else 0
    else:
        row["label"] = random.choice(["DDoS-SYN_Flood", "DoS-UDP_Flood",
                                       "Recon-PortScan", "Mirai-greeth_flood"])
        row["syn_flag_number"] = 1
        row["syn_count"] = random.randint(5, 50)
        row["Rate"] = row["Rate"] * random.uniform(5, 20)
    return row

def write_file(path, n_benign=300, n_malicious=100):
    rows = [gen_row(True) for _ in range(n_benign)] + \
           [gen_row(False) for _ in range(n_malicious)]
    random.shuffle(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    write_file("/home/claude/project1/data/ciciot2023/part-SYNTHETIC-00000.csv")
    print("wrote synthetic CICIoT2023 style file")
