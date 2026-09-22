"""
Generates a small synthetic file in the SAME format as a real IoT-23
conn.log.labeled file (Zeek/Bro TSV with #fields header), just so we can
test the feature-extraction pipeline end-to-end before the real dataset
is downloaded. Replace this file's output with the real conn.log.labeled
once you have it -- the extractor script doesn't care which one it reads.
"""
import random

random.seed(42)

FIELDS = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
          "proto", "service", "duration", "orig_bytes", "resp_bytes",
          "conn_state", "local_orig", "local_resp", "missed_bytes",
          "history", "orig_pkts", "orig_ip_bytes", "resp_pkts",
          "resp_ip_bytes", "tunnel_parents", "label", "detailed-label"]
TYPES = ["time", "string", "addr", "port", "addr", "port", "enum", "string",
         "interval", "count", "count", "string", "bool", "bool", "count",
         "string", "count", "count", "count", "count", "set[string]",
         "string", "string"]

def rand_ip(device=True):
    return f"192.168.1.{random.randint(2,50)}" if device else \
           f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def gen_row(ts, benign=True):
    proto = random.choice(["tcp", "udp"])
    orig_pkts = random.randint(1, 20)
    resp_pkts = random.randint(0, 20)
    orig_bytes = orig_pkts * random.randint(40, 200)
    resp_bytes = resp_pkts * random.randint(40, 200)
    duration = round(random.uniform(0.001, 5.0), 6)
    if benign:
        label, detail = "benign", "-"
        dst_port = random.choice([80, 443, 53, 123])
    else:
        label, detail = "Malicious", random.choice(
            ["PartOfAHorizontalPortScan", "DDoS", "C&C", "Okiru"])
        dst_port = random.randint(1, 65535)
        orig_pkts = random.randint(1, 3)
        resp_pkts = 0
        resp_bytes = 0
    row = [f"{ts:.6f}", f"C{random.randint(100000,999999)}", rand_ip(True),
           str(random.randint(1024, 65535)), rand_ip(False), str(dst_port),
           proto, "-", f"{duration:.6f}", str(orig_bytes), str(resp_bytes),
           "SF" if benign else "S0", "T", "F", "0", "ShAdDaf" if benign else "S",
           str(orig_pkts), str(orig_bytes + orig_pkts * 40), str(resp_pkts),
           str(resp_bytes + resp_pkts * 40), "-", label, detail]
    return "\t".join(row)

def write_file(path, n_benign=300, n_malicious=100):
    with open(path, "w") as f:
        f.write("#separator \\x09\n")
        f.write("#set_separator\t,\n")
        f.write("#empty_field\t(empty)\n")
        f.write("#unset_field\t-\n")
        f.write("#path\tconn\n")
        f.write("#fields\t" + "\t".join(FIELDS) + "\n")
        f.write("#types\t" + "\t".join(TYPES) + "\n")
        ts = 1600000000.0
        rows = []
        for _ in range(n_benign):
            ts += random.uniform(0.5, 3.0)
            rows.append(gen_row(ts, benign=True))
        for _ in range(n_malicious):
            ts += random.uniform(0.01, 0.2)
            rows.append(gen_row(ts, benign=False))
        random.shuffle(rows)
        for r in rows:
            f.write(r + "\n")
        f.write("#close\t2020-01-01-00-00-00\n")

if __name__ == "__main__":
    write_file("/home/claude/project1/data/iot23/CTU-Honeypot-Capture-SYNTHETIC.log.labeled")
    print("wrote synthetic IoT-23 style file")
