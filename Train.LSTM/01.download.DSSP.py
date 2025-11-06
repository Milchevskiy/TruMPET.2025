#!/usr/bin/env python3
import requests
import sys

OUTDIR="DSSP"

# Читаем PDB ID из файла, по одной записи в строке
ids = []

if len(sys.argv) != 2:
    print("Usage: python3 01.download.DSSP.py chainlist.txt")
    exit(-1)

input_filename = sys.argv[1]

with open(input_filename, encoding="utf-8") as f:
    for line in f:
        entry = line.strip()
        if len(entry) < 4:
            continue
        pdb_id = entry[:4].lower()   # первые 4 символа, в строчные
        ids.append(pdb_id)

base_url = "https://pdb-redo.eu/dssp/db/{}/mmcif"

for pdb_id in ids:
    url = base_url.format(pdb_id)
    tries = 0
    while True:
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            break  # success, exit retry loop
        except requests.HTTPError as e:
            if resp.status_code == 503:
                tries += 1
                if tries >= 20:
                    print(f"Service unavailable for {pdb_id}, failed after 20 retries. Exiting.")
                    sys.exit(1)  # break everything
                print(f"503 error for {pdb_id}, retry {tries}/20 in 5s...")
                time.sleep(5)
                continue
            else:
                print(f"Error while downloading {pdb_id}: {e}")
                break  # skip this pdb_id
        except requests.RequestException as e:
            print(f"Network error for {pdb_id}: {e}")
            break  # skip this pdb_id

    else:
        # will never hit because of while True/break structure
        continue

    filename = f"{OUTDIR}/{pdb_id}.cif"
    with open(filename, "wb") as f:
        f.write(resp.content)
#    print(f"{filename} downloaded")
