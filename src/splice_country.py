"""Row splice: take all S1 rows of --country from --donor, everything else from --main (matching never crosses countries).
Usage: python splice_country.py --main output/v16 --donor output/v14 --country France --out output/v16u"""
import argparse
import pathlib
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--main"); ap.add_argument("--donor"); ap.add_argument("--country", default="France"); ap.add_argument("--out")
a = ap.parse_args()
R = pathlib.Path(__file__).resolve().parents[1]
s1 = pl.read_parquet(R / "cache/raw/test_source1.parquet", columns=["entity_id", "country"])
ids = set(s1.filter(pl.col("country") == a.country)["entity_id"])
print(f"{a.country} S1: {len(ids):,}")
out = R / a.out
out.mkdir(parents=True, exist_ok=True)
for f in ("matching_results.tsv", "candidate_pairs.tsv"):
    main = open(R / a.main / f).read().splitlines()
    don = {l.split("\t", 1)[0]: l for l in open(R / a.donor / f).read().splitlines()[1:]}
    n = 0
    with open(out / f, "w") as w:
        w.write(main[0] + "\n")
        for l in main[1:]:
            k = l.split("\t", 1)[0]
            if k in ids:
                n += l != don[k]
                l = don[k]
            w.write(l + "\n")
    print(f"{f}: {len(main) - 1:,} rows, {n:,} {a.country} rows replaced by donor")
