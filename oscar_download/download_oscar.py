"""Download OSCAR surface currents for the TPOSE24 region and time period.

Downloads daily OSCAR_L4_OC_FINAL_V2.0 granules (PODAAC, 0.25-degree) for the
model period (Oct 1 - Dec 30 2012), subsets each to the TPOSE24 domain, and
writes them as netCDF to OUT_DIR.

Requires NASA Earthdata credentials in ~/.netrc. Run in the tpose env:
    conda run -n tpose python download_oscar.py
"""

from oscar_utils import download_oscar_range

# Matches the TPOSE24 model runs (Oct 1 - Dec 30 2012).
START_DATE = "2012-10-01"
END_DATE = "2012-12-30"

OUT_DIR = "/data/SO3/edavenport/tpose24/oscar"


def main():
    paths = download_oscar_range(START_DATE, END_DATE, OUT_DIR)
    print(f"\nDone. {len(paths)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
