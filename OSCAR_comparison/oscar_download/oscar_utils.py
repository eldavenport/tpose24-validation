"""Utilities for downloading OSCAR surface currents for the TPOSE24 region.

Data source: PODAAC collection OSCAR_L4_OC_FINAL_V2.0 (concept-id
C2098858642-POCLOUD), daily 0.25-degree global ocean surface currents,
1993-01-01 through 2022-08-05. Access requires NASA Earthdata credentials
(read from ~/.netrc via earthaccess).

Each daily granule is a global file. download_oscar_range downloads each
granule to a temporary directory, subsets it to the requested lon/lat box,
and writes the subset to the output directory as netCDF.
"""

import os
import tempfile

import earthaccess
import xarray as xr

SHORT_NAME = "OSCAR_L4_OC_FINAL_V2.0"

# TPOSE24 model domain is ~209.35-230.65 E, -5.48-10.48 N. A margin is added
# so the box fully brackets the model grid for later regridding/interpolation.
LON_MIN, LON_MAX = 207.0, 233.0   # degrees East (0-360 convention)
LAT_MIN, LAT_MAX = -8.0, 13.0     # degrees North


def login():
    """Authenticate to Earthdata using credentials in ~/.netrc."""
    return earthaccess.login(strategy="netrc")


def search_granules(start_date, end_date):
    """Return OSCAR granule results for the inclusive UTC date range.

    start_date, end_date: 'YYYY-MM-DD' strings.
    """
    return earthaccess.search_data(
        short_name=SHORT_NAME,
        temporal=(f"{start_date}T00:00:00Z", f"{end_date}T23:59:59Z"),
    )


def _subset_to_region(ds):
    """Subset a global OSCAR dataset to the TPOSE24 lon/lat box.

    In OSCAR_L4_OC_FINAL_V2.0 the dimensions are named longitude/latitude while
    the coordinate variables are named lon/lat, so the coords are renamed to
    their dimension names to make them indexable. Longitude is stored 0-360;
    latitude may be ascending or descending, so the box is applied with
    sorted-order-agnostic selection.
    """
    rename = {}
    if "lon" in ds.coords and "longitude" in ds.dims and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if "lat" in ds.coords and "latitude" in ds.dims and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if rename:
        ds = ds.rename(rename)

    # Renaming a non-dimension coordinate to its dimension name does not build a
    # pandas index in current xarray, so set one explicitly before .sel().
    for dim in ("longitude", "latitude"):
        if dim in ds.coords and dim not in ds.indexes:
            ds = ds.set_xindex(dim)

    lat = ds["latitude"]
    if float(lat[0]) > float(lat[-1]):
        lat_slice = slice(LAT_MAX, LAT_MIN)
    else:
        lat_slice = slice(LAT_MIN, LAT_MAX)
    return ds.sel(longitude=slice(LON_MIN, LON_MAX), latitude=lat_slice)


def download_oscar_range(start_date, end_date, out_dir):
    """Download and region-subset OSCAR daily granules to out_dir.

    Returns the list of written file paths. Existing subset files in out_dir
    are skipped so the download is resumable.
    """
    os.makedirs(out_dir, exist_ok=True)
    login()
    granules = search_granules(start_date, end_date)
    print(f"Found {len(granules)} OSCAR granules for {start_date}..{end_date}")

    written = []
    with tempfile.TemporaryDirectory(prefix="oscar_raw_") as tmp:
        for i, g in enumerate(granules, 1):
            name = _granule_name(g)
            out_path = os.path.join(out_dir, name)
            if os.path.exists(out_path):
                print(f"[{i}/{len(granules)}] skip existing {name}")
                written.append(out_path)
                continue

            local = earthaccess.download([g], local_path=tmp)
            raw_path = local[0]
            with xr.open_dataset(raw_path) as ds:
                sub = _subset_to_region(ds).load()
            sub.attrs["region_subset"] = (
                f"lon {LON_MIN}-{LON_MAX}E, lat {LAT_MIN}-{LAT_MAX}N (TPOSE24)"
            )
            sub.to_netcdf(out_path)
            sub.close()
            os.remove(raw_path)
            print(f"[{i}/{len(granules)}] wrote {name}")
            written.append(out_path)

    return written


def _granule_name(granule):
    """Best-effort granule filename ending in .nc."""
    for url in granule.data_links():
        base = url.split("/")[-1]
        if base.endswith(".nc"):
            return base
    return f"{granule['umm']['GranuleUR']}.nc"
