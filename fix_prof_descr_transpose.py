"""
Fix transposed prof_descr in fastCTD_prof.nc and FCTD_1min.nc.

Problem: prof_descr is stored with dims ('iTXT', 'iPROF') = shape (30, nprof).
MATLAB's ncgetvar reverses C-order dims on read, so MATLAB sees (30, nprof)
instead of the required (nprof, 30). Every other working MITprof file uses
('iPROF', 'iTXT') = (nprof, 30).

Fix: rewrite the file with prof_descr dims swapped to ('iPROF', 'iTXT') and
data transposed accordingly. Original is preserved as <name>_old.nc.
"""

import os
import shutil
import netCDF4 as nc

PROF_DIR = "/data/SO3/edavenport/tpose24/profiles/"
FILES_TO_FIX = ["fastCTD_prof.nc", "FCTD_1min.nc"]


def fix_file(fname):
    src_path = os.path.join(PROF_DIR, fname)
    old_path = os.path.join(PROF_DIR, fname.replace(".nc", "_old.nc"))
    tmp_path = src_path + ".tmp"

    print(f"\n--- Fixing {fname} ---")

    # Preserve original before touching anything
    shutil.copy2(src_path, old_path)
    print(f"  Backed up original to: {os.path.basename(old_path)}")

    src = nc.Dataset(src_path, "r")
    dst = nc.Dataset(tmp_path, "w", format=src.file_format)

    # Global attributes
    dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})

    # Dimensions
    for name, dim in src.dimensions.items():
        dst.createDimension(name, None if dim.isunlimited() else len(dim))

    # Variables
    for vname, vobj in src.variables.items():
        old_dims = vobj.dimensions
        fill_kw = {}
        if "_FillValue" in vobj.ncattrs():
            fill_kw["fill_value"] = vobj._FillValue

        if vname == "prof_descr":
            if old_dims != ("iTXT", "iPROF"):
                raise ValueError(
                    f"prof_descr has unexpected dims {old_dims}; "
                    f"expected ('iTXT', 'iPROF')"
                )
            new_dims = ("iPROF", "iTXT")
            raw = vobj[:]          # shape (30, nprof), dtype |S1
            new_var = dst.createVariable(vname, vobj.dtype, new_dims,
                                         **fill_kw)
            new_var[:] = raw.T     # transpose -> (nprof, 30)
            print(f"  prof_descr: {old_dims} {raw.shape} "
                  f"-> {new_dims} {raw.T.shape}  [transposed]")
        else:
            new_var = dst.createVariable(vname, vobj.dtype, old_dims,
                                         **fill_kw)
            new_var[:] = vobj[:]

        # Variable attributes (skip _FillValue, already set via fill_value=)
        atts = {a: vobj.getncattr(a) for a in vobj.ncattrs()
                if a != "_FillValue"}
        if atts:
            new_var.setncatts(atts)

    src.close()
    dst.close()

    # Atomically replace the original with the fixed version
    os.replace(tmp_path, src_path)
    print(f"  Fixed file written: {fname}")


for f in FILES_TO_FIX:
    path = os.path.join(PROF_DIR, f)
    if not os.path.exists(path):
        print(f"Skipping {f}: not found at {path}")
        continue
    fix_file(f)

print("\nDone. Originals are in *_old.nc; fixed files keep original names.")
