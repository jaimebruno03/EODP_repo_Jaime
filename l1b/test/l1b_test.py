# CROSS VALIDATE L1B OUTPUTS EQUALIZED

# PLOT FROM YOUR OUTPUTS THE EQUALISED OUTPUT VERSUS NOT EQUALISED VERSUS THE TRUTH
# TRUTH = EODP-TS-L1B/input/ism_toa_isrf_VNIR-0.nc

from pathlib import Path
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

def compare_nc_folders(
    ref_dir: str, test_dir: str, rtol: float = 1e-4, atol: float = 1e-4
) -> None:
    ref_path = Path(ref_dir)
    test_path = Path(test_dir)

    ref_files = sorted(list(ref_path.glob("*.nc")))
    if not ref_files:
        print(f"No .nc files found in: {ref_path}")
        return

    print(f"Evaluating {len(ref_files)} NetCDF file(s)...\n")

    for ref_file in ref_files:
        test_file = test_path / ref_file.name
        print("=" * 65)
        print(f"File: {ref_file.name}")

        if not test_file.exists():
            print(f"❌ Missing counterpart file in test folder: {test_file.name}\n")
            continue

        with (
            nc.Dataset(ref_file, "r") as ds_ref,
            nc.Dataset(test_file, "r") as ds_test,
        ):
            ref_vars = set(ds_ref.variables.keys())
            test_vars = set(ds_test.variables.keys())

            common_vars = sorted(list(ref_vars.intersection(test_vars)))
            missing_vars = ref_vars - test_vars

            if missing_vars:
                print(f"  ⚠️ Missing variables: {missing_vars}")

            all_matched = True

            for var_name in common_vars:
                ref_data = ds_ref.variables[var_name][:]
                test_data = ds_test.variables[var_name][:]

                # 1. Dimension validation
                if ref_data.shape != test_data.shape:
                    print(
                        f"  ❌ [{var_name}] Shape mismatch: "
                        f"Ref {ref_data.shape} vs Test {test_data.shape}"
                    )
                    all_matched = False
                    continue

                # 2. Skip non-numeric data arrays (strings, characters)
                if not np.issubdtype(ref_data.dtype, np.number):
                    continue

                # 3. Numerical verification
                close_mask = np.isclose(
                    ref_data, test_data, rtol=rtol, atol=atol, equal_nan=True
                )

                if np.all(close_mask):
                    max_diff = np.nanmax(np.abs(ref_data - test_data))
                    print(f"  ✅ [{var_name}] Match (Max diff: {max_diff:.2e})")
                else:
                    all_matched = False
                    max_diff = np.nanmax(np.abs(ref_data - test_data))
                    bad_indices = np.where(~close_mask)

                    # Extract coordinates of the first failing element regardless of dimensions
                    first_coord = tuple(idx[0] for idx in bad_indices)
                    expected_val = ref_data[first_coord]
                    actual_val = test_data[first_coord]

                    print(f"  ❌ [{var_name}] Mismatch found!")
                    print(f"     Max diff: {max_diff:.6e}")
                    print(
                        f"     First error at indices {first_coord}: "
                        f"Expected {expected_val} vs Got {actual_val}"
                    )

            if all_matched and not missing_vars:
                print("  🎉 Status: IDENTICAL")
            else:
                print("  ⚠️ Status: DIFFERENCES DETECTED")
            print()


if __name__ == "__main__":
    REFERENCE_DIR = "C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/output/"
    MY_OUTPUT_DIR = "C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/myOutput/"

    compare_nc_folders(REFERENCE_DIR, MY_OUTPUT_DIR)

    # --- 1. File paths ---
    file_with_eq = Path("C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/myOutput/l1b_toa_VNIR-0.nc")
    file_no_eq = Path("C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/myOutput_NotEq/l1b_toa_VNIR-0.nc")
    file_isrf = Path("C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/input/ism_toa_isrf_VNIR-0.nc")

    # --- 2. Load data ---
    with nc.Dataset(file_with_eq, "r") as ds:
        # If 2D (alt_lines, act_columns), take a representative line (e.g., line 0)
        data_with_eq = ds.variables["toa"][:]
        profile_with_eq = (
            data_with_eq[0, :] if data_with_eq.ndim > 1 else data_with_eq
        )

    with nc.Dataset(file_no_eq, "r") as ds:
        data_no_eq = ds.variables["toa"][:]
        profile_no_eq = data_no_eq[0, :] if data_no_eq.ndim > 1 else data_no_eq

    with nc.Dataset(file_isrf, "r") as ds:
        # Adjust variable name if named differently (e.g., 'toa', 'radiance', 'isrf')
        var_name = "toa" if "toa" in ds.variables else list(ds.variables.keys())[-1]
        data_isrf = ds.variables[var_name][:]
        profile_isrf = data_isrf[0, :] if data_isrf.ndim > 1 else data_isrf

    # Across-track pixel indices (X axis)
    act_pixels = np.arange(len(profile_with_eq))

    # --- 3. Generate plot ---
    plt.figure(figsize=(10, 5.5))

    plt.plot(
        act_pixels,
        profile_with_eq,
        color="black",
        linewidth=1.5,
        label="TOA L1B with eq",
    )
    plt.plot(
        act_pixels, profile_no_eq, color="red", linewidth=1.5, label="TOA L1B no eq"
    )
    plt.plot(
        act_pixels,
        profile_isrf,
        color="blue",
        linewidth=1.5,
        label="TOA after the ISRF",
    )

    plt.title("Effect of the Equalization for VNIR-0", fontsize=13)
    plt.xlabel("ACT pixel [-]", fontsize=11)
    plt.ylabel("TOA [mW/m2/sr]", fontsize=11)

    plt.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)
    plt.legend(loc="upper left", frameon=True, fontsize=9)

    plt.xlim(-5, len(act_pixels) + 5)
    plt.tight_layout()
    plt.show()