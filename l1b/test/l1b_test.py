# CROSS VALIDATE L1B OUTPUTS EQUALIZED
# PLOT FROM YOUR OUTPUTS: EQUALIZED OUTPUT vs NOT EQUALIZED vs TRUTH BENCHMARK
# TRUTH = EODP-TS-L1B/input/ism_toa_isrf_VNIR-0.nc

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# 1. Resolve base directory and configure global logger
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "eodp_validation.log"
PLOT_FILE = SCRIPT_DIR / "equalization_plot.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def compare_nc_folders(
    ref_dir: str, test_dir: str, rtol: float = 1e-3, atol: float = 1e-4
) -> None:
    ref_path = Path(ref_dir)
    test_path = Path(test_dir)

    ref_files = sorted(list(ref_path.glob("*.nc")))
    if not ref_files:
        logger.warning(f"No .nc files found in: {ref_path}")
        return

    logger.info(f"Evaluating {len(ref_files)} NetCDF file(s)...")

    for ref_file in ref_files:
        test_file = test_path / ref_file.name
        logger.info("=" * 65)
        logger.info(f"File: {ref_file.name}")

        if not test_file.exists():
            logger.error(
                f"Missing counterpart file in test folder: {test_file.name}"
            )
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
                logger.warning(f"  Missing variables in test file: {missing_vars}")

            all_matched = True

            for var_name in common_vars:
                ref_data = ds_ref.variables[var_name][:]
                test_data = ds_test.variables[var_name][:]

                # 1. Dimension validation
                if ref_data.shape != test_data.shape:
                    logger.error(
                        f"  [{var_name}] Shape mismatch: Ref {ref_data.shape} vs Test {test_data.shape}"
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
                    logger.info(
                        f"  [PASSED] [{var_name}] Match (Max diff: {max_diff:.2e})"
                    )
                else:
                    all_matched = False
                    max_diff = np.nanmax(np.abs(ref_data - test_data))
                    bad_indices = np.where(~close_mask)

                    # Extract coordinates of the first failing element
                    first_coord = tuple(idx[0] for idx in bad_indices)
                    expected_val = ref_data[first_coord]
                    actual_val = test_data[first_coord]

                    logger.error(f"  [FAILED] [{var_name}] Discrepancy detected!")
                    logger.error(f"     Max absolute diff: {max_diff:.6e}")
                    logger.error(
                        f"     First error at {first_coord}: Expected {expected_val} vs Got {actual_val}"
                    )

            if all_matched and not missing_vars:
                logger.info("  STATUS: IDENTICAL (MATCH)")
            else:
                logger.warning("  STATUS: DISCREPANCIES DETECTED")


if __name__ == "__main__":
    logger.info("Starting EODP validation run.")
    logger.info(f"Log will be saved at: {LOG_FILE}")

    REFERENCE_DIR = "C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/output/"
    MY_OUTPUT_DIR = "C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/myOutput/"

    # Execute cross-validation batch comparison
    compare_nc_folders(REFERENCE_DIR, MY_OUTPUT_DIR)

    # --- 1. File paths for spatial plotting ---
    file_with_eq = Path(
        "C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/myOutput/l1b_toa_VNIR-0.nc"
    )
    file_no_eq = Path(
        "C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/myOutput_NotEq/l1b_toa_VNIR-0.nc"
    )
    file_isrf = Path(
        "C:/MasterEODP/EODP_TER_2021/EODP-TS-L1B/input/ism_toa_isrf_VNIR-0.nc"
    )

    # --- 2. Load data arrays ---
    with nc.Dataset(file_with_eq, "r") as ds:
        data_with_eq = ds.variables["toa"][:]
        profile_with_eq = (
            data_with_eq[0, :] if data_with_eq.ndim > 1 else data_with_eq
        )

    with nc.Dataset(file_no_eq, "r") as ds:
        data_no_eq = ds.variables["toa"][:]
        profile_no_eq = (
            data_no_eq[0, :] if data_no_eq.ndim > 1 else data_no_eq
        )

    with nc.Dataset(file_isrf, "r") as ds:
        var_name = (
            "toa" if "toa" in ds.variables else list(ds.variables.keys())[-1]
        )
        data_isrf = ds.variables[var_name][:]
        profile_isrf = data_isrf[0, :] if data_isrf.ndim > 1 else data_isrf

    # Across-track detector indices (X axis)
    act_pixels = np.arange(len(profile_with_eq))

    # --- 3. Generate comparison plot ---
    plt.figure(figsize=(10, 5.5))

    plt.plot(
        act_pixels,
        profile_with_eq,
        color="black",
        linewidth=1.5,
        label="TOA L1B with eq",
    )
    plt.plot(
        act_pixels,
        profile_no_eq,
        color="red",
        linewidth=1.5,
        label="TOA L1B no eq",
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

    # Save figure to disk before display
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved successfully at: {PLOT_FILE}")

    plt.show()
    logger.info("Process finished successfully.")