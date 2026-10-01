import logging
from pathlib import Path
import netCDF4 as nc
import numpy as np

DIR_MY = Path(r"C:/MasterEODP/EODP_TER_2021/EODP-TS-ISM/myOutput")
DIR_REF = Path(r"C:/MasterEODP/EODP_TER_2021/EODP-TS-ISM/output")
LOG_PATH = DIR_MY / "eodp_validation.log"

ATOL = 1e-4
RTOL = 1e-4


def setup_logger(log_file: Path) -> logging.Logger:
  log_file.parent.mkdir(parents=True, exist_ok=True)
  logger = logging.getLogger("EODP_ISM_Validator")
  logger.setLevel(logging.INFO)
  logger.handlers.clear()


  formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")


  fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  ch = logging.StreamHandler()
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  return logger


def validate_ism_netcdf(my_dir: Path, ref_dir: Path, log_file: Path):
  logger = setup_logger(log_file)

  logger.info("Starting EODP validation run.")
  logger.info(f"Log will be saved at: {log_file.resolve()}")

  my_files = sorted(my_dir.glob("*.nc"))
  logger.info(f"Evaluating {len(my_files)} NetCDF file(s)...")

  separator = "=" * 80

  for my_file in my_files:
    logger.info(separator)
    logger.info(f"File: {my_file.name}")

    ref_file = ref_dir / my_file.name
    if not ref_file.exists():
      logger.warning(
          f"  [MISSING] Reference file not found in: {ref_dir.name}"
      )
      logger.info("  STATUS: FAILED (MISSING REFERENCE)")
      continue

    try:
      with (
          nc.Dataset(my_file, "r") as ds_my,
          nc.Dataset(ref_file, "r") as ds_ref,
      ):
        common_vars = set(ds_my.variables.keys()) & set(
            ds_ref.variables.keys()
        )

        if not common_vars:
          logger.warning("  [NO COMMON VARIABLES]")
          logger.info("  STATUS: FAILED (NO VARIABLES)")
          continue

        all_matched = True
        is_stochastic = any(
            k in my_file.name for k in ["prnu", "ds", "detection"]
        )

        for var_name in sorted(common_vars):
          arr_my = np.array(ds_my.variables[var_name][:])
          arr_ref = np.array(ds_ref.variables[var_name][:])

          if arr_my.shape != arr_ref.shape:
            logger.error(
                f"  [SHAPE MISMATCH] [{var_name}] {arr_my.shape} vs"
                f" {arr_ref.shape}"
            )
            all_matched = False
            continue

          abs_diff = np.abs(arr_my - arr_ref)
          max_abs = np.nanmax(abs_diff)
          close = np.allclose(
              arr_my, arr_ref, rtol=RTOL, atol=ATOL, equal_nan=True
          )

          if close:
            logger.info(
                f"  [PASSED] [{var_name}] Match (Max diff: {max_abs:.2e})"
            )
          else:
            all_matched = False
            if is_stochastic:
              logger.info(
                  f"  [STOCHASTIC] [{var_name}] Random variation (Max diff:"
                  f" {max_abs:.2e})"
              )
            else:
              logger.error(
                  f"  [FAILED] [{var_name}] Mismatch (Max diff: {max_abs:.2e})"
              )

        if all_matched:
          logger.info("  STATUS: IDENTICAL (MATCH)")
        elif is_stochastic:
          logger.info("  STATUS: EXPECTED DEVIATION (STOCHASTIC NOISE)")
        else:
          logger.info("  STATUS: MISMATCH")

    except Exception as e:
      logger.error(f"  [ERROR] Reading NetCDF: {str(e)}")
      logger.info("  STATUS: FAILED (EXCEPTION)")

  logger.info(separator)
  logger.info("Process finished successfully.")


if __name__ == "__main__":
  validate_ism_netcdf(DIR_MY, DIR_REF, LOG_PATH)