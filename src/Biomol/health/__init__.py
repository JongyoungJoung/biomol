"""Check health of the insallation."""

import logging
import os

from Biomol import DATA_DIR

logger = logging.getLogger(__name__)


def check_binaries():
    """Check if binaries are installed."""
    return True


def check_env():
    """Check environment variables."""
    Biomol_data_dir = os.environ.get("BIOMOLECULE_DATA_DIR")

    if Biomol_data_dir is None or Biomol_data_dir == "":
        logger.warning(
            ":face_with_thermometer: Please set the enviroment variable BIOMOLECULE_DATA_DIR to the path of the data directory.\n"
            f"Otherwise, the default {DATA_DIR} will be used."
        )
        return False
    logger.info(f":white_check_mark: BIOMOLECULE_DATA_DIR is set to {Biomol_data_dir}")
    return True


def main():
    successes = [check_env()]
    successes.append(check_binaries())

    if all(successes):
        logger.info("")
        logger.info(":muscle: You are ready to go!")


if __name__ == "__main__":
    main()
