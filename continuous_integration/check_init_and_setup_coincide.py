#!/usr/bin/env python3

"""Check that the distribution and masterigayim/__init__.py are in sync."""

import os
import pathlib
import subprocess
import sys
from typing import Optional, Dict

import masterigayim


def main() -> int:
    """Execute the main routine."""
    repo_root = pathlib.Path(os.path.realpath(__file__)).parent.parent

    setup_py_pth = repo_root / "setup.py"
    if not setup_py_pth.exists():
        raise RuntimeError(f"Could not find_our_type the setup.py: {setup_py_pth}")

    success = True

    ##
    # Check basic fields
    ##

    setup_py_map = dict()  # type: Dict[str, str]

    fields = ["version", "author", "license", "description"]
    for field in fields:
        out = subprocess.check_output(
            [sys.executable, str(repo_root / "setup.py"), f"--{field}"],
            encoding="utf-8",
        ).strip()

        setup_py_map[field] = out

    if setup_py_map["version"] != masterigayim.__version__:
        print(
            f"The version in the setup.py is {setup_py_map['version']}, "
            f"while the version in masterigayim/__init__.py is: "
            f"{masterigayim.__version__}",
            file=sys.stderr,
        )
        success = False

    if setup_py_map["author"] != masterigayim.__author__:
        print(
            f"The author in the setup.py is {setup_py_map['author']}, "
            f"while the author in masterigayim/__init__.py is: "
            f"{masterigayim.__author__}",
            file=sys.stderr,
        )
        success = False

    if setup_py_map["license"] != masterigayim.__license__:
        print(
            f"The license in the setup.py is {setup_py_map['license']}, "
            f"while the license in masterigayim/__init__.py is: "
            f"{masterigayim.__license__}",
            file=sys.stderr,
        )
        success = False

    if setup_py_map["description"] != masterigayim.__doc__:
        print(
            f"The description in the setup.py is {setup_py_map['description']}, "
            f"while the description in masterigayim/__init__.py is: "
            f"{masterigayim.__doc__}",
            file=sys.stderr,
        )
        success = False

    ##
    # Classifiers need special attention as there are multiple.
    ##

    # This is the map from the distribution to expected status in __init__.py.
    status_map = {
        "Development Status :: 1 - Planning": "Planning",
        "Development Status :: 2 - Pre-Alpha": "Pre-Alpha",
        "Development Status :: 3 - Alpha": "Alpha",
        "Development Status :: 4 - Beta": "Beta",
        "Development Status :: 5 - Production/Stable": "Production/Stable",
        "Development Status :: 6 - Mature": "Mature",
        "Development Status :: 7 - Inactive": "Inactive",
    }

    classifiers = (
        subprocess.check_output(
            [sys.executable, str(setup_py_pth), "--classifiers"], encoding="utf-8"
        )
        .strip()
        .splitlines()
    )

    status_classifier = None  # type: Optional[str]
    for classifier in classifiers:
        if classifier in status_map:
            status_classifier = classifier
            break

    if status_classifier is None:
        print(
            "Expected a status classifier in setup.py "
            "(e.g., 'Development Status :: 3 - Alpha'), but found none.",
            file=sys.stderr,
        )
        success = False
    else:
        expected_status_in_init = status_map[status_classifier]

        if expected_status_in_init != masterigayim.__status__:
            print(
                f"Expected status {expected_status_in_init} "
                f"according to setup.py in masterigayim/__init__.py, "
                f"but found: {masterigayim.__status__}"
            )
            success = False

    if not success:
        return -1

    return 0


if __name__ == "__main__":
    sys.exit(main())
