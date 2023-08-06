#!/bin/bash
set -xeu

# Create a temporary directory and clone the git repo
readonly TEMPDIR=$(mktemp -d)

# Trap to cleanup the TEMPDIR on exit
trap "rm -rf $TEMPDIR" EXIT

# Prepare the environment
git clone https://github.com/iwiwi/epochraft.git ${TEMPDIR}
python3 -m venv "${TEMPDIR}/venv"
source "${TEMPDIR}/venv/bin/activate"
cd "${TEMPDIR}"

# Install twine
pip install -U pip
pip install -U twine wheel build

# Check if the version ends with 'dev'
readonly OLD_VERSION=$(grep '__version__' epochraft/version.py | cut -d '"' -f 2)
if [[ "${OLD_VERSION}" != *dev ]]; then
  echo "Version does not end with 'dev'. Aborting."
  exit 1
fi

# Change the version file for scheduled TestPyPI upload
readonly DATE=$(date +"%Y%m%d")
readonly NEW_VERSION="${OLD_VERSION}${DATE}"
sed -i.bak -e "s/${OLD_VERSION}/${NEW_VERSION}/" epochraft/version.py

echo "Preparing to upload version: ${NEW_VERSION}"

# Build a tar ball
python -m build --sdist --wheel

# Verify the distributions
twine check dist/*

# Publish distribution to TestPyPI

# Ask for confirmation before uploading to TestPyPI
read -r -p "Do you really want to upload to production PyPI? (y/N): " yn
: ${yn:=n}
case $yn in
  [Yy]* )
    # Publish distribution to TestPyPI
    twine upload -r testpypi-epochraft dist/*
esac

# Ask for confirmation before uploading to PyPI
read -r -p "Do you really want to upload to production PyPI? (y/N): " yn
: ${yn:=n}
case $yn in
  [Yy]* )
    # Publish distribution to PyPI
    twine upload -r pypi-epochraft dist/* ;;
esac

# Clean up
deactivate
