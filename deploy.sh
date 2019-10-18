#!/bin/sh
set -e

rm -rf dist toai.egg-info build
python3 setup.py sdist bdist_wheel
twine upload dist/*
