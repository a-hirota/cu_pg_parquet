#!/bin/bash
echo "=== Python環境・CuDFバージョン確認 ==="
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser
python check_cudf_version.py