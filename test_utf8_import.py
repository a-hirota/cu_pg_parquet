#!/usr/bin/env python3
"""UTF8インポートのテスト"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.type_map import UTF8, BINARY
    print(f"UTF8 = {UTF8}")
    print(f"BINARY = {BINARY}")
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    
try:
    from src.type_map import *
    print(f"UTF8 (wildcard import) = {UTF8}")
    print(f"BINARY (wildcard import) = {BINARY}")
except Exception as e:
    print(f"Wildcard import failed: {e}")