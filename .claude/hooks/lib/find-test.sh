#!/usr/bin/env bash
# lib/find-test.sh — map a source file to its closest test file.
# Sourced by post-edit.sh.
#
# Strategy (in order):
#   1. Exact match:     tests/test_<module>.py
#   2. Directory match: tests/test_<package>.py
#   3. Glob match:      tests/test_*<module>*.py  (first result)
#   4. Empty string     (no test found → caller skips)

find_related_test() {
  local src_file="$1"       # absolute path to the source file
  local tests_dir="$2"      # absolute path to the tests/ directory

  local module package test_path found

  module=$(basename "$src_file" .py)
  package=$(basename "$(dirname "$src_file")")

  # 1. Exact match on module name
  test_path="$tests_dir/test_${module}.py"
  if [ -f "$test_path" ]; then echo "$test_path"; return; fi

  # 2. Match on package/directory name
  test_path="$tests_dir/test_${package}.py"
  if [ -f "$test_path" ]; then echo "$test_path"; return; fi

  # 3. Glob: any test file that contains the module name
  found=$(find "$tests_dir" -maxdepth 1 -name "test_*${module}*.py" 2>/dev/null | sort | head -1)
  if [ -n "$found" ]; then echo "$found"; return; fi

  # 4. Nothing found
  echo ""
}
