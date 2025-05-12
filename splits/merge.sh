#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 <file1> <file2> ..."
  exit 1
fi
mkdir -p mono
# Loop through the provided files and cat each one individually
echo "--- Individual Files ---"
for file in "$@"; do
  sox "$file" "mono/$file" remix 1 
done

# Concatenate all files together at the end.  We use /dev/null
# as the initial input to avoid an "extra" newline at the beginning.
echo "--- Combined Files ---"
sox -M mono/* merged.wav
