#!/bin/bash

# Check if basename is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <basename>"
  exit 1
fi

basename="$1"
output_file="mixed.${basename}.wav"

# Initialize an empty string for the SoX command
sox_command="sox -M"

# Loop over all matching files in clips/number directory
for file in clips/numbered/"$basename".*.wav; do
  # Skip if no files found
  [ -f "$file" ] || continue

  # Create a temporary mono file with a .wav extension
  temp_file=$(mktemp /tmp/tempfile.XXXXXX)
  temp_file="${temp_file}.wav"
  
  # Convert to mono (using remix 1 to take the left channel)
  sox "$file" "$temp_file" remix 1
  
  # Append the temp file to the SoX command
  sox_command="$sox_command $temp_file"
done

# Add the output file to the command
sox_command="$sox_command $output_file"

# Run the command
eval $sox_command

# Clean up temporary files
rm -f /tmp/tempfile*.wav

echo "Output file created: $output_file"
