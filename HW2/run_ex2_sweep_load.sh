#!/bin/bash

# Define max load
MAX_LOAD=51538 

# Output file for the table
RESULTS_FILE="performance_results.csv"

# Write the CSV header
echo "Actual Load,Median Latency,Throughput" > $RESULTS_FILE
echo "Results will be saved to $RESULTS_FILE"
echo "----------------------------------------"

# Use 'bc' to calculate min, max, and step with decimals
MIN_LOAD=$(echo "scale=2; $MAX_LOAD / 10" | bc)
END_LOAD=$(echo "scale=2; $MAX_LOAD * 2" | bc)
STEP=$(echo "scale=4; ($END_LOAD - $MIN_LOAD) / 9" | bc)

# Loop 10 times
for i in {0..9}; do
    # Calculate the requested load to pass to the executable
    if [ $i -eq 9 ]; then
        REQUESTED_LOAD=$END_LOAD
    else
        REQUESTED_LOAD=$(echo "scale=2; $MIN_LOAD + ($i * $STEP)" | bc)
    fi

    echo "--- Requesting load: $REQUESTED_LOAD ---"
    
    # Run the command and capture stdout into a variable
    OUTPUT=$(./ex2 queue 256 $REQUESTED_LOAD)
    
    # Print the output to the terminal live
    echo "$OUTPUT"
    
    # Parse the output using awk
    ACTUAL_LOAD=$(echo "$OUTPUT" | awk '/load =/ {print $3}')
    MEDIAN=$(echo "$OUTPUT" | awk '/median/ {getline; print $3}')
    THROUGHPUT=$(echo "$OUTPUT" | awk '/throughput =/ {print $3}')
    
    # Fallbacks in case the program crashes or output format changes
    ACTUAL_LOAD=${ACTUAL_LOAD:-$REQUESTED_LOAD}
    MEDIAN=${MEDIAN:-"N/A"}
    THROUGHPUT=${THROUGHPUT:-"N/A"}

    # Append the REAL extracted data to the CSV file
    echo "$ACTUAL_LOAD,$MEDIAN,$THROUGHPUT" >> $RESULTS_FILE
    
    echo ""
done

echo "Experiment complete. Data written to $RESULTS_FILE."