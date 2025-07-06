#!/bin/bash

# Script to repeatedly run the supervoxel generation Julia script with 24 threads

# Configuration 
JULIA_SCRIPT="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/get_synth_data/get_synth_grey_poligons.jl"
THREADS=24
MAX_ITERATIONS=1000  # Set to -1 for infinite loop
DELAY_SECONDS=5  # Delay between iterations to allow system to cool down

# Function to handle script termination
cleanup() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Script terminated by user. Cleaning up..."
    exit 0
}

# Set trap for Ctrl+C
trap cleanup SIGINT SIGTERM

# Check if Julia script exists
if [ ! -f "$JULIA_SCRIPT" ]; then
    echo "Error: Julia script not found at $JULIA_SCRIPT"
    exit 1
fi

# Initialize counter
iteration=1

# Main loop
echo "Starting repeated execution of supervoxel generation with $THREADS threads"
echo "Press Ctrl+C to stop"

while [ $MAX_ITERATIONS -eq -1 ] || [ $iteration -le $MAX_ITERATIONS ]; do
    echo "============================================================"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting iteration $iteration"
    
    # Run the Julia script with specified number of threads
    JULIA_NUM_THREADS=$THREADS julia "$JULIA_SCRIPT"
    
    # Check exit status
    if [ $? -ne 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Warning: Julia script exited with error"
    fi
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed iteration $iteration"
    
    # Increment counter
    ((iteration++))
    
    # Add delay between runs
    if [ $MAX_ITERATIONS -eq -1 ] || [ $iteration -le $MAX_ITERATIONS ]; then
        echo "Waiting $DELAY_SECONDS seconds before next run..."
        sleep $DELAY_SECONDS
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - Finished all $MAX_ITERATIONS iterations"