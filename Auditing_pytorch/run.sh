#!/bin/bash

# Define the array of epsilon values
EPSILONS=(0.1 0.5 1 1.5 2.0 3.0 3.5 4 4.5 5 5.5 6)

# Iterate through the epsilon values
for i in "${EPSILONS[@]}"
do
    echo "Running with epsilon = $i"

    # Run your Python script for 'original' datatype
    python main.py --noise 0.6 --epsilon $i --data_type 'original'

    # Run your Python script for 'processed' datatype
    python main.py --noise 0.6 --epsilon $i --data_type 'processed'

    echo "Iteration with epsilon = $i completed"
done

echo "All iterations completed."
