"""
The main script to run all simulation scripts in the src/simulation folder.
"""

import os
import argparse

# We need to fill this in with the correct path to the simulation folder, after all done!
def main(args):
    if not args.experiment:
        # run the experiments
        os.system('python3 simulation/experiments.py')
    else:
        # run the multiple experiments
        os.system('python3 simulation/run_multiple_experiments.py')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiple', type=bool, default=False, help='Run multiple experiments')
    args = parser.parse_args()
    main(args)
