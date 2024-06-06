import os
import pandas as pd
import pickle
import sys

def main():
    if len(sys.argv) < 2:
        print("Please provide a command line argument.")
        return

    arg = sys.argv[1]

    # Load model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    # Read the CSV file
    df = pd.read_csv(arg, index_col=0)

    # Make predictions
    predictions = model.predict(df)

    # Print predictions
    for i, pred in enumerate(predictions):
        print(f"Prediction for row {i}: {pred}")

if __name__ == "__main__":
    main()
