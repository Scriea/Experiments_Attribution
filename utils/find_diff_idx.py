import argparse
import pandas as pd

def find_differing_attributions(setting1_csv, setting2_csv):
    """Find question indices where attribution is Y in Setting 1 and N in Setting 2."""
    # Load CSVs into DataFrames
    setting1_df = pd.read_csv(setting1_csv)
    setting2_df = pd.read_csv(setting2_csv)

    # Ensure both CSVs have the same questions in the same order
    if not setting1_df['question'].equals(setting2_df['question']):
        raise ValueError("The questions in the two CSV files do not match.")

    # Find indices where attribution differs as specified
    differing_indices = setting1_df[
        (setting1_df['autoais'] == 'Y') & (setting2_df['autoais'] == 'N')
    ].index.tolist()

    return differing_indices

def main():
    parser = argparse.ArgumentParser(description="Find question indices with differing attributions across settings.")
    parser.add_argument(
        "--setting1_csv", required=True, help="Path to the CSV file for Setting 1."
    )
    parser.add_argument(
        "--setting2_csv", required=True, help="Path to the CSV file for Setting 2."
    )

    args = parser.parse_args()

    differing_indices = find_differing_attributions(args.setting1_csv, args.setting2_csv)

    print("\nIndices with attribution Y in Setting 1 and N in Setting 2:")
    print(differing_indices)

if __name__ == "__main__":
    main()
