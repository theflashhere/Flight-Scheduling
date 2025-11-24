import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Path to CSV/XLSX")
    args = p.parse_args()

    if args.path.lower().endswith(".xlsx"):
        df = pd.read_excel(args.path)
    else:
        df = pd.read_csv(args.path)

    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nHead:\n", df.head(10))

if __name__ == "__main__":
    main()
