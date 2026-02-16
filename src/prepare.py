import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split


def main():
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)

    df = df.sample(n=1000000, random_state=365)

    drop_columns = ["Property ID", "Posted On", "Floor", "Area Locality", "City"]
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    selected_columns = [
        "BHK",
        "Size",
        "Bathroom",
        "Building Type",
        "Area Type",
        "Furnishing Status",
        "Tenant Preferred",
        "Point of Contact",
        "Rent"
    ]

    df = df[selected_columns]

    df = df.dropna()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=365
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Data preparation completed.")


if __name__ == "__main__":
    main()
