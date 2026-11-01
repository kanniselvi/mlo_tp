from io import BytesIO
from urllib.request import Request, urlopen
from datetime import datetime, timezone

import pandas as pd

# GitHub repository configuration
GITHUB_OWNER = "kanniselvi"
REPOSITORY = "mlo_tp"
BRANCH = "main"
DATASET_PATH = "tourism_project/data/tourism.csv"

# Direct URL to the CSV stored in GitHub
DATASET_URL = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_OWNER}/{REPOSITORY}/{BRANCH}/{DATASET_PATH}"
)

# Required columns for the tourism prediction project
EXPECTED_COLUMNS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "ProdTaken",
]


def register_dataset():
    """Read the GitHub dataset, validate columns, and print updated details."""

    print("=" * 60)
    print("TOURISM DATASET REGISTRATION")
    print("=" * 60)

    try:
        # Fetch the CSV directly from GitHub
        request = Request(
            DATASET_URL,
            headers={"User-Agent": "Tourism-MLOps-Dataset-Registry"},
        )

        with urlopen(request, timeout=30) as response:
            csv_content = response.read()
            last_modified = response.headers.get("Last-Modified", "Unavailable")

        # Load the latest committed CSV content
        df = pd.read_csv(BytesIO(csv_content))

        # Remove accidental whitespace from column names
        df.columns = df.columns.str.strip()

        # Validate expected columns
        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            print("Dataset validation: FAILED")
            print("Missing required columns:", missing_columns)
            print("Available columns:", df.columns.tolist())
            raise ValueError("Dataset does not contain all required columns.")

        # Generate updated dataset details
        print("Dataset status       : Successfully loaded")
        print(f"GitHub repository    : {GITHUB_OWNER}/{REPOSITORY}")
        print(f"Branch               : {BRANCH}")
        print(f"Dataset path         : {DATASET_PATH}")
        print(f"Dataset URL          : {DATASET_URL}")
        print(f"Last modified        : {last_modified}")
        print(
            "Checked at (UTC)     : "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"Total records        : {len(df)}")
        print(f"Total columns        : {len(df.columns)}")
        print(f"Required columns     : {len(EXPECTED_COLUMNS)}")
        print(f"Missing values       : {df.isnull().sum().sum()}")
        print(f"Duplicate records    : {df.duplicated().sum()}")
        print("Dataset validation   : PASSED")

        # Target column summary
        print("\nTARGET COLUMN SUMMARY")
        print("-" * 60)
        print(df["ProdTaken"].value_counts(dropna=False).to_string())

        # Show column names and data types
        print("\nCOLUMN DETAILS")
        print("-" * 60)
        print(df.dtypes.to_string())

        # Display a sample of the latest dataset
        print("\nFIRST 5 RECORDS")
        print("-" * 60)
        print(df.head().to_string(index=False))

        print("\nDataset registration completed successfully.")

        return df

    except Exception as error:
        print(f"\nDataset registration failed: {error}")
        raise


if __name__ == "__main__":
    register_dataset()