import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Ensure data directory exists
    os.makedirs('tourism_project/data', exist_ok=True)

    # 1. Load the dataset from local repository folder
    dataset_path = 'tourism_project/data/tourism.csv'
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # 2. Preprocess: Remove unnecessary columns if they exist
    columns_to_drop = ['Unnamed: 0', 'CustomerID']
    existing_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_to_drop:
        print(f"Dropping unnecessary columns: {existing_to_drop}")
        df = df.drop(columns=existing_to_drop)

    # 3. Split the dataset into train and test sets (80/20 split)
    print("Splitting data into train and test sets...")
    stratify_col = df['ProdTaken'] if 'ProdTaken' in df.columns else None
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_col)

    # 4. Save splits locally as CSV files
    train_path = 'tourism_project/data/train.csv'
    test_path = 'tourism_project/data/test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Successfully saved train split to: {train_path} ({len(train_df)} rows)")
    print(f"Successfully saved test split to: {test_path} ({len(test_df)} rows)")

if __name__ == '__main__':
    main()