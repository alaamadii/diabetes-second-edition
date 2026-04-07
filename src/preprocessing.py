import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'diabetes.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # 1. Split data into X and y
    print("\nSplitting data into X (features) and y (target)...")
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # 2. Train/Test Split (80/20)
    print("\nPerforming Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # 3. Feature Scaling using StandardScaler
    print("\nApplying Feature Scaling using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Save processed data
    print("\nSaving processed data to 'data/processed'...")
    X_train_scaled_df.to_csv(os.path.join(processed_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled_df.to_csv(os.path.join(processed_dir, 'X_test_scaled.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()
