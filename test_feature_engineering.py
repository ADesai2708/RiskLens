from src.data_loader import load_data
from src.feature_engineering import create_features


# Load raw data
df = load_data()

print("Original shape:")
print(df.shape)

# Create engineered features
df_engineered = create_features(df)

print("\nNew shape:")
print(df_engineered.shape)

print("\nNew features:")
print([
    "AvgMonthlySpend",
    "ServiceCount",
    "IsMonthToMonth",
    "UsesElectronicCheck",
])

print("\nSample engineered data:")
print(
    df_engineered[
        [
            "tenure",
            "TotalCharges",
            "AvgMonthlySpend",
            "ServiceCount",
            "IsMonthToMonth",
            "UsesElectronicCheck",
        ]
    ].head()
)