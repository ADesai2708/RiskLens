from src.data_loader import load_data
from src.preprocessing import (
    split_features_target,
    encode_target,
    get_feature_types,
)


df = load_data()

X, y = split_features_target(df)

y = encode_target(y)

numerical_features, categorical_features = get_feature_types(X)

print("Original dataset shape:", df.shape)

print("\nX shape:", X.shape)

print("\ny shape:", y.shape)

print("\nTarget values:")
print(y.value_counts())

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)