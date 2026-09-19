from src.data_loader import load_data
from src.preprocessing import (
    split_features_target,
    encode_target,
)
from src.split_data import split_data


# Load data
df = load_data()

# Separate features and target
X, y = split_features_target(df)

# Convert target to 0/1
y = encode_target(y)

# Split data
(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
) = split_data(X, y)


print("Dataset sizes")
print("--------------------")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTarget distribution")
print("--------------------")

print("\nTraining:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\nValidation:")
print(y_val.value_counts())
print(y_val.value_counts(normalize=True))

print("\nTest:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))