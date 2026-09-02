from pathlib import Path
import pandas as pd


project_folder = Path(__file__).resolve().parent


file_path = project_folder / "clear_panel_with_one_year_fd_labels.xlsx"


print("Excel file:", file_path)
print("File exists:", file_path.exists())


df = pd.read_excel(
    file_path,
    sheet_name="Clean_Panel",
    na_values=["NaN", "#N/A N/A", "#N/A", "N/A"]
)


print("\nData shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nOne-year target distribution:")
print(df["FD_t_plus_1"].value_counts(dropna=False))

print("\nValid targets:")
print(df["FD_t_plus_1"].notna().sum())

model_df = df[df["FD_t_plus_1"].notna()].copy()

model_df["FD_t_plus_1"] = (
    model_df["FD_t_plus_1"].astype(int)
)

print("\nModel data shape:")
print(model_df.shape)

print("\nTarget distribution:")
print(model_df["FD_t_plus_1"].value_counts())

print("\nDistress rate:")
print(model_df["FD_t_plus_1"].mean())