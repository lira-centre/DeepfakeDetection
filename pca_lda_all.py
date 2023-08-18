import pandas as pd

# Read the Excel file
file_path = ''
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Drop columns with empty first element
df = df.dropna(axis=1, how='all', subset=[df.columns[0]])
df = df[df.iloc[:, 1].str.contains("Mammoth", case=False, na=False)]

# Print the resulting DataFrame
print(df)