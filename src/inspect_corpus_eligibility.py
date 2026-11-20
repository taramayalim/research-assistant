import pandas as pd
from pathlib import Path

files = {
    "Article metadata": Path("data/inventory/article_metadata.csv"),
    "PDF inventory": Path("data/inventory/pdf_file_inventory.csv"),
    "Article screening": Path("data/screening/article_screening.csv"),
}

for name, path in files.items():
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    candidates = [
        c for c in df.columns
        if any(word in c.lower() for word in
               ["status", "decision", "eligible", "include",
                "exclude", "screen", "reason", "relevance"])
    ]

    for col in candidates:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(dropna=False).head(20).to_string())

    if name == "Article screening":
        print("\nScreening records:")
        print(df.to_string(index=False, max_rows=35))

print("\n" + "=" * 70)
print("RESEARCH CORPUS REVIEW WORKBOOK")
print("=" * 70)

workbook = Path("data/inventory/research_corpus_review.xlsx")
xls = pd.ExcelFile(workbook)
print("Sheets:", xls.sheet_names)

for sheet in xls.sheet_names:
    df = pd.read_excel(workbook, sheet_name=sheet)
    print(f"\nSheet: {sheet}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    candidates = [
        c for c in df.columns
        if any(word in c.lower() for word in
               ["status", "decision", "eligible", "include",
                "exclude", "reason", "relevance"])
    ]

    for col in candidates:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(dropna=False).head(20).to_string())
