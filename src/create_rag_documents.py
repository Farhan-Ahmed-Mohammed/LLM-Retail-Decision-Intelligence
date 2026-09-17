import pandas as pd
import json
from pathlib import Path


# PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "integrated_retail_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "rag_documents.jsonl"


# LOAD DATA

print("=" * 60)
print("LOADING INTEGRATED RETAIL DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE, low_memory=False)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# HELPER FUNCTION

def clean_value(value):
    """Convert missing values into a readable string."""
    if pd.isna(value):
        return "Unknown"

    return str(value).strip()


# CREATE RAG DOCUMENTS

documents = []

print("\nCreating RAG documents...")


# AMAZON REVIEW DOCUMENTS

amazon_df = df[df["source"].astype(str).str.lower() == "amazon"]

print(f"\nAmazon records: {len(amazon_df)}")

for _, row in amazon_df.iterrows():

    product_id = clean_value(row["product_id"])
    product_name = clean_value(row["product_name"])
    category = clean_value(row["category"])
    brand = clean_value(row["brand"])
    customer_id = clean_value(row["customer_id"])
    date = clean_value(row["date"])
    rating = clean_value(row["rating"])
    review_text = clean_value(row["review_text"])

    text = f"""
Retail Review Information

Source: Amazon Reviews
Entity Type: Product Review

Product ID: {product_id}
Product Name: {product_name}
Category: {category}
Brand: {brand}
Customer ID: {customer_id}
Review Date: {date}
Rating: {rating}

Customer Review:
{review_text}

This document represents an Amazon customer review associated with the
specified product. It can be used to understand customer sentiment,
product feedback, ratings, and review patterns.
""".strip()

    documents.append({
        "document_id": f"amazon_review_{product_id}_{len(documents)}",
        "source": "Amazon",
        "entity_type": "review",
        "product_id": product_id,
        "text": text
    })


# M5 SALES DOCUMENTS

m5_df = df[df["source"].astype(str).str.lower() == "m5"]

print(f"M5 records: {len(m5_df)}")

for _, row in m5_df.iterrows():

    product_id = clean_value(row["product_id"])
    product_name = clean_value(row["product_name"])
    category = clean_value(row["category"])
    store_id = clean_value(row["store_id"])
    state_id = clean_value(row["state_id"])
    date = clean_value(row["date"])
    sales = clean_value(row["sales"])

    text = f"""
Retail Sales Information

Source: M5 Forecasting Dataset
Entity Type: Sales

Product ID: {product_id}
Product Name: {product_name}
Category: {category}
Store: {store_id}
State: {state_id}
Date: {date}
Sales: {sales}

This document represents sales information for a retail product.
It can be used to analyze product demand, sales performance,
store-level performance, state-level performance, and temporal
sales patterns.
""".strip()

    documents.append({
        "document_id": f"m5_sales_{product_id}_{len(documents)}",
        "source": "M5",
        "entity_type": "sales",
        "product_id": product_id,
        "text": text
    })


# INSTACART PURCHASE DOCUMENTS

instacart_df = df[
    df["source"].astype(str).str.lower() == "instacart"
]

print(f"Instacart records: {len(instacart_df)}")

for _, row in instacart_df.iterrows():

    product_id = clean_value(row["product_id"])
    product_name = clean_value(row["product_name"])
    category = clean_value(row["category"])
    customer_id = clean_value(row["customer_id"])
    date = clean_value(row["date"])
    purchase = clean_value(row["purchase"])

    text = f"""
Retail Purchase Information

Source: Instacart Basket Analysis Dataset
Entity Type: Purchase

Product ID: {product_id}
Product Name: {product_name}
Category: {category}
Customer ID: {customer_id}
Purchase Date: {date}
Purchase Information: {purchase}

This document represents a customer purchase record.
It can be used to analyze customer purchasing behavior,
product demand, repeat purchases, and purchase patterns.
""".strip()

    documents.append({
        "document_id": f"instacart_purchase_{product_id}_{len(documents)}",
        "source": "Instacart",
        "entity_type": "purchase",
        "product_id": product_id,
        "text": text
    })


# SAVE DOCUMENTS

print("\n" + "=" * 60)
print("SAVING RAG DOCUMENTS")
print("=" * 60)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for document in documents:
        f.write(json.dumps(document, ensure_ascii=False) + "\n")


# SUMMARY

print(f"Total RAG documents: {len(documents)}")

print("\nDocuments by source:")

source_counts = {}

for document in documents:
    source = document["source"]
    source_counts[source] = source_counts.get(source, 0) + 1

for source, count in source_counts.items():
    print(f"{source:15}: {count}")


print("\nRAG documents saved successfully!")

print(OUTPUT_FILE)

print("=" * 60)