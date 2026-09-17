
# F7 RETAIL DECISION INTELLIGENCE
# DECISION ENGINE

import os
import re
from pathlib import Path

import pandas as pd

from src.retriever import (
    smart_retrieve,
    detect_query_intent,
    is_negative_review_query,
    summarize_low_rating_products,
)


# PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "integrated_retail_data.csv"
)


# GEMINI SETUP

try:
    from google import genai

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        client = genai.Client(
            api_key=GEMINI_API_KEY
        )
    else:
        client = None

except Exception:
    client = None


# LOAD RETAIL DATA

def load_retail_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Integrated retail dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    return df


# CATEGORY DETECTION

def detect_category(query):

    q = query.lower()

    categories = []

    if "food" in q:
        categories.append("FOODS")

    if "household" in q:
        categories.append("HOUSEHOLD")

    if "hobbies" in q or "hobby" in q:
        categories.append("HOBBIES")

    return categories


# PRODUCT ID DETECTION

def detect_m5_product_id(query):

    """
    Detect M5 product IDs such as:

    FOODS_3_090
    FOODS_3_586
    HOBBIES_1_371
    HOUSEHOLD_1_001
    """

    pattern = r"\b(?:FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+\b"

    match = re.search(
        pattern,
        query.upper()
    )

    if match:
        return match.group(0)

    return None


# PREPARE M5 SALES DATA

def get_m5_sales_data(df):

    if "source" not in df.columns:
        return pd.DataFrame()

    sales_df = df[
        df["source"]
        .astype(str)
        .str.lower()
        == "m5"
    ].copy()

    if len(sales_df) == 0:
        return sales_df

    if "sales" in sales_df.columns:

        sales_df["sales"] = pd.to_numeric(
            sales_df["sales"],
            errors="coerce"
        ).fillna(0)

    return sales_df


# OVERALL SALES ANALYSIS

def analyze_overall_sales(df):

    sales_df = get_m5_sales_data(df)

    if len(sales_df) == 0:

        return (
            "STRUCTURED SALES ANALYSIS\n"
            "No M5 sales records are available."
        )

    total_sales = sales_df["sales"].sum()

    sales_records = len(sales_df)

    unique_products = (
        sales_df["product_id"].nunique()
        if "product_id" in sales_df.columns
        else 0
    )

    average_sales = (
        total_sales / sales_records
        if sales_records > 0
        else 0
    )

    # CATEGORY SUMMARY

    category_summary = pd.DataFrame()

    if "category" in sales_df.columns:

        category_summary = (
            sales_df
            .groupby("category")
            .agg(
                total_sales=("sales", "sum"),
                products=("product_id", "nunique"),
                records=("sales", "count")
            )
            .reset_index()
            .sort_values(
                "total_sales",
                ascending=False
            )
        )

    # TOP PRODUCTS

    product_summary = pd.DataFrame()

    if "product_id" in sales_df.columns:

        product_summary = (
            sales_df
            .groupby("product_id")
            .agg(
                total_sales=("sales", "sum"),
                sales_records=("sales", "count")
            )
            .reset_index()
            .sort_values(
                "total_sales",
                ascending=False
            )
            .head(10)
        )

    # STORE SUMMARY

    store_summary = pd.DataFrame()

    if "store_id" in sales_df.columns:

        store_summary = (
            sales_df
            .groupby("store_id")
            .agg(
                total_sales=("sales", "sum"),
                sales_records=("sales", "count")
            )
            .reset_index()
            .sort_values(
                "total_sales",
                ascending=False
            )
            .head(10)
        )

    # STATE SUMMARY

    state_summary = pd.DataFrame()

    if "state_id" in sales_df.columns:

        state_summary = (
            sales_df
            .groupby("state_id")
            .agg(
                total_sales=("sales", "sum"),
                sales_records=("sales", "count")
            )
            .reset_index()
            .sort_values(
                "total_sales",
                ascending=False
            )
        )

    # BUILD EVIDENCE

    evidence = []

    evidence.append(
        "OVERALL SALES PERFORMANCE\n"
        f"Total recorded sales volume: "
        f"{total_sales:,.0f} units\n"
        f"Sales records analyzed: "
        f"{sales_records:,}\n"
        f"Unique products analyzed: "
        f"{unique_products:,}\n"
        f"Average sales per record: "
        f"{average_sales:,.2f} units\n"
        "\n"
        "IMPORTANT: M5 sales represent recorded sales "
        "quantity/units, NOT revenue."
    )

    # CATEGORY

    if len(category_summary) > 0:

        category_text = (
            "\nCATEGORY PERFORMANCE\n"
        )

        for _, row in category_summary.iterrows():

            category_text += (
                f"- {row['category']}: "
                f"{row['total_sales']:,.0f} units "
                f"({int(row['products']):,} products, "
                f"{int(row['records']):,} records)\n"
            )

        evidence.append(
            category_text
        )

    # TOP PRODUCTS

    if len(product_summary) > 0:

        product_text = (
            "\nTOP PRODUCTS BY UNIT SALES\n"
        )

        for index, row in product_summary.iterrows():

            product_text += (
                f"{product_summary.index.get_loc(index) + 1}. "
                f"{row['product_id']}: "
                f"{row['total_sales']:,.0f} units\n"
            )

        evidence.append(
            product_text
        )

    # STORES

    if len(store_summary) > 0:

        store_text = (
            "\nTOP STORES BY UNIT SALES\n"
        )

        for index, row in store_summary.iterrows():

            store_text += (
                f"{store_summary.index.get_loc(index) + 1}. "
                f"{row['store_id']}: "
                f"{row['total_sales']:,.0f} units\n"
            )

        evidence.append(
            store_text
        )

    # STATES

    if len(state_summary) > 0:

        state_text = (
            "\nSTATE PERFORMANCE\n"
        )

        for index, row in state_summary.iterrows():

            state_text += (
                f"{state_summary.index.get_loc(index) + 1}. "
                f"{row['state_id']}: "
                f"{row['total_sales']:,.0f} units\n"
            )

        evidence.append(
            state_text
        )

    # LIMITATIONS

    evidence.append(
        "\nDATA LIMITATIONS\n"
        "- Product price is not available.\n"
        "- Revenue is not available.\n"
        "- Profit margin is not available.\n"
        "- Promotion information is not available.\n"
        "- Inventory levels are not available.\n"
        "- Stockout information is not available.\n"
        "- Therefore, financial performance and stockout impact "
        "cannot be directly measured."
    )

    return "\n".join(evidence)


# PRODUCT SALES ANALYSIS

def analyze_product_sales(
    df,
    product_id
):

    sales_df = get_m5_sales_data(df)

    if len(sales_df) == 0:

        return (
            "STRUCTURED SALES ANALYSIS\n"
            "No M5 sales records are available."
        )

    product_df = sales_df[
        sales_df["product_id"]
        .astype(str)
        .str.upper()
        == product_id.upper()
    ].copy()

    if len(product_df) == 0:

        return (
            "STRUCTURED PRODUCT SALES ANALYSIS\n"
            f"No matching M5 sales records were found "
            f"for product {product_id}.\n"
            "\n"
            "Important: absence of matching records does "
            "NOT prove that the product had zero sales."
        )

    total_sales = product_df["sales"].sum()

    records = len(product_df)

    average_sales = (
        total_sales / records
        if records > 0
        else 0
    )

    # OVERALL RANK

    all_product_sales = (
        sales_df
        .groupby("product_id")["sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    rank = None

    if product_id in all_product_sales.index:

        rank = (
            all_product_sales
            .index
            .get_loc(product_id)
            + 1
        )

    # CATEGORY RANK

    category = None
    category_rank = None

    if "category" in product_df.columns:

        category_values = (
            product_df["category"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(category_values) > 0:

            category = category_values[0]

            category_df = sales_df[
                sales_df["category"]
                .astype(str)
                .str.upper()
                == category.upper()
            ]

            category_product_sales = (
                category_df
                .groupby("product_id")["sales"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            if product_id in category_product_sales.index:

                category_rank = (
                    category_product_sales
                    .index
                    .get_loc(product_id)
                    + 1
                )

    # STORE PERFORMANCE

    store_summary = pd.DataFrame()

    if "store_id" in product_df.columns:

        store_summary = (
            product_df
            .groupby("store_id")["sales"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    evidence = []

    evidence.append(
        "STRUCTURED PRODUCT SALES ANALYSIS"
    )

    evidence.append(
        f"Product: {product_id}"
    )

    evidence.append(
        f"Total recorded sales: "
        f"{total_sales:,.0f} units"
    )

    evidence.append(
        f"Sales records: {records:,}"
    )

    evidence.append(
        f"Average sales per record: "
        f"{average_sales:,.2f} units"
    )

    if rank is not None:

        evidence.append(
            f"Overall sales rank: #{rank}"
        )

    if category is not None:

        evidence.append(
            f"Category: {category}"
        )

    if category_rank is not None:

        evidence.append(
            f"Category sales rank: #{category_rank}"
        )

    if len(store_summary) > 0:

        evidence.append(
            "\nSTORE PERFORMANCE FOR PRODUCT"
        )

        for store, sales in store_summary.head(10).items():

            evidence.append(
                f"- {store}: "
                f"{sales:,.0f} units"
            )

    evidence.append(
        "\nIMPORTANT:\n"
        "These are recorded sales quantities, "
        "not revenue or profit."
    )

    evidence.append(
        "\nMISSING BUSINESS DATA:\n"
        "Price, revenue, margin, promotion, inventory "
        "and stockout data are unavailable."
    )

    return "\n".join(evidence)


# CATEGORY SALES ANALYSIS

def analyze_category_sales(
    df,
    category
):

    sales_df = get_m5_sales_data(df)

    if len(sales_df) == 0:

        return (
            "STRUCTURED CATEGORY SALES ANALYSIS\n"
            "No M5 sales records are available."
        )

    category_df = sales_df[
        sales_df["category"]
        .astype(str)
        .str.upper()
        == category.upper()
    ].copy()

    if len(category_df) == 0:

        return (
            "STRUCTURED CATEGORY SALES ANALYSIS\n"
            f"No sales records were found for "
            f"category {category}."
        )

    total_sales = category_df["sales"].sum()

    records = len(category_df)

    unique_products = (
        category_df["product_id"].nunique()
        if "product_id" in category_df.columns
        else 0
    )

    average_sales = (
        total_sales / records
        if records > 0
        else 0
    )

    # CATEGORY SHARE

    all_sales = sales_df["sales"].sum()

    category_share = (
        total_sales / all_sales * 100
        if all_sales > 0
        else 0
    )

    # PRODUCT SUMMARY

    product_summary = (
        category_df
        .groupby("product_id")["sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    # SUBCATEGORY

    subcategory_summary = pd.DataFrame()

    if "product_id" in category_df.columns:

        product_series = (
            category_df["product_id"]
            .astype(str)
        )

        subcategory = (
            product_series
            .str.split("_")
            .str[1]
        )

        temp = category_df.copy()

        temp["subcategory"] = subcategory.values

        subcategory_summary = (
            temp
            .groupby("subcategory")["sales"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    # TOP / LOW PRODUCTS

    top_products = product_summary.head(5)

    low_products = (
        product_summary
        .sort_values(
            ascending=True
        )
        .head(5)
    )

    evidence = []

    evidence.append(
        "STRUCTURED CATEGORY SALES ANALYSIS"
    )

    evidence.append(
        f"Category: {category.upper()}"
    )

    evidence.append(
        f"Total recorded sales: "
        f"{total_sales:,.0f} units"
    )

    evidence.append(
        f"Sales records: {records:,}"
    )

    evidence.append(
        f"Unique products: {unique_products:,}"
    )

    evidence.append(
        f"Share of total M5 unit sales: "
        f"{category_share:.1f}%"
    )

    evidence.append(
        f"Average sales per record: "
        f"{average_sales:,.2f} units"
    )

    if len(subcategory_summary) > 0:

        evidence.append(
            "\nSUBCATEGORY PERFORMANCE"
        )

        for subcat, sales in subcategory_summary.items():

            evidence.append(
                f"- {category.upper()}_{subcat}: "
                f"{sales:,.0f} units"
            )

    evidence.append(
        "\nTOP PRODUCTS"
    )

    for product, sales in top_products.items():

        evidence.append(
            f"- {product}: "
            f"{sales:,.0f} units"
        )

    evidence.append(
        "\nLOWEST-VOLUME PRODUCTS"
    )

    for product, sales in low_products.items():

        evidence.append(
            f"- {product}: "
            f"{sales:,.0f} units"
        )

    evidence.append(
        "\nIMPORTANT LIMITATION:\n"
        "Low sales do not prove poor product quality, "
        "lack of demand, stockouts, or pricing problems. "
        "Those factors require additional data."
    )

    return "\n".join(evidence)


# STORE SALES ANALYSIS

def analyze_store_sales(df):

    sales_df = get_m5_sales_data(df)

    if len(sales_df) == 0:

        return (
            "STRUCTURED STORE SALES ANALYSIS\n"
            "No M5 sales records are available."
        )

    if "store_id" not in sales_df.columns:

        return (
            "STRUCTURED STORE SALES ANALYSIS\n"
            "Store information is not available."
        )

    store_summary = (
        sales_df
        .groupby("store_id")
        .agg(
            total_sales=("sales", "sum"),
            sales_records=("sales", "count")
        )
        .reset_index()
        .sort_values(
            "total_sales",
            ascending=False
        )
    )

    evidence = [
        "STRUCTURED STORE SALES ANALYSIS",
        f"Total stores analyzed: {len(store_summary):,}",
        f"Total M5 sales records: {len(sales_df):,}",
        "\nSTORE RANKING"
    ]

    for i, (_, row) in enumerate(
        store_summary.head(10).iterrows(),
        start=1
    ):

        evidence.append(
            f"{i}. {row['store_id']}: "
            f"{row['total_sales']:,.0f} units"
        )

    if len(store_summary) >= 2:

        best_store = store_summary.iloc[0]
        second_store = store_summary.iloc[1]

        difference = (
            best_store["total_sales"]
            - second_store["total_sales"]
        )

        percentage = (
            difference
            / second_store["total_sales"]
            * 100
            if second_store["total_sales"] != 0
            else 0
        )

        evidence.append(
            "\nLEADING STORE COMPARISON"
        )

        evidence.append(
            f"{best_store['store_id']} leads "
            f"{second_store['store_id']} by "
            f"{difference:,.0f} units "
            f"({percentage:.1f}%)."
        )

    evidence.append(
        "\nIMPORTANT:\n"
        "Store sales are recorded unit quantities, "
        "not revenue."
    )

    return "\n".join(evidence)


# STATE SALES ANALYSIS

def analyze_state_sales(df):

    sales_df = get_m5_sales_data(df)

    if len(sales_df) == 0:

        return (
            "STRUCTURED STATE SALES ANALYSIS\n"
            "No M5 sales records are available."
        )

    if "state_id" not in sales_df.columns:

        return (
            "STRUCTURED STATE SALES ANALYSIS\n"
            "State information is not available."
        )

    state_summary = (
        sales_df
        .groupby("state_id")
        .agg(
            total_sales=("sales", "sum"),
            sales_records=("sales", "count")
        )
        .reset_index()
        .sort_values(
            "total_sales",
            ascending=False
        )
    )

    evidence = [
        "STRUCTURED STATE SALES ANALYSIS",
        f"Total states analyzed: {len(state_summary):,}",
        "\nSTATE RANKING"
    ]

    total_sales = sales_df["sales"].sum()

    for i, (_, row) in enumerate(
        state_summary.iterrows(),
        start=1
    ):

        share = (
            row["total_sales"]
            / total_sales
            * 100
            if total_sales > 0
            else 0
        )

        evidence.append(
            f"{i}. {row['state_id']}: "
            f"{row['total_sales']:,.0f} units "
            f"({share:.1f}% of total)"
        )

    return "\n".join(evidence)


# SALES ROUTER

def build_sales_evidence(
    query,
    df
):

    q = query.lower()

    product_id = detect_m5_product_id(query)

    categories = detect_category(query)

    # EXACT PRODUCT QUERY

    if product_id:

        print(
            f"[SALES ROUTER] Exact product detected: "
            f"{product_id}"
        )

        return analyze_product_sales(
            df,
            product_id
        )

    # CATEGORY QUERY

    if categories:

        # If the query asks to improve/increase category sales
        if (
            "improve" in q
            or "increase" in q
            or "grow" in q
            or "boost" in q
            or "performance" in q
        ):

            category_evidence = []

            for category in categories:

                category_evidence.append(
                    analyze_category_sales(
                        df,
                        category
                    )
                )

            return "\n\n".join(
                category_evidence
            )

        # Otherwise still provide category sales
        return "\n\n".join(
            analyze_category_sales(
                df,
                category
            )
            for category in categories
        )

    # STORE QUERY

    if (
        "store" in q
        or "stores" in q
        or "location" in q
        or "locations" in q
    ):

        return analyze_store_sales(df)

    # STATE QUERY

    if (
        "state" in q
        or "states" in q
        or "region" in q
        or "regional" in q
    ):

        return analyze_state_sales(df)

    # PRODUCT RANKING QUERY

    if (
        "which products" in q
        or "highest sales" in q
        or "top products" in q
        or "best selling" in q
        or "best-selling" in q
        or "most sold" in q
    ):

        sales_df = get_m5_sales_data(df)

        if len(sales_df) == 0:

            return (
                "STRUCTURED PRODUCT SALES ANALYSIS\n"
                "No M5 sales records are available."
            )

        product_summary = (
            sales_df
            .groupby("product_id")["sales"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        evidence = [
            "STRUCTURED PRODUCT SALES RANKING",
            f"M5 sales records analyzed: {len(sales_df):,}",
            f"Unique products analyzed: "
            f"{sales_df['product_id'].nunique():,}",
            "\nTOP PRODUCTS BY RECORDED UNIT SALES"
        ]

        for i, (product, sales) in enumerate(
            product_summary.items(),
            start=1
        ):

            evidence.append(
                f"{i}. {product}: "
                f"{sales:,.0f} units"
            )

        evidence.append(
            "\nIMPORTANT:\n"
            "Sales values represent recorded unit quantities, "
            "not revenue."
        )

        return "\n".join(evidence)

    # OVERALL SALES QUERY

    return analyze_overall_sales(df)


# REVIEW ANALYSIS

def analyze_reviews(df, query):

    if "source" not in df.columns:

        return (
            "STRUCTURED REVIEW ANALYSIS\n"
            "Amazon review data is not available."
        )

    amazon_df = df[
        df["source"]
        .astype(str)
        .str.lower()
        == "amazon"
    ].copy()

    if len(amazon_df) == 0:

        return (
            "STRUCTURED REVIEW ANALYSIS\n"
            "No Amazon review records are available."
        )

    if "rating" not in amazon_df.columns:

        return (
            "STRUCTURED REVIEW ANALYSIS\n"
            "Review rating information is not available."
        )

    amazon_df["rating"] = pd.to_numeric(
        amazon_df["rating"],
        errors="coerce"
    )

    valid = amazon_df[
        amazon_df["rating"].notna()
    ].copy()

    if len(valid) == 0:

        return (
            "STRUCTURED REVIEW ANALYSIS\n"
            "No valid review ratings are available."
        )

    avg_rating = valid["rating"].mean()

    total_reviews = len(valid)

    rating_counts = (
        valid["rating"]
        .value_counts()
        .sort_index()
    )

    low_reviews = len(
        valid[
            valid["rating"] <= 2
        ]
    )

    high_reviews = len(
        valid[
            valid["rating"] >= 4
        ]
    )

    evidence = [
        "STRUCTURED CUSTOMER REVIEW ANALYSIS",
        f"Total valid Amazon reviews: {total_reviews:,}",
        f"Average rating: {avg_rating:.2f}/5",
        f"High-rated reviews (4-5): {high_reviews:,}",
        f"Low-rated reviews (1-2): {low_reviews:,}",
        "\nRATING DISTRIBUTION"
    ]

    for rating, count in rating_counts.items():

        evidence.append(
            f"- {rating:.0f} stars: "
            f"{int(count):,}"
        )

    return "\n".join(evidence)


# PRODUCT REVIEW ANALYSIS

def analyze_product_reviews(
    df,
    query
):

    if "source" not in df.columns:

        return (
            "STRUCTURED PRODUCT REVIEW ANALYSIS\n"
            "Amazon review data is unavailable."
        )

    amazon_df = df[
        df["source"]
        .astype(str)
        .str.lower()
        == "amazon"
    ].copy()

    if len(amazon_df) == 0:

        return (
            "STRUCTURED PRODUCT REVIEW ANALYSIS\n"
            "No Amazon reviews are available."
        )

    # PRODUCT SEARCH

    search_terms = query.lower()

    if "product_name" in amazon_df.columns:

        names = (
            amazon_df["product_name"]
            .dropna()
            .astype(str)
        )

        matches = names[
            names.str.lower().apply(
                lambda x: any(
                    term in x
                    for term in [
                        "fire tv",
                        "echo",
                        "kindle",
                        "tablet"
                    ]
                    if term in search_terms
                )
            )
        ]

    else:

        matches = pd.Series(
            dtype=str
        )

    # FIRE TV

    if "fire tv" in search_terms:

        if "product_name" in amazon_df.columns:

            fire_df = amazon_df[
                amazon_df["product_name"]
                .astype(str)
                .str.lower()
                .str.contains(
                    "fire tv",
                    na=False
                )
            ].copy()

        else:

            fire_df = pd.DataFrame()

        if len(fire_df) > 0:

            if "rating" in fire_df.columns:

                fire_df["rating"] = pd.to_numeric(
                    fire_df["rating"],
                    errors="coerce"
                )

            total = len(fire_df)

            avg_rating = (
                fire_df["rating"].mean()
                if "rating" in fire_df.columns
                else None
            )

            evidence = [
                "STRUCTURED PRODUCT REVIEW ANALYSIS",
                "Product: Amazon Fire TV",
                f"Reviews analyzed: {total:,}"
            ]

            if avg_rating is not None:

                evidence.append(
                    f"Average rating: "
                    f"{avg_rating:.2f}/5"
                )

            if "rating" in fire_df.columns:

                for rating in [5, 4, 3, 2, 1]:

                    count = len(
                        fire_df[
                            fire_df["rating"]
                            == rating
                        ]
                    )

                    evidence.append(
                        f"- {rating} stars: "
                        f"{count:,}"
                    )

            evidence.append(
                "\nSALES DATA LIMITATION:\n"
                "No matching M5 product-level sales records "
                "were found for Amazon Fire TV. This does "
                "NOT prove zero sales; it means the available "
                "M5 sales data does not contain a matching "
                "product record."
            )

            return "\n".join(evidence)

    return analyze_reviews(
        df,
        query
    )


# SEMANTIC RAG EVIDENCE

def get_semantic_rag_evidence(query):

    try:

        results = smart_retrieve(
            query
        )

    except Exception as e:

        return (
            "RAG RETRIEVAL ERROR\n"
            f"{str(e)}"
        )

    if not results:

        return (
            "SEMANTIC RAG EVIDENCE\n"
            "No relevant semantic documents were retrieved."
        )

    evidence = []

    evidence.append(
        "SEMANTIC RAG EVIDENCE"
    )

    # Chroma result format
    if isinstance(results, dict):

        documents = results.get(
            "documents",
            []
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        if documents and isinstance(
            documents[0],
            list
        ):

            documents = documents[0]

        if metadatas and isinstance(
            metadatas[0],
            list
        ):

            metadatas = metadatas[0]

        for i, doc in enumerate(
            documents[:8]
        ):

            metadata = (
                metadatas[i]
                if i < len(metadatas)
                else {}
            )

            source = metadata.get(
                "source",
                "unknown"
            )

            evidence.append(
                f"\nEvidence {i + 1} "
                f"(source={source}):\n"
                f"{doc}"
            )

    else:

        for i, item in enumerate(
            results[:8]
        ):

            evidence.append(
                f"\nEvidence {i + 1}:\n"
                f"{item}"
            )

    return "\n".join(evidence)


# CLEAN GEMINI OUTPUT

def clean_model_output(answer):

    if not answer:
        return ""

    answer = str(answer)

    # REMOVE STREAMLIT LOCALHOST SVG ARTIFACTS

    answer = re.sub(
        r'\[svg\]\(\s*https?://localhost(?::\d+)?/#[^)]+\)',
        '',
        answer,
        flags=re.IGNORECASE
    )

    # REMOVE GENERIC SVG MARKDOWN ARTIFACTS

    answer = re.sub(
        r'\[svg\]\([^)]+\)',
        '',
        answer,
        flags=re.IGNORECASE
    )

    # REMOVE RAW HTML ANCHORS IF GENERATED

    answer = re.sub(
        r'<a[^>]*href=["\']?http://localhost:[^"\'>\s]+["\']?[^>]*>.*?</a>',
        '',
        answer,
        flags=re.IGNORECASE | re.DOTALL
    )

    # REMOVE REPEATED EMPTY MARKDOWN HEADINGS

    answer = re.sub(
        r'\n\s*\n\s*\n+',
        '\n\n',
        answer
    )

    return answer.strip()


# MAIN BUSINESS DECISION ENGINE

def generate_business_decision(query):

    print(
        "\n" + "=" * 70
    )

    print(
        "RETAIL DECISION ENGINE"
    )

    print(
        "=" * 70
    )

    print(
        f"User Query: {query}"
    )

    # STEP 1 — LOAD DATA

    print(
        "[1] Loading integrated retail data..."
    )

    df = load_retail_data()

    print(
        f"Dataset shape: {df.shape}"
    )

    # STEP 2 — QUERY INTENT

    try:

        intent = detect_query_intent(
            query
        )

    except Exception:

        intent = "general"

    print(
        f"[2] Detected intent: {intent}"
    )

    q = query.lower()

    # STEP 3 — DETERMINE WHETHER SALES QUESTION

    sales_keywords = [
        "sales",
        "sold",
        "selling",
        "sell",
        "revenue",
        "performance",
        "increase sales",
        "improve sales",
        "boost sales",
        "highest sales",
        "top sales"
    ]

    is_sales_query = any(
        keyword in q
        for keyword in sales_keywords
    )

    # STEP 4 — STRUCTURED EVIDENCE

    sales_evidence = ""

    review_evidence = ""

    if is_sales_query:

        print(
            "[3] Building structured sales evidence..."
        )

        sales_evidence = build_sales_evidence(
            query,
            df
        )

    # REVIEW QUESTION

    is_review_query = (
        "review" in q
        or "reviews" in q
        or "rating" in q
        or "ratings" in q
        or "customer feedback" in q
        or is_negative_review_query(query)
    )

    if is_review_query:

        print(
            "[3] Building structured review evidence..."
        )

        review_evidence = analyze_product_reviews(
            df,
            query
        )

    # GENERAL QUERY

    if not is_sales_query and not is_review_query:

        print(
            "[3] Building general retail evidence..."
        )

        sales_evidence = analyze_overall_sales(
            df
        )

    # STEP 5 — SEMANTIC RAG

    print(
        "[4] Retrieving semantic RAG evidence..."
    )

    semantic_evidence = get_semantic_rag_evidence(
        query
    )

    # STEP 6 — COMBINE EVIDENCE

    evidence_sections = []

    if sales_evidence:

        evidence_sections.append(
            sales_evidence
        )

    if review_evidence:

        evidence_sections.append(
            review_evidence
        )

    if semantic_evidence:

        evidence_sections.append(
            semantic_evidence
        )

    final_evidence = "\n\n".join(
        evidence_sections
    )

    # STEP 7 — GEMINI CHECK

    if client is None:

        # Return structured evidence if Gemini
        # is not configured.

        return clean_model_output(
            final_evidence
        )

    # STEP 8 — GEMINI PROMPT

    prompt = f"""
You are an intelligent retail decision-support assistant.

User question:

{query}

The following evidence comes from the retail
knowledge base and structured analytics system:

============================================================
RETAIL EVIDENCE
============================================================

{final_evidence}

============================================================
IMPORTANT RULES
============================================================

1. Use ONLY the provided evidence.

2. NEVER invent products, sales numbers, ratings,
   stores, states, categories, customer feedback,
   prices, inventory levels, revenue, profit,
   promotions, or trends.

3. For sales questions, prioritize STRUCTURED SALES
   ANALYSIS over semantic retrieval results.

4. For review questions, prioritize STRUCTURED CUSTOMER
   REVIEW ANALYSIS.

5. SEMANTIC RAG EVIDENCE is supporting evidence for
   qualitative information.

6. If the requested product or category does not exist
   in the evidence, explicitly say that the current
   dataset does not contain sufficient information.

7. Never use an unrelated category as a substitute.

   For example:
   - HOBBIES is not laptops.
   - FOODS is not smartphones.
   - HOUSEHOLD is not USB products.

8. Distinguish clearly between:

   - Dataset facts
   - Business interpretation
   - Recommendations

9. Do not claim that one factor caused another unless
   the evidence proves it.

10. When recommending a strategy, make clear when it is
    a business hypothesis rather than a proven cause.

11. If important data is missing, identify the missing
    data instead of inventing it.

12. M5 sales represent recorded SALES QUANTITY / UNITS.
    They are NOT revenue.

13. Do not describe M5 sales as money, dollars, rupees,
    revenue, or profit.

14. If price data is unavailable, explicitly state that
    revenue cannot be calculated.

15. If inventory or stockout data is unavailable,
    do not claim that stockouts caused low sales.

16. For Amazon products that do not have a corresponding
    M5 product record, distinguish customer-review evidence
    from M5 sales evidence.

17. Absence of a matching M5 product record does NOT mean
    zero sales.

18. Do NOT generate Streamlit HTML.

19. Do NOT generate SVG links.

20. Do NOT generate localhost URLs.

21. Do NOT generate text such as:

    [svg](http://localhost:8501/...)

22. Return ONLY clean Markdown.

============================================================
RESPONSE FORMAT
============================================================

### 1. BUSINESS FINDING

Answer the user's question directly in 1-2 paragraphs.

### 2. EVIDENCE

Provide the most important numerical and retrieved evidence
using concise bullet points.

### 3. BUSINESS INTERPRETATION

Explain what the evidence means for the retailer.

### 4. RECOMMENDATION

Give practical recommendations based only on the evidence.

Clearly label recommendations as business hypotheses when
the available data does not prove causality.

### 5. CONFIDENCE

Give High, Medium, or Low confidence and explain why.

============================================================
STYLE
============================================================

Keep the response concise, clear, professional,
and business-oriented.

Do not add extra sections.

Do not add HTML.

Do not add SVG.

Do not add localhost links.
"""

    # STEP 9 — SEND TO GEMINI

    print(
        "[5] Sending evidence to Gemini..."
    )

    try:

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        answer = interaction.output_text

    except Exception as e:

        print(
            f"Gemini error: {e}"
        )

        # Fallback to structured evidence
        answer = final_evidence

    # STEP 10 — CLEAN FINAL OUTPUT

    answer = clean_model_output(
        answer
    )

    # STEP 11 — FINAL OUTPUT

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL BUSINESS ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )

    return answer


# MAIN

if __name__ == "__main__":

    query = input(
        "\nEnter your retail business question: "
    ).strip()

    if query:

        generate_business_decision(
            query
        )

    else:

        print(
            "Please enter a business question."
        )