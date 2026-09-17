import os
import re
import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer


# PATHS

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "integrated_retail_data.csv"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "chroma_db"
)

COLLECTION_NAME = "retail_documents"

MODEL_NAME = "all-MiniLM-L6-v2"


# LOAD EMBEDDING MODEL

print("\n" + "=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully!")


# CONNECT TO CHROMADB

print("\n" + "=" * 60)
print("CONNECTING TO CHROMADB")
print("=" * 60)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB connected successfully!")
print(
    f"Documents available: {collection.count()}"
)


# LOAD RETAIL DATA

print("\n" + "=" * 60)
print("LOADING RETAIL DATA FOR SMART FILTERING")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print(
    f"Retail records loaded: {len(df)}"
)


# QUERY INTENT DETECTION

def detect_query_intent(query):

    query_lower = query.lower()

    review_keywords = [
        "review",
        "reviews",
        "rating",
        "ratings",
        "feedback",
        "complaint",
        "complaints",
        "negative",
        "poor",
        "bad",
        "unhappy",
        "dissatisfied"
    ]

    sales_keywords = [
        "sales",
        "sell",
        "selling",
        "sold",
        "revenue",
        "top selling",
        "highest sales",
        "best selling",
        "increase sales",
        "boost sales",
        "improve sales"
    ]

    purchase_keywords = [
        "purchase",
        "purchases",
        "bought",
        "buy",
        "buying",
        "orders",
        "frequently purchased"
    ]

    for word in review_keywords:
        if word in query_lower:
            return "review"

    for word in sales_keywords:
        if word in query_lower:
            return "sales"

    for word in purchase_keywords:
        if word in query_lower:
            return "purchase"

    return "general"


# SOURCE DETECTION

def detect_source(query):

    query_lower = query.lower()

    intent = detect_query_intent(query)

    if intent == "review":
        return "Amazon"

    if intent == "sales":
        return "M5"

    if intent == "purchase":
        return "Instacart"

    if "amazon" in query_lower:
        return "Amazon"

    if "m5" in query_lower:
        return "M5"

    if "instacart" in query_lower:
        return "Instacart"

    return None


# PRODUCT / CATEGORY KEYWORD DETECTION

def detect_product_keywords(query):

    query_lower = query.lower()

    keywords = [
        "laptop",
        "laptops",
        "phone",
        "phones",
        "smartphone",
        "smartphones",
        "tablet",
        "tablets",
        "electronics",
        "electronic",
        "usb",
        "tv",
        "television",
        "headphone",
        "headphones",
        "camera",
        "cameras",
        "computer",
        "computers",
        "food",
        "foods",
        "household",
        "hobbies"
    ]

    found = []

    for keyword in keywords:

        if keyword in query_lower:
            found.append(keyword)

    return found


# EXACT PRODUCT ID DETECTION

def detect_product_id(query):

    """
    Detect exact M5 product IDs such as:

        HOBBIES_1_371
        FOODS_3_090
        HOUSEHOLD_1_001

    Also works with Amazon / Instacart IDs when they
    appear explicitly in the query.
    """

    # M5-style product ID
    m5_pattern = r"\b(?:FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+\b"

    match = re.search(
        m5_pattern,
        query.upper()
    )

    if match:
        return match.group(0)

    # Generic product_id pattern
    generic_pattern = r"\b[A-Z0-9]+_[A-Z0-9]+_[A-Z0-9]+\b"

    match = re.search(
        generic_pattern,
        query.upper()
    )

    if match:
        candidate = match.group(0)

        if candidate in set(
            df["product_id"]
            .dropna()
            .astype(str)
            .str.upper()
        ):
            return candidate

    return None


# NEGATIVE REVIEW DETECTION

def is_negative_review_query(query):

    query_lower = query.lower()

    negative_words = [
        "poor",
        "bad",
        "negative",
        "worst",
        "complaint",
        "complaints",
        "unhappy",
        "dissatisfied",
        "low rating",
        "low ratings",
        "one star",
        "two star",
        "1 star",
        "2 star"
    ]

    for word in negative_words:

        if word in query_lower:
            return True

    return False


# BUILD CHROMA FILTER

def build_chroma_filter(
    query,
    source=None
):

    filters = []

    if source:
        filters.append(
            {
                "source": source
            }
        )

    # Exact product ID gets highest priority
    product_id = detect_product_id(query)

    if product_id:
        filters.append(
            {
                "product_id": product_id
            }
        )

    if len(filters) == 0:
        return None

    if len(filters) == 1:
        return filters[0]

    return {
        "$and": filters
    }


# EXACT PRODUCT RETRIEVAL

def retrieve_exact_product(
    product_id,
    top_k=5
):

    """
    Retrieve records belonging specifically to
    the requested product ID.

    This is more reliable than semantic similarity
    when the user explicitly gives a product ID.
    """

    try:

        results = collection.get(
            where={
                "product_id": product_id
            },
            limit=top_k,
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = results.get(
            "documents",
            []
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        if not documents:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]]
            }

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [[]],
            "ids": [
                results.get("ids", [])
            ]
        }

    except Exception as e:

        print(
            f"Exact product retrieval error: {e}"
        )

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]]
        }


# PRODUCT MATCHING

def product_matches_query(
    row,
    query
):

    query_lower = query.lower()

    product_name = str(
        row.get("product_name", "")
    ).lower()

    category = str(
        row.get("category", "")
    ).lower()

    product_id = str(
        row.get("product_id", "")
    ).lower()

    product_keywords = detect_product_keywords(
        query
    )

    # No product keyword means no additional
    # product filtering is required.
    if not product_keywords:
        return True

    for keyword in product_keywords:

        if (
            keyword in product_name
            or keyword in category
            or keyword in product_id
        ):
            return True

    return False


# FILTER SEMANTIC RESULTS

def filter_retrieved_results(
    results,
    query
):

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    ids = results.get(
        "ids",
        [[]]
    )[0]

    if not documents:
        return results

    product_id = detect_product_id(query)

    # Exact product ID

    if product_id:

        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []
        filtered_ids = []

        for i, metadata in enumerate(
            metadatas
        ):

            metadata_product_id = str(
                metadata.get(
                    "product_id",
                    ""
                )
            ).upper()

            if metadata_product_id == product_id.upper():

                filtered_documents.append(
                    documents[i]
                )

                filtered_metadatas.append(
                    metadata
                )

                if i < len(distances):
                    filtered_distances.append(
                        distances[i]
                    )

                if i < len(ids):
                    filtered_ids.append(
                        ids[i]
                    )

        return {
            "documents": [
                filtered_documents
            ],
            "metadatas": [
                filtered_metadatas
            ],
            "distances": [
                filtered_distances
            ],
            "ids": [
                filtered_ids
            ]
        }

    # Product/category keyword filtering

    if detect_product_keywords(query):

        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []
        filtered_ids = []

        for i, metadata in enumerate(
            metadatas
        ):

            row = {
                "product_id": metadata.get(
                    "product_id",
                    ""
                ),
                "product_name": metadata.get(
                    "product_name",
                    ""
                ),
                "category": metadata.get(
                    "category",
                    ""
                )
            }

            if product_matches_query(
                row,
                query
            ):

                filtered_documents.append(
                    documents[i]
                )

                filtered_metadatas.append(
                    metadata
                )

                if i < len(distances):
                    filtered_distances.append(
                        distances[i]
                    )

                if i < len(ids):
                    filtered_ids.append(
                        ids[i]
                    )

        return {
            "documents": [
                filtered_documents
            ],
            "metadatas": [
                filtered_metadatas
            ],
            "distances": [
                filtered_distances
            ],
            "ids": [
                filtered_ids
            ]
        }

    return results


# SEMANTIC RETRIEVAL

def semantic_retrieve(
    query,
    top_k=5
):

    # Query analysis

    intent = detect_query_intent(
        query
    )

    source = detect_source(
        query
    )

    product_keywords = detect_product_keywords(
        query
    )

    product_id = detect_product_id(
        query
    )

    print("\n" + "=" * 60)
    print("QUERY ANALYSIS")
    print("=" * 60)

    print(
        f"Detected intent : {intent}"
    )

    print(
        f"Detected source : {source}"
    )

    print(
        f"Detected product/category : "
        f"{product_keywords if product_keywords else None}"
    )

    if product_id:

        print(
            f"Detected exact product ID : "
            f"{product_id}"
        )

    # EXACT PRODUCT ID RETRIEVAL

    if product_id:

        print(
            "\nUsing EXACT PRODUCT ID retrieval..."
        )

        exact_results = retrieve_exact_product(
            product_id,
            top_k
        )

        if exact_results["documents"][0]:

            return exact_results

        print(
            "Exact product ID not found in ChromaDB."
        )

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]]
        }

    # Create query embedding

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Retrieve more candidates first

    candidate_k = max(
        top_k * 10,
        50
    )

    # Build metadata filter

    chroma_filter = build_chroma_filter(
        query,
        source
    )

    # ChromaDB query

    if chroma_filter:

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=candidate_k,
            where=chroma_filter,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    else:

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=candidate_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    # Product/category filtering

    results = filter_retrieved_results(
        results,
        query
    )

    # Limit final results

    results["documents"][0] = (
        results["documents"][0][:top_k]
    )

    results["metadatas"][0] = (
        results["metadatas"][0][:top_k]
    )

    if results.get("distances"):
        results["distances"][0] = (
            results["distances"][0][:top_k]
        )

    if results.get("ids"):
        results["ids"][0] = (
            results["ids"][0][:top_k]
        )

    # Display results

    display_semantic_results(
        results
    )

    return results


# LOW-RATING PRODUCT ANALYSIS

def get_low_rating_products():

    amazon_df = df[
        df["entity_type"]
        .astype(str)
        .str.lower()
        == "review"
    ].copy()

    amazon_df["rating"] = pd.to_numeric(
        amazon_df["rating"],
        errors="coerce"
    )

    low_rating_df = amazon_df[
        amazon_df["rating"].isin([1, 2])
    ].copy()

    return low_rating_df


# SUMMARIZE LOW-RATING PRODUCTS

def summarize_low_rating_products():

    low_rating_df = get_low_rating_products()

    if low_rating_df.empty:

        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "low_rating_count",
                "average_rating"
            ]
        )

    summary = (
        low_rating_df
        .groupby(
            [
                "product_id",
                "product_name"
            ],
            dropna=False
        )
        .agg(
            low_rating_count=(
                "rating",
                "count"
            ),
            average_rating=(
                "rating",
                "mean"
            )
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "low_rating_count",
        ascending=False
    )

    return summary


# DISPLAY NEGATIVE REVIEW RESULTS

def display_negative_review_results():

    summary = summarize_low_rating_products()

    print("\n" + "=" * 60)
    print("NEGATIVE REVIEW ANALYSIS")
    print("=" * 60)

    if summary.empty:

        print(
            "No products with poor ratings found."
        )

        return

    print(
        f"\nProducts with poor ratings: "
        f"{len(summary)}"
    )

    print("\nTop products:\n")

    for _, row in summary.head(10).iterrows():

        product_name = row["product_name"]

        if pd.isna(product_name):
            product_name = "Unknown"

        print(
            f"{product_name} | "
            f"Poor Reviews: "
            f"{int(row['low_rating_count'])} | "
            f"Average Rating: "
            f"{row['average_rating']:.2f}"
        )


# DISPLAY SEMANTIC RESULTS

def display_semantic_results(
    results
):

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    print("\n" + "=" * 60)
    print("SEMANTICALLY RELEVANT DOCUMENTS")
    print("=" * 60)

    if not documents:

        print(
            "\nNo relevant documents found."
        )

        return

    for i, document in enumerate(
        documents,
        start=1
    ):

        print(
            "\n" + "-" * 60
        )

        print(
            f"RESULT {i}"
        )

        print(
            "-" * 60
        )

        if i - 1 < len(distances):

            print(
                f"Distance: "
                f"{distances[i - 1]:.4f}"
            )

        if i - 1 < len(metadatas):

            metadata = metadatas[i - 1]

            print(
                f"Source: "
                f"{metadata.get('source', 'Unknown')}"
            )

            print(
                f"Entity Type: "
                f"{metadata.get('entity_type', 'Unknown')}"
            )

            print(
                f"Product ID: "
                f"{metadata.get('product_id', 'Unknown')}"
            )

        print(
            "\nDocument:\n"
        )

        print(document)


# SMART RETRIEVAL

def smart_retrieve(
    query,
    top_k=5
):

    intent = detect_query_intent(
        query
    )

    # Negative review query

    if (
        intent == "review"
        and is_negative_review_query(query)
    ):

        display_negative_review_results()

    # Semantic retrieval

    results = semantic_retrieve(
        query,
        top_k
    )

    return results


# MAIN

if __name__ == "__main__":

    query = input(
        "\nEnter your retail business question: "
    ).strip()

    if query:

        smart_retrieve(
            query
        )

    else:

        print(
            "Please enter a question."
        )