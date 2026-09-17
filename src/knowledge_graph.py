import pandas as pd
import networkx as nx
from pathlib import Path


# 1. PROJECT PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "integrated_retail_data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_knowledge_graph.graphml"
)


# 2. LOAD INTEGRATED DATASET

print("=" * 60)
print("LOADING INTEGRATED RETAIL DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# 3. CREATE KNOWLEDGE GRAPH

G = nx.MultiDiGraph()


# 4. PROCESS EACH RECORD

print("\nBuilding knowledge graph...")

for index, row in df.iterrows():

    source = str(row["source"]).strip()
    entity_type = str(row["entity_type"]).strip()

    # AMAZON RECORDS

    if source.lower() == "amazon":

        product_id = str(row["product_id"]).strip()

        product_node = f"amazon_product:{product_id}"

        # Product
        G.add_node(
            product_node,
            type="Product",
            source="Amazon",
            product_id=product_id,
            name=str(row["product_name"])
            if pd.notna(row["product_name"])
            else ""
        )

        # Brand
        if pd.notna(row["brand"]) and str(row["brand"]).strip():

            brand = str(row["brand"]).strip()

            brand_node = f"amazon_brand:{brand}"

            G.add_node(
                brand_node,
                type="Brand",
                source="Amazon",
                name=brand
            )

            G.add_edge(
                product_node,
                brand_node,
                relationship="HAS_BRAND"
            )

        # Category
        if pd.notna(row["category"]) and str(row["category"]).strip():

            category = str(row["category"]).strip()

            category_node = f"amazon_category:{category}"

            G.add_node(
                category_node,
                type="Category",
                source="Amazon",
                name=category
            )

            G.add_edge(
                product_node,
                category_node,
                relationship="BELONGS_TO_CATEGORY"
            )

        # Review
        if entity_type.lower() == "review":

            review_node = f"amazon_review:{index}"

            rating = (
                float(row["rating"])
                if pd.notna(row["rating"])
                else None
            )

            review_text = (
                str(row["review_text"])
                if pd.notna(row["review_text"])
                else ""
            )

            G.add_node(
                review_node,
                type="Review",
                source="Amazon",
                rating=rating,
                text=review_text
            )

            G.add_edge(
                product_node,
                review_node,
                relationship="HAS_REVIEW"
            )


    # M5 RECORDS

    elif source.lower() == "m5":

        product_id = str(row["product_id"]).strip()

        product_node = f"m5_product:{product_id}"

        # Product
        G.add_node(
            product_node,
            type="Product",
            source="M5",
            product_id=product_id
        )

        # Category
        if pd.notna(row["category"]) and str(row["category"]).strip():

            category = str(row["category"]).strip()

            category_node = f"m5_category:{category}"

            G.add_node(
                category_node,
                type="Category",
                source="M5",
                name=category
            )

            G.add_edge(
                product_node,
                category_node,
                relationship="BELONGS_TO_CATEGORY"
            )

        # Store
        if pd.notna(row["store_id"]):

            store = str(row["store_id"]).strip()

            store_node = f"m5_store:{store}"

            G.add_node(
                store_node,
                type="Store",
                source="M5",
                name=store
            )

            G.add_edge(
                product_node,
                store_node,
                relationship="SOLD_AT"
            )

            # State
            if pd.notna(row["state_id"]):

                state = str(row["state_id"]).strip()

                state_node = f"m5_state:{state}"

                G.add_node(
                    state_node,
                    type="State",
                    source="M5",
                    name=state
                )

                G.add_edge(
                    store_node,
                    state_node,
                    relationship="LOCATED_IN"
                )

        # Sales
        if pd.notna(row["sales"]):

            sales_node = f"m5_sales:{index}"

            sales_value = float(row["sales"])

            G.add_node(
                sales_node,
                type="Sales",
                source="M5",
                sales=sales_value
            )

            G.add_edge(
                product_node,
                sales_node,
                relationship="HAS_SALES"
            )


    # INSTACART RECORDS

    elif source.lower() == "instacart":

        product_id = str(row["product_id"]).strip()

        product_node = f"instacart_product:{product_id}"

        # Product
        G.add_node(
            product_node,
            type="Product",
            source="Instacart",
            product_id=product_id,
            name=str(row["product_name"])
            if pd.notna(row["product_name"])
            else ""
        )

        # Customer
        if pd.notna(row["customer_id"]):

            customer_id = str(row["customer_id"]).strip()

            customer_node = (
                f"instacart_customer:{customer_id}"
            )

            G.add_node(
                customer_node,
                type="Customer",
                source="Instacart",
                customer_id=customer_id
            )

            G.add_edge(
                customer_node,
                product_node,
                relationship="PURCHASED"
            )

        # Purchase
        if pd.notna(row["purchase"]):

            purchase_node = f"instacart_purchase:{index}"

            purchase_value = float(row["purchase"])

            G.add_node(
                purchase_node,
                type="Purchase",
                source="Instacart",
                purchase=purchase_value,
                date=str(row["date"])
                if pd.notna(row["date"])
                else ""
            )

            G.add_edge(
                product_node,
                purchase_node,
                relationship="HAS_PURCHASE"
            )


# 5. GRAPH STATISTICS

print("\n" + "=" * 60)
print("KNOWLEDGE GRAPH CREATED")
print("=" * 60)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


# 6. NODE TYPE COUNTS

print("\nNode types:")

node_types = {}

for node, data in G.nodes(data=True):

    node_type = data.get("type", "Unknown")

    node_types[node_type] = (
        node_types.get(node_type, 0) + 1
    )


for node_type, count in sorted(node_types.items()):

    print(f"{node_type:15} : {count}")


# 7. RELATIONSHIP COUNTS

print("\nRelationships:")

relationship_counts = {}

for source, target, data in G.edges(data=True):

    relationship = data.get(
        "relationship",
        "UNKNOWN"
    )

    relationship_counts[relationship] = (
        relationship_counts.get(relationship, 0) + 1
    )


for relationship, count in sorted(
    relationship_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{relationship:25} : {count}")


# 8. SAVE KNOWLEDGE GRAPH

# Replace None values because GraphML does not support Python None
for node, attributes in G.nodes(data=True):
    for key, value in attributes.items():
        if value is None:
            attributes[key] = "Unknown"

for source, target, attributes in G.edges(data=True):
    for key, value in attributes.items():
        if value is None:
            attributes[key] = "Unknown"

nx.write_graphml(
    G,
    OUTPUT_FILE
)

print("\nKnowledge graph saved successfully!")

print(OUTPUT_FILE)


# 9. SHOW SAMPLE RELATIONSHIPS

print("\n" + "=" * 60)
print("SAMPLE RELATIONSHIPS")
print("=" * 60)

count = 0

for source, target, data in G.edges(data=True):

    print(
        f"{source} "
        f"--[{data.get('relationship')}]--> "
        f"{target}"
    )

    count += 1

    if count >= 20:
        break


print("\nKnowledge Graph construction completed!")