import streamlit as st
import pandas as pd
from pathlib import Path
import re


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Retail Decision Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# PROJECT PATH

# dashboard.py is inside:
# C:\Users\Jameel\Desktop\LLM\src\
#
# parent.parent gives:
# C:\Users\Jameel\Desktop\LLM\

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "integrated_retail_data.csv"
)


# LOAD DATA

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


df = load_data()


# DATASET CHECK

if df is None:

    st.error("Integrated retail dataset was not found.")

    st.write("Expected file:")

    st.code(str(DATA_PATH))

    st.stop()


# IMPORT DECISION ENGINE

try:

    from decision_engine import generate_business_decision

except Exception:

    try:

        from src.decision_engine import generate_business_decision

    except Exception as e:

        st.error("Could not load the decision engine.")

        st.code(str(e))

        st.stop()


# PAGE TITLE

st.title("🛍️ Retail Decision Intelligence")

st.write(
    "LLM-Augmented Retail Analytics using "
    "RAG, Knowledge Graphs and Retail Data"
)

st.divider()


# SIDEBAR

with st.sidebar:

    st.header("⚙️ Retail Intelligence")

    st.divider()

    st.subheader("Data Sources")

    st.write("🟠 Amazon Reviews")
    st.write("🔵 M5 Sales")
    st.write("🟢 Instacart Purchases")

    st.divider()

    st.subheader("AI Pipeline")

    st.write("📊 Integrated Dataset")
    st.write("🕸️ Knowledge Graph")
    st.write("📄 RAG Documents")
    st.write("🔎 Semantic Retrieval")
    st.write("🧠 LLM Decision Engine")
    st.write("💡 Business Recommendation")

    st.divider()

    st.caption(
        "F7 Retail Decision Intelligence Platform"
    )


# DATASET STATISTICS

total_records = len(df)

unique_products = (
    df["product_id"].nunique()
    if "product_id" in df.columns
    else 0
)

amazon_reviews = (
    len(
        df[
            df["source"].astype(str).str.lower() == "amazon"
        ]
    )
    if "source" in df.columns
    else 0
)

m5_sales_records = (
    len(
        df[
            df["source"].astype(str).str.lower() == "m5"
        ]
    )
    if "source" in df.columns
    else 0
)

instacart_purchases = (
    len(
        df[
            df["source"].astype(str).str.lower() == "instacart"
        ]
    )
    if "source" in df.columns
    else 0
)


# DATASET OVERVIEW

st.header("📊 Retail Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Retail Records",
        f"{total_records:,}"
    )


with col2:

    st.metric(
        "Unique Products",
        f"{unique_products:,}"
    )


with col3:

    st.metric(
        "Amazon Reviews",
        f"{amazon_reviews:,}"
    )


with col4:

    st.metric(
        "M5 Sales Records",
        f"{m5_sales_records:,}"
    )


with col5:

    st.metric(
        "Instacart Purchases",
        f"{instacart_purchases:,}"
    )


# M5 SALES SUMMARY

if (
    "sales" in df.columns
    and "source" in df.columns
):

    m5_df = df[
        df["source"].astype(str).str.lower() == "m5"
    ].copy()

    if len(m5_df) > 0:

        m5_df["sales"] = pd.to_numeric(
            m5_df["sales"],
            errors="coerce"
        ).fillna(0)

        total_sales = m5_df["sales"].sum()

        st.info(
            f"Total recorded M5 sales: "
            f"{total_sales:,.0f} units. "
            "This represents recorded sales quantity, not revenue, "
            "because product price is not available in the integrated dataset."
        )


# BUSINESS INTELLIGENCE SECTION

st.header("🤖 Ask the Retail Intelligence Engine")

st.write(
    "Ask a business question about products, sales, stores, "
    "categories, customers or reviews."
)


# EXAMPLE QUESTIONS

example_questions = [
    "How are sales going?",
    "Which products have highest sales?",
    "Which HOBBIES products have highest sales?",
    "Which store has highest sales?",
    "How can I increase sales of HOBBIES_1_371?",
    "How can I improve sales of FOODS?",
    "Which products have poor customer reviews?",
    "How can I improve sales of Amazon Fire TV?"
]


selected_example = st.selectbox(
    "Example Questions",
    ["-- Select an example --"] + example_questions
)


if selected_example != "-- Select an example --":

    default_query = selected_example

else:

    default_query = ""


query = st.text_area(
    "Business Question",
    value=default_query,
    height=100,
    placeholder="Example: Which products have highest sales?"
)


# ANALYZE BUTTON

if st.button(
    "🔍 Analyze Retail Data",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning(
            "Please enter a business question."
        )

    else:

        with st.spinner(
            "Analyzing retail data and generating business insight..."
        ):

            try:

                answer = generate_business_decision(
                    query.strip()
                )

                st.session_state["answer"] = answer

                st.session_state["last_query"] = query.strip()

            except Exception as e:

                st.error(
                    "An error occurred while generating the business decision."
                )

                st.code(str(e))


# BUSINESS ANSWER

if "answer" in st.session_state:

    st.divider()

    st.header("💡 Business Intelligence Result")

    answer = st.session_state["answer"]

    # --------------------------------------------------------
    # CLEAN UNWANTED SVG / STREAMLIT ANCHOR ARTIFACTS
    # --------------------------------------------------------

    # Remove patterns such as:
    # [svg](http://localhost:8501/#1-business-finding)
    # [svg](http://localhost:8501/#2-evidence)
    # etc.

    answer = re.sub(
        r'\[svg\]\(http://localhost:8501/#[^)]+\)',
        '',
        answer
    )

    # Also remove possible localhost SVG variants
    answer = re.sub(
        r'\[svg\]\(http://localhost:[0-9]+/#[^)]+\)',
        '',
        answer
    )

    # Remove excessive blank lines created after cleanup
    answer = re.sub(
        r'\n{3,}',
        '\n\n',
        answer
    )

    # Remove unnecessary whitespace at beginning/end
    answer = answer.strip()

    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.markdown(answer)


# RAG EXPLANATION

if "answer" in st.session_state:

    with st.expander(
        "🔎 How was this answer generated?"
    ):

        st.markdown(answer)


# EXECUTIVE ANALYTICS

st.divider()

st.header("📈 Executive Retail Analytics")


# SOURCE DISTRIBUTION

col_left, col_right = st.columns(2)


with col_left:

    st.subheader("Records by Data Source")

    if "source" in df.columns:

        source_counts = (
            df["source"]
            .value_counts()
            .rename_axis("Source")
            .reset_index(name="Records")
        )

        st.bar_chart(
            source_counts.set_index("Source")
        )


with col_right:

    st.subheader("Records by Entity Type")

    if "entity_type" in df.columns:

        entity_counts = (
            df["entity_type"]
            .value_counts()
            .rename_axis("Entity Type")
            .reset_index(name="Records")
        )

        st.bar_chart(
            entity_counts.set_index("Entity Type")
        )


# TOP PRODUCTS BY SALES

if (
    "product_id" in df.columns
    and "sales" in df.columns
    and "source" in df.columns
):

    st.subheader(
        "🏆 Top 10 Products by Recorded M5 Sales"
    )

    m5_sales_df = df[
        df["source"].astype(str).str.lower() == "m5"
    ].copy()

    if len(m5_sales_df) > 0:

        m5_sales_df["sales"] = pd.to_numeric(
            m5_sales_df["sales"],
            errors="coerce"
        ).fillna(0)

        top_products = (
            m5_sales_df
            .groupby(
                "product_id",
                as_index=False
            )["sales"]
            .sum()
            .sort_values(
                "sales",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            top_products,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            top_products.set_index("product_id")["sales"]
        )


# SALES BY STORE

if (
    "store_id" in df.columns
    and "sales" in df.columns
    and "source" in df.columns
):

    st.subheader(
        "🏪 Sales by Store"
    )

    m5_store_df = df[
        df["source"].astype(str).str.lower() == "m5"
    ].copy()

    if len(m5_store_df) > 0:

        m5_store_df["sales"] = pd.to_numeric(
            m5_store_df["sales"],
            errors="coerce"
        ).fillna(0)

        store_sales = (
            m5_store_df
            .groupby(
                "store_id",
                as_index=False
            )["sales"]
            .sum()
            .sort_values(
                "sales",
                ascending=False
            )
        )

        st.dataframe(
            store_sales,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            store_sales.set_index("store_id")["sales"]
        )


# CUSTOMER REVIEW ANALYTICS

if "source" in df.columns:

    amazon_df = df[
        df["source"].astype(str).str.lower() == "amazon"
    ].copy()

else:

    amazon_df = pd.DataFrame()


if (
    len(amazon_df) > 0
    and "rating" in amazon_df.columns
):

    st.subheader(
        "⭐ Customer Review Analytics"
    )

    amazon_df["rating"] = pd.to_numeric(
        amazon_df["rating"],
        errors="coerce"
    )

    valid_ratings = amazon_df[
        amazon_df["rating"].notna()
    ]

    if len(valid_ratings) > 0:

        review_col1, review_col2 = st.columns(2)

        with review_col1:

            st.metric(
                "Average Rating",
                f"{valid_ratings['rating'].mean():.2f} / 5"
            )

        with review_col2:

            st.metric(
                "Total Reviews",
                f"{len(valid_ratings):,}"
            )

        rating_distribution = (
            valid_ratings["rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            rating_distribution
        )


# DATASET INFORMATION

with st.expander(
    "📁 Dataset Information"
):

    st.write(
        "Dataset location:"
    )

    st.code(
        str(DATA_PATH)
    )

    st.write(
        f"Dataset shape: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    st.write(
        "Columns:"
    )

    st.write(
        list(df.columns)
    )

# FOOTER

st.divider()

st.caption(
    "F7 — Large Language Model-Augmented Retail Decision Intelligence Platform"
)

st.caption(
    "RAG • Knowledge Graph • ChromaDB • LLM Decision Reasoning"
)