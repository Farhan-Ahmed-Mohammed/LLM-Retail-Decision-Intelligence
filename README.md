# LLM Retail Decision Intelligence

A retail analytics project that uses **RAG, a Knowledge Graph, ChromaDB, and Gemini** to answer business-related questions from retail data.

## About the Project

Retail data can come from different sources such as sales records, customer reviews, and purchase history. It can be difficult to get useful information from all these sources together.

In this project, I combined data from the **Amazon Reviews, M5 Forecasting, and Instacart datasets** and built a system where a user can ask questions in normal language.

For example:

* How are sales going?
* Which category performs best?
* Which products have the highest sales?
* Which store has the highest sales?
* What are customers saying about a product?
* Which products have negative reviews?

The system retrieves relevant information, performs calculations on the sales data when needed, and then uses Gemini to generate the final response.

## Main Features

* Data cleaning and integration
* Retail Knowledge Graph using NetworkX
* RAG document creation
* Text embeddings using Sentence Transformers
* ChromaDB for vector storage and retrieval
* Query and product detection
* Structured sales analysis
* Gemini-based business reasoning
* Streamlit dashboard

## Datasets

### Amazon Reviews

Used mainly for customer feedback, ratings, products, brands, and review analysis.

### M5 Forecasting

Used for product sales, stores, states, categories, and sales-volume analysis.

### Instacart

Used for customer purchase and product information.

After preprocessing:

| Dataset        |     Records |
| -------------- | ----------: |
| Amazon Reviews |      34,660 |
| M5             |      30,490 |
| Instacart      |      38,006 |
| **Total**      | **103,156** |

The datasets don't have a common product ID, so I used a common schema to combine the data rather than treating products from the three datasets as the same products.

## Project Structure

```text
LLM-Retail-Decision-Intelligence/
│
├── dashboard.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── knowledge_graph.py
│   ├── create_rag_documents.py
│   ├── create_embeddings.py
│   ├── retriever.py
│   └── decision_engine.py
│
├── data/
│   └── processed/
│
└── screenshots/
```

## Knowledge Graph

I used **NetworkX** to create a retail Knowledge Graph.

The generated graph contains:

* 110,375 nodes
* 301,952 edges

The graph includes entities such as:

* Products
* Categories
* Brands
* Customers
* Reviews
* Purchases
* Sales
* Stores
* States

Some of the relationships used are:

```text
HAS_BRAND
BELONGS_TO_CATEGORY
HAS_REVIEW
SOLD_AT
LOCATED_IN
PURCHASED
HAS_PURCHASE
```

The graph is saved in GraphML format.

## RAG

The RAG part of the project is used to retrieve relevant information before generating an answer.

The basic flow is:

```text
User Question
      ↓
Query Understanding
      ↓
Retrieve Relevant Documents
      ↓
Structured Data Analysis
      ↓
Gemini
      ↓
Business Answer
```

I created **103,156 RAG documents** from the processed retail data.

For embeddings, I used:

```text
all-MiniLM-L6-v2
```

The embeddings have a dimension of **384**.

The vectors are stored in **ChromaDB**.

Collection name:

```text
retail_documents
```

## Retrieval

The retriever handles different types of questions, including:

* Sales
* Products
* Categories
* Stores
* States
* Reviews
* Purchases

It also supports exact M5 product IDs such as:

```text
HOBBIES_1_371
FOODS_3_090
HOUSEHOLD_1_001
```

This helps when the user asks about a specific M5 product.

## Decision Engine

The decision engine combines retrieved information with structured analysis.

For sales-related questions, it can calculate:

* Overall sales
* Product sales
* Category sales
* Store sales
* State sales

The Gemini model then receives the relevant evidence and generates a response in the following format:

```text
1. BUSINESS FINDING
2. EVIDENCE
3. BUSINESS INTERPRETATION
4. RECOMMENDATION
5. CONFIDENCE
```

The system is also instructed not to make claims about revenue or profit when those values are not available in the data.

## Example Result

For the question:

```text
How are sales going?
```

The system found:

* Total recorded sales: **65,695,409 units**
* Sales records: **30,490**
* Unique products: **3,049**
* FOODS: **45,089,939 units**
* California: **28,675,547 units**
* Top store: **CA_3**
* Top product: **FOODS_3_090**

The sales values here represent **unit quantities**, not revenue.

For the question:

```text
Which category performs best?
```

the system identifies **FOODS** based on recorded unit sales volume.

## Dashboard

The project has a Streamlit dashboard where users can enter business questions and view the generated results.

The dashboard also contains:

* Dataset information
* Sales analytics
* Store and state analysis
* Review analytics
* RAG pipeline explanation
* Business decision results

## Technologies Used

```text
Python
Pandas
NumPy
NetworkX
Sentence Transformers
ChromaDB
Gemini
Streamlit
Scikit-learn
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/LLM-Retail-Decision-Intelligence.git
```

Go to the project folder:

```bash
cd LLM-Retail-Decision-Intelligence
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Set the Gemini API key in your environment.

For PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Run the dashboard:

```bash
python -m streamlit run dashboard.py
```

## Rebuilding the Data Pipeline

If the processed data or RAG documents are changed, the main scripts can be run in this order:

```bash
python src/knowledge_graph.py
python src/create_rag_documents.py
python src/create_embeddings.py
```

Then run:

```bash
python -m streamlit run dashboard.py
```

The embedding step takes significant time, so it doesn't need to be repeated unless the RAG documents are changed.

## Future Improvements

Some things that can be added later:

* Product price and revenue data
* Profit and margin analysis
* Inventory and stockout data
* Better matching of products across datasets
* Demand forecasting
* Anomaly detection
* More advanced Knowledge Graph retrieval

## Project

**LLM Retail Decision Intelligence**

This project was developed as an Artificial Intelligence and Machine Learning project to explore the use of **LLMs, RAG, Knowledge Graphs, vector databases, and retail analytics** for business decision support.
