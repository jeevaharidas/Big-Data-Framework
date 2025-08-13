# 🛒 RetailRocket Recommender System – Scalable Data Intelligence Platform with PySpark

This project builds an end-to-end **data analytics and machine learning platform** using **Apache Spark (PySpark)** to analyze user behavior and predict purchase intent in an e-commerce environment using the **RetailRocket dataset**. The solution includes ingestion, processing, feature engineering, modeling, and an interactive frontend to visualize and query insights.

---

## 🎯 Project Objective

> Build a scalable data intelligence platform to predict **customer purchase intent** based on session behavior (views, cart adds, purchases), and provide recommendations using PySpark with a user-friendly frontend for visual interaction.

---

## 📦 Dataset Summary

| Feature        | Description                          |
|----------------|--------------------------------------|
| Dataset Name   | RetailRocket Recommender Dataset     |
| Source         | [Kaggle Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) |
| Size           | ~1.3 GB (uncompressed)               |
| Format         | CSV (clean, tabular)                 |
| Events         | Over 2.7 million (views, carts, purchases) |
| Product Info   | Includes item metadata, categories   |

**Files used:**
- `events.csv` – User actions (views, add-to-cart, transactions)
- `item_properties_part1.csv` + `part2.csv` – Item metadata
- `category_tree.csv` – Product category hierarchy

---

## 🧱 Tech Stack

| Layer     | Tools Used                        |
|-----------|-----------------------------------|
| Language  | Python                            |
| Backend   | PySpark (RDDs, Spark SQL, MLlib)  |
| Frontend  | Streamlit or Flask (interactive)  |
| Libraries | Matplotlib, Plotly, Pandas        |
| Storage   | CSV / Parquet files               |
| Platform  | VS Code / Jupyter / Cloud         |

---

## 🧩 Project Architecture

1. **Data Ingestion**
   - Load large CSV files using Spark
   - Handle RDD transformation + DataFrame creation

2. **Data Cleaning**
   - Remove nulls, filter irrelevant events
   - Parse timestamps and session identifiers

3. **Feature Engineering**
   - Aggregate user sessions: count of views, carts, transactions
   - Session-based metrics: time of day, item diversity
   - Create binary target: **Purchase Intent (1/0)**

4. **Machine Learning**
   - Vectorize features using `VectorAssembler`
   - Train a `LogisticRegression` model with `MLlib`
   - Evaluate using `BinaryClassificationEvaluator` (AUC, Accuracy)

5. **Visualization & Reporting**
   - Spark SQL for behavior queries
   - Use Plotly/Matplotlib to plot:
     - Purchase funnels
     - Category-wise conversion rates
     - User activity by time

6. **Frontend App**
   - Upload CSV or use preset dataset
   - Display queries, charts, and predictions
   - Allow user input for model prediction
   - Enable download of results

---

## 🧠 Problem Statement

> **Can we predict whether a user session will lead to a purchase, based on behavioral patterns like product views and add-to-cart events?**

---

## 🔬 Features Used for Modeling

| Feature Name           | Description |
|------------------------|-------------|
| `num_views`            | No. of views in session |
| `num_adds`             | No. of add-to-cart events |
| `distinct_products`    | Unique products viewed |
| `session_hour`         | Hour of activity |
| `purchase_intent`      | Target variable (1 if purchase, else 0) |

---

## ⚙️ Model Training (MLlib)

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

assembler = VectorAssembler(inputCols=["num_views", "num_adds", "distinct_products", "session_hour"], outputCol="features")
data_vectorized = assembler.transform(session_df)

train, test = data_vectorized.randomSplit([0.8, 0.2])
model = LogisticRegression(labelCol="purchase_intent", featuresCol="features")
model.fit(train)
| Metric    | Value (example) |
| --------- | --------------- |
| AUC       | 0.87            |
| Accuracy  | 82%             |
| Precision | 76%             |
| Recall    | 79%             |

📊 Frontend Dashboard Features (Streamlit)
📥 Upload CSV / Load Preset Data

📊 Query Builder using Spark SQL

📈 Purchase Funnel Visualization

🎯 Run Purchase Prediction

💾 Download Predicted Results

📂 Folder Structure
├── data/
│   ├── events.csv
│   ├── item_properties_part1.csv
│   └── item_properties_part2.csv
├── spark/
│   ├── ingest_and_clean.py
│   ├── feature_engineering.py
│   ├── model_training.py
├── frontend/
│   └── app.py (Streamlit interface)
├── output/
│   ├── predictions.csv
│   └── purchase_intent_model/
├── screenshots/
├── README.md
└── requirements.txt

📥 Setup Instructions
1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/your-username/retailrocket-pyspark.git
cd retailrocket-pyspark

2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt

3. Run PySpark Pipeline
bash
Copy
Edit
spark-submit spark/model_training.py

4. Launch Frontend
bash
Copy
Edit
streamlit run frontend/app.py

✅ Evaluation Criteria Mapping
Criterion	Addressed?
Dataset ≥ 1 GB	✅ Yes
Use of RDDs & Spark SQL	✅ Yes
MLlib Integration	✅ Yes
Frontend & UX	✅ Yes
Innovation (funnel + intent)	✅ Yes
Code Quality & Docs	✅ Yes

📌 License
This project is for academic use under the MIT License.

yaml
Copy
Edit
