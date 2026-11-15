# Retail Cross-Sell Catalogue Assistant 🛒

An end-to-end e-commerce recommendation system and analytics web application built on the UCI Online Retail dataset. This platform recommends cross-sell products for individual items and multi-item shopping baskets using a from-scratch **Apriori Association Rule Mining** pipeline, paired with a custom **TF-IDF Cosine Similarity** engine to handle "cold-start" products.

## ✨ Key Features

* **Market Basket Analysis Engine:** Custom Python implementation of the Apriori algorithm calculating Support, Confidence, and Lift.
* **Smart Rule Aggregation:** Scores basket recommendations based on a custom weighted formula rather than raw Lift alone, preventing rare flukes from dominating results.
* **TF-IDF Cold-Start Fallback:** Uses NLP tokenization and Cosine Similarity to find the closest existing catalogue items for brand-new products, borrowing their rules to generate instant starter bundles.
* **Interactive Dashboard:** A lightweight Single-Page Application (SPA) featuring live threshold sliders, a persistent shopping basket, and 1-click curated demo presets.
* **Data Visualizations:** Dynamic, auto-generated SVG scatter plots, histograms, and an interactive product co-occurrence network graph.

## 🛠️ Tech Stack

* **Backend:** Python 3, Flask
* **Algorithms:** Pure Python standard libraries (`math`, `collections`, `itertools`, `re`) — built without heavy ML frameworks to demonstrate core algorithmic mechanics.
* **Frontend:** Vanilla HTML5, CSS3, ES6 JavaScript (No React/Vue overhead).
* **Testing:** Python `unittest` suite.

## 📁 Project Structure

* `app.py`: Flask REST API routing and input validation.
* `engine.py`: Core scoring logic, rule aggregation, and TF-IDF search index.
* `pipeline.py`: Offline data cleaning and Apriori mining CLI.
* `static/index.html`: The interactive frontend dashboard.
* `test_engine.py`: Comprehensive 11-test suite covering pipeline generation, rule logic, and API endpoints.
* `catalog.json` & `rules.json`: Pre-mined model artifacts (400 products, 500 rules).

## 🚀 Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/priyamshree/retail-cross-sell-engine.git](https://github.com/priyamshree/retail-cross-sell-engine.git)
cd retail-cross-sell-engine