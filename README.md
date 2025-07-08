# Insurance-Product-Recommendation-Engine

A hybrid AI/ML-powered system that recommends personalized insurance products based on customers' life stages, risk profiles, and purchase behaviors — with clear explanations for every suggestion.

# Project Objective

To develop a smart recommendation engine that:
- Understands a customer's life stage and potential risks
- Matches them with suitable insurance products
- Uses both **content-based** and **collaborative filtering**
- Explains every recommendation in natural language

# Problem Statement

Build a system that recommends appropriate insurance products to customers based on their life circumstances and risk profile.
The system must:
- Implement AI/ML techniques (collaborative filtering, content-based)
- Analyze customer needs based on life events
- Deliver explainable, ranked recommendations
- Be modular and scalable

# Project Structure
insurance-product-recommendation/
├── insurance_recommender.py # Main logic for recommendation engine
├── clients.csv # Customer demographic and purchase data
├── products.csv # Product catalog with insurance type tags
├── README.md # Documentation

# System Architecture

Customer Data + Product Data
              ↓
Age Calculation & Life Stage Detection
              ↓
Insurance Category Inference (Health, Life, etc.)
              ↓
Content-Based Filtering
              ↓
Collaborative Filtering (cosine similarity)
              ↓
Final Ranked Recommendations with Explanations

# How It Works

1️⃣ Customer Profiling
- Derives age from birth year
- Classifies users into life stages:
  - Early Career (<30)
  - Mid Career (30–39)
  - Parenting (40–54)
  - Pre-retirement (55+)

2️⃣ Insurance Need Mapping
- `Health` if age ≥ 35
- `Life` if Mid Career or Parenting
- `Property` if Pre-retirement
- `Vehicle` if the client owns a vehicle

3️⃣ Content-Based Filtering
- Matches inferred needs with product categories
- Scores products based on overlap
- Sorts by score

4️⃣ Collaborative Filtering
- Builds user-product matrix from historical purchases
- Computes cosine similarity between users
- Recommends products that similar users purchased

# Sample Output
 Client ID: 4WKQSBB
Age: 38 years
Life Stage: Mid Career
Recommended Insurance Categories: ['Health', 'Life']

 Top Recommended Products (Content-Based):

1. Personal Health Insurance
   • Insurance Types: ['Health']
   • Why? Recommended because age is 35 or above — higher health risk.

2. Comprehensive Vehicle Insurance
   • Insurance Types: ['Vehicle', 'Life', 'Property']
   • Why? Recommended due to mid-career or parenting stage to support dependents.

3. Comprehensive Accident Insurance
   • Insurance Types: ['Life', 'Property']
   • Why? Recommended due to mid-career or parenting stage to support dependents.

4. Life Insurance
   • Insurance Types: ['Life']
   • Why? Recommended due to mid-career or parenting stage to support dependents.

5. Disability Insurance
   • Insurance Types: ['Health']
   • Why? Recommended because age is 35 or above — higher health risk.

Collaborative Recommendations (Similar Users):
✔ Critical Illness Insurance (K6QO) → Score: 16.5 (similar customers bought this)
✔ Roadside Assistance Plan (RVSZ) → Score: 13.2 (similar customers bought this)
✔ Family Term Insurance (BSTQ) → Score: 12.0 (similar customers bought this)

# Tech Stack

- Python
- Pandas
- Scikit-learn
- Content-Based Filtering
- Collaborative Filtering (cosine similarity)

# Future Improvements

Streamlit UI for interactive recommendations

Export results as downloadable reports

Use ML classifiers for more personalized predictions

Rank by profitability for the company


