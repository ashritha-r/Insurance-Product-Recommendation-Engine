import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("📊 Insurance Product Recommendation Engine")

# Load client and product data
clients = pd.read_csv("clients.csv")
products = pd.read_csv("products.csv")

# Step 1: Customer profiling
clients['Age'] = 2025 - clients['birth_year']

def get_life_stage(age):
    if age < 30:
        return 'Early Career'
    elif age < 40:
        return 'Mid Career'
    elif age < 55:
        return 'Parenting'
    else:
        return 'Pre-retirement'

clients['LifeStage'] = clients['Age'].apply(get_life_stage)

# Step 2: Prepare insurance type list
products['InsuranceTypeList'] = products['InsuranceType'].apply(lambda x: [i.strip() for i in str(x).split('|')])

# Step 3: Recommendation logic
def recommend_insurance_types(profile):
    types = set()
    if profile['LifeStage'] in ['Mid Career', 'Parenting']:
        types.add('Life')
    if profile['LifeStage'] == 'Pre-retirement':
        types.add('Property')
    if profile['Age'] >= 35:
        types.add('Health')
    if profile.get('VehicleOwner', 'No') == 'Yes':
        types.add('Vehicle')
    return list(types)

def match_products(types_needed, product_df):
    def score(row):
        return len(set(row['InsuranceTypeList']).intersection(types_needed))
    product_df['Score'] = product_df.apply(score, axis=1)
    return product_df[product_df['Score'] > 0].sort_values(by='Score', ascending=False)

# Step 4: Explanation dictionary
explanations = {
    "Life": "Recommended due to mid-career or parenting stage to support dependents.",
    "Health": "Recommended because age is 35 or above — higher health risk.",
    "Vehicle": "Suggested since the client owns a vehicle and might need vehicle protection.",
    "Property": "Recommended for clients nearing retirement to protect assets.",
    "General": "Additional coverage based on general lifestyle and risk."
}

# Step 5: Collaborative filtering setup
known_product_codes = products['ProductCode'].tolist()

purchase_cols = []
for col in known_product_codes:
    if col in clients.columns and pd.api.types.is_numeric_dtype(clients[col]):
        purchase_cols.append(col)

if not purchase_cols:
    purchase_matrix = pd.DataFrame()
else:
    purchase_matrix = clients[purchase_cols].fillna(0).astype(float)

def collaborative_recommendation(client_index, top_n=3):
    if purchase_matrix.empty:
        return pd.Series(dtype=int)
    
    similarity_scores = cosine_similarity([purchase_matrix.iloc[client_index]], purchase_matrix)[0]
    similar_indices = similarity_scores.argsort()[::-1][1:6]
    similar_users = purchase_matrix.iloc[similar_indices]
    product_scores = similar_users.sum()
    current_user = purchase_matrix.iloc[client_index]
    not_purchased = product_scores[current_user < 1]
    recommended = not_purchased.sort_values(ascending=False).head(top_n)
    return recommended

# --- UI Section ---
st.sidebar.title("👤 Select Client")
client_index = st.sidebar.slider("Client Index", 0, len(clients) - 1, 0)

sample = clients.iloc[client_index]
types = recommend_insurance_types(sample)
content_based = match_products(types, products.copy())

# Display client info
st.subheader(f"Client ID: {sample['ClientID']}")
st.markdown(f"**Age:** {sample['Age']} years")
st.markdown(f"**Life Stage:** {sample['LifeStage']}")
st.markdown(f"**Recommended Insurance Categories:** `{', '.join(types)}`")

# Content-Based Recommendations
st.subheader("📌 Top Recommended Products (Content-Based)")

for idx, row in enumerate(content_based.head(5).itertuples(), start=1):
    matched_types = set(row.InsuranceTypeList).intersection(types)
    reasons = " ".join([explanations.get(t, "") for t in matched_types])
    
    st.markdown(f"**{idx}. {row.ProductDescription}**")
    st.markdown(f"• Insurance Types: `{row.InsuranceTypeList}`")
    st.markdown(f"• 🧠 Why? _{reasons}_")

# Collaborative Filtering
st.subheader("🤝 Collaborative Recommendations (Similar Users)")
collab = collaborative_recommendation(client_index)

if collab.empty or all(score <= 0 for score in collab):
    st.warning("Not enough data to generate collaborative recommendations.")
else:
    for product_code, score in collab.items():
        if score > 0:
            desc_row = products[products['ProductCode'] == product_code]
            if not desc_row.empty:
                desc = desc_row['ProductDescription'].values[0]
                st.markdown(f"✔ **{desc}** ({product_code}) → _Score: {score:.2f}_")
