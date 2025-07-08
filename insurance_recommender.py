import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

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

# Identify valid numeric purchase columns
purchase_cols = []
for col in known_product_codes:
    if col in clients.columns and pd.api.types.is_numeric_dtype(clients[col]):
        purchase_cols.append(col)

# Debug print
print(f"Detected {len(purchase_cols)} valid purchase columns:", purchase_cols)

# Handle empty purchase columns case
if not purchase_cols:
    print("No valid purchase columns found! Skipping collaborative recommendations.")
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
    not_purchased = product_scores[current_user <1]
    recommended = not_purchased.sort_values(ascending=False).head(top_n)
    return recommended
    


# Step 6: Run recommendation for a customer
client_index = 0  # You can change this index
sample = clients.iloc[client_index]
types = recommend_insurance_types(sample)
content_based = match_products(types, products.copy())

print(f"\n👤 Client ID: {sample['ClientID']}")
print(f"Age: {sample['Age']} years")
print(f"Life Stage: {sample['LifeStage']}")
print(f"Recommended Insurance Categories: {types}")
print("\n Top Recommended Products (Content-Based):\n")

for idx, row in enumerate(content_based.head(5).itertuples(), start=1):
    matched_types = set(row.InsuranceTypeList).intersection(types)
    reasons = " ".join([explanations.get(t, "") for t in matched_types])
    
    print(f"{idx}. {row.ProductDescription}")
    print(f"   • Insurance Types: {row.InsuranceTypeList}")
    print(f"   • Why? {reasons}\n")

# Step 7: Collaborative filtering output
print("\n Collaborative Recommendations (Similar Users):\n")

collab = collaborative_recommendation(client_index)

if collab.empty or all(score <= 0 for score in collab):
    print("Not enough data to generate collaborative recommendations.")
else:
    for product_code, score in collab.items():
        if score > 0:
            desc_row = products[products['ProductCode'] == product_code]
            if not desc_row.empty:
                desc = desc_row['ProductDescription'].values[0]
                print(f"✔ {desc} ({product_code}) → Score: {score:.2f} (similar customers bought this)")
