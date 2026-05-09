import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Food Recommendation System",
    page_icon="🍔",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #FFB703;
}

div.stButton > button {
    background-color: #FFB703;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #ffaa00;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

food = pd.read_csv("food.csv")

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

food["Features"] = (
    food["Category"] + " " +
    food["Cuisine"] + " " +
    food["Ingredients"]
)

# ---------------------------------------------------
# TEXT VECTORIZATION
# ---------------------------------------------------

cv = CountVectorizer()

matrix = cv.fit_transform(food["Features"])

# ---------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------

similarity = cosine_similarity(matrix)

# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------

def recommend(food_name):

    food_name = food_name.lower()

    if food_name not in food["Food"].str.lower().values:
        return []

    index = food[food["Food"].str.lower() == food_name].index[0]

    distances = list(enumerate(similarity[index]))

    foods_list = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = []

    for i in foods_list:

        recommended_food = food.iloc[i[0]].Food
        similarity_score = round(i[1], 2)

        recommendations.append(
            (recommended_food, similarity_score)
        )

    return recommendations

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🍔 Food Recommendation System")

st.write(
    "Recommend similar foods using Machine Learning & Recommender Systems"
)

# ---------------------------------------------------
# CENTERED IMAGE
# ---------------------------------------------------

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(
        "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
        width=500
    )

# ---------------------------------------------------
# DATASET INFO
# ---------------------------------------------------

st.subheader("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Foods", len(food))

with col2:
    st.metric("Total Cuisines", food["Cuisine"].nunique())

# ---------------------------------------------------
# FOOD SELECTION
# ---------------------------------------------------

st.subheader("🍕 Select Food")

selected_food = st.selectbox(
    "Choose a food item",
    food["Food"].values
)

# ---------------------------------------------------
# RECOMMEND BUTTON
# ---------------------------------------------------

if st.button("🍽️ Recommend Foods"):

    recommendations = recommend(selected_food)

    st.subheader("✅ Recommended Foods")

    if recommendations:

        for item, score in recommendations:

            st.success(
                f"🍴 {item}   |   Similarity Score: {score}"
            )

    else:
        st.error("Food item not found!")

# ---------------------------------------------------
# FOOD CATEGORY CHART
# ---------------------------------------------------

st.subheader("📈 Food Category Distribution")

category_count = food["Category"].value_counts()

st.bar_chart(category_count)

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(food)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    "✨ Developed using Python, Streamlit & Machine Learning"
)