import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Kaduna GPI Dashboard", layout="wide")

# --- DATA CLEANING FUNCTION ---
def clean_data(data):
    """Applies your specific cleaning logic to any uploaded dataset"""
    data.columns = data.columns.str.strip()
    if 'LGA' in data.columns:
        data['LGA'] = data['LGA'].astype(str).str.upper().str.strip()
        # Mapping from your notebook
        lga_mapping = {
            'BIRNINGWARI': 'BIRNIN GWARI',
            'K NORTH': 'KADUNA NORTH',
            'K SOUTH': 'KADUNA SOUTH',
            'S GARI': 'SABON GARI',
            'Z KATAF': 'ZANGON KATAF',
            'ZANGONKATAF': 'ZANGON KATAF'
        }
        data['LGA'] = data['LGA'].replace(lga_mapping)
    return data

# --- SIDEBAR: DATA SOURCE SELECTION ---
st.sidebar.title("Data Source")
data_source = st.sidebar.radio("Select Data Source:", ["Use Repository Data", "Upload New File"])

df = None

if data_source == "Upload New File":
    uploaded_file = st.sidebar.file_uploader("Upload your GPI Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = clean_data(pd.read_excel(uploaded_file))
    else:
        st.info("Please upload an Excel file to proceed.")
        st.stop()
else:
    # Proper try-except block for loading the local file
    try:
        df = clean_data(pd.read_excel('GPI data.xlsx'))
    except FileNotFoundError:
        st.error("Repository file 'GPI data.xlsx' not found. Please upload a file via the sidebar.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Regional Heatmap", "Model Performance"])

# --- PAGE 1: DASHBOARD ---
if page == "Dashboard":
    st.title("Kaduna GPI Overview")
    if 'Year' in df.columns:
        selected_year = st.sidebar.selectbox("Select Academic Year", options=df['Year'].unique())
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("GPI by LGA")
        fig, ax = plt.subplots()
        sns.barplot(data=filtered_df, x='LGA', y='GPI', ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)
    with col2:
        st.subheader("Data Summary")
        st.dataframe(filtered_df)

# --- PAGE 2: REGIONAL HEATMAP ---
elif page == "Regional Heatmap":
    st.title("Regional Variation Heatmap")
    if 'LGA' in df.columns and 'Year' in df.columns:
        # Creating heatmap based on notebook data structure
        heatmap_data = df.pivot_table(index='LGA', columns='Year', values='GPI')
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Heatmap requires 'LGA' and 'Year' columns.")

# --- PAGE 3: MODEL PERFORMANCE ---
elif page == "Model Performance":
    st.title("Model Performance Metrics")
    
    # Feature selection matching your notebook
    required_cols = ['LGA', 'GPI', 'total number of schools', 'total number of pupils']
    if all(col in df.columns for col in required_cols):
        model_df = df.copy().dropna(subset=required_cols)
        le = LabelEncoder()
        model_df['LGA_Encoded'] = le.fit_transform(model_df['LGA'])
        
        X = model_df[['LGA_Encoded', 'total number of schools', 'total number of pupils']]
        y = (model_df['GPI'] >= 1.0).astype(int) 

        if len(y.unique()) < 2:
            st.error("The data doesn't have enough variety (both parity and non-parity) to train a model.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
            clf = RandomForestClassifier(random_state=42) if model_choice == "Random Forest" else LogisticRegression()

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_probs = clf.predict_proba(X_test)[:, 1]

            metrics_cols = st.columns(5)
            metrics_cols[0].metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            metrics_cols[1].metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
            metrics_cols[2].metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
            metrics_cols[3].metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")
            metrics_cols[4].metric("AUC-ROC", f"{roc_auc_score(y_test, y_probs):.2f}")

            st.subheader("Detailed Classification Report")
            st.text(classification_report(y_test, y_pred))
    else:
        st.error(f"Modeling requires the following columns: {required_cols}")
