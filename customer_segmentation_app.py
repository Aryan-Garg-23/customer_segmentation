import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import StringIO

st.title("Customer Segmentation using K-Means Clustering")

st.write("Upload a CSV file containing customer data with 'Annual Income' and 'Spending Score' columns.")

uploaded_file = st.file_uploader(r"c:\Users\Mehak\Downloads\Mall_Customers.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Uploaded Dataset Preview:")
    st.dataframe(df.head())

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    st.write("### Elbow Method to Determine Optimal Clusters")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.write("### Clustered Data Preview:")
    st.dataframe(df.head())

    st.write("### Customer Segmentation Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='viridis', ax=ax)
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.set_title('Customer Segmentation Clusters')
    st.pyplot(fig)
