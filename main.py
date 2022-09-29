import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px

st.set_page_config(page_title="Credit Card Clustering", page_icon="💳", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)

with open("style.css") as f:
  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# get the cluster data from JSON
with open("cluster.json") as f:
    cluster_json = json.load(f)

#Navigation Bar
st.markdown(
"""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #004A3D;">
  <a href="/" target="_self" id="main-btn">Credit Card Clustering</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a id="notebook" class="nav-link active" href="https://www.kaggle.com/code/danielsimamora/credit-card-customer-segmentation" target="_blank">📄Notebook</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html = True)

st.markdown("""<p id="title-1z2x">Credit Card Clustering Web App</p>""", unsafe_allow_html=True)
st.markdown("""<p id="caption-1z2x">This app helps to recommend treatments for credit card customers!</p>""", unsafe_allow_html=True)

st.sidebar.header('User Input Features')
st.sidebar.text("Use these widgets to input values")

# user inputs that will be retrieved from sidebar 
def user_input_features():
    BALANCE = st.sidebar.slider('BALANCE', 0.0, 20000.0, 10000.0)
    BALANCE_FREQUENCY = st.sidebar.slider('BALANCE_FREQUENCY', 0.0, 1.0, 0.5)
    PURCHASES = st.sidebar.slider('PURCHASES', 0.0, 50000.0, 25000.0)
    ONEOFF_PURCHASES = st.sidebar.slider('ONEOFF_PURCHASES', 0.0, 45000.0, 22500.0)
    INSTALLMENTS_PURCHASES = st.sidebar.slider('INSTALLMENTS_PURCHASES', 0.0, 22500.0, 11250.0)
    CASH_ADVANCE = st.sidebar.slider('CASH_ADVANCE', 0.0, 50000.0, 25000.0)
    PURCHASES_FREQUENCY = st.sidebar.slider('PURCHASES_FREQUENCY', 0.0, 1.0, 0.5)
    ONEOFF_PURCHASES_FREQUENCY = st.sidebar.slider('ONEOFF_PURCHASES_FREQUENCY', 0.0, 1.0, 0.5)
    PURCHASES_INSTALLMENTS_FREQUENCY = st.sidebar.slider('PURCHASES_INSTALLMENTS_FREQUENCY', 0.0, 50000.0, 25000.0)
    CASH_ADVANCE_FREQUENCY = st.sidebar.slider('CASH_ADVANCE_FREQUENCY', 0.0, 1.0, 0.5)
    CASH_ADVANCE_TRX = st.sidebar.slider('CASH_ADVANCE_TRX', 0.0, 150.0, 75.0)
    PURCHASES_TRX = st.sidebar.slider('PURCHASES_TRX', 0.0, 400.0, 200.0)
    CREDIT_LIMIT = st.sidebar.slider('CREDIT_LIMIT', 50.0, 30000.0, 15025.0)
    PAYMENTS = st.sidebar.slider('PAYMENTS', 0.0, 55000.0, 27500.0)
    MINIMUM_PAYMENTS = st.sidebar.slider('MINIMUM_PAYMENTS', 0.0, 80000.0, 40000.0)
    PRC_FULL_PAYMENT = st.sidebar.slider('PRC_FULL_PAYMENT', 0.0, 1.0, 0.5)
    TENURE = st.sidebar.slider('TENURE', 6.0, 12.0, 9.0)
    
    data = {'BALANCE': BALANCE,
            'BALANCE_FREQUENCY': BALANCE_FREQUENCY,
            'PURCHASES': PURCHASES,
            'ONEOFF_PURCHASES': ONEOFF_PURCHASES,
            'INSTALLMENTS_PURCHASES': INSTALLMENTS_PURCHASES,
            'CASH_ADVANCE': CASH_ADVANCE,
            'PURCHASES_FREQUENCY': PURCHASES_FREQUENCY,
            'ONEOFF_PURCHASES_FREQUENCY': ONEOFF_PURCHASES_FREQUENCY,
            'PURCHASES_INSTALLMENTS_FREQUENCY': PURCHASES_INSTALLMENTS_FREQUENCY,
            'CASH_ADVANCE_FREQUENCY': CASH_ADVANCE_FREQUENCY,
            'CASH_ADVANCE_TRX': CASH_ADVANCE_TRX,
            'PURCHASES_TRX': PURCHASES_TRX,
            'CREDIT_LIMIT': CREDIT_LIMIT,
            'PAYMENTS': PAYMENTS,
            'MINIMUM_PAYMENTS': MINIMUM_PAYMENTS,
            'PRC_FULL_PAYMENT': PRC_FULL_PAYMENT,
            'TENURE': TENURE}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.text("")
st.text("")

# User inputs shown as a DataFrame
st.text("User Input Dataframe:")
st.write(input_df)

# read MinMaxScaler and PCA model
load_scaler = pickle.load(open('scaler_fitted.pkl', 'rb'))
load_pca = pickle.load(open('pca_fitted.pkl', 'rb'))

centers = np.array([[-0.59083384, -0.7881446 ,  0.03965857],
                    [ 1.4101118 , -0.19984676, -0.02020388],
                    [-0.16261439,  0.35734836, -0.88594748],
                    [ 0.48122374,  0.80464824, -0.02944048],
                    [-0.98090311,  0.1488673 , -0.1214479 ],
                    [ 0.06820227, -0.22927245,  0.89944224],
                    [-0.33326958,  0.63722453,  0.72909274]])

# predict customer label to the nearest cluster
def predict_customer(data):
    logscaled_data = np.log2(data + 0.01)
    scaler_result = load_scaler.transform(logscaled_data.values.reshape(1,-1))
    
    pca_data = load_pca.transform(scaler_result)
    top3_pc = pca_data[:,:3].reshape(-1)
    
    # applying euclidean distance method
    list_distance = []
    for i in centers:
        distance = np.sqrt((top3_pc[0] - i[0])**2 + (top3_pc[1] - i[1])**2 + (top3_pc[2] - i[2])**2)
        list_distance.append(distance)
        
    return np.argmin(list_distance)

prediction = predict_customer(input_df)

# Details on every cluster and recommended treatment
explanation = [["Using Installment Purchase Only", "They make expensive purchases using installments and their credit's balance is low. We can give them promotions for cheap item purchases using their Credit Card (e.g: Buy 1 Get 1 Starbucks Promo). The idea of this promotion is to encourage more frequent purchases within their behavior, we hope that through the discounts we can improve this clusters retention and improve the consistency of revenue."],
               ["Using Cash Advances Only", "This is the largest group among the 6 clusters. The marketing strategy for this cluster is to increase their credit limit for Cash Advance & to reduce the cash advances interest. Through increasing the limit & reducing interest we will put ourselves in a position over our competitors since we have our users best interest. Again we hope to improve retention and frequency with this strategy. Since this is the largest group amongst the customer base, we must prioritize our strategies here."],
               ["Using One Off Purchase Only", "They tend to make cheap purchases using One Off Purchase. We can give them higher One Off credit limit to keep them using One Off Purchases."],
               ["Less Spenders but Not Using Installment Purchases", """Less Spenders but Not Using Installment Purchases - Marketing strategy for this cluster is to increase the limit of cash advance and one off purchases. Every other purchase could be awarded with 0% interest for installment purchases. This would keep customers purchasing and chase the 0% interest reward."""],
               ["Big Spenders but Not Using Cash Advance", "Big Spenders but Not Using Cash Advance - There are no treatments for this Cluster because this group is using credit card very well and likely to continue to do so. It is better to focus on users that uses cash advances so that we can gain interest."],
               ["Low Financial", "This appears to be a small group of customers that using installment and cash advance from credit card. It is best to not focus strategies to this cluster as well, mainly due to its small group size."],
               ["Highest Average Credit Limit with All Types of Purchases", "The best marketing strategy for this cluster is to give credit points for every time they make transaction using credit card. These points would incentivies the customers to be loyal to the bank and reduce churn rates."]]

st.markdown(f"""<div id="fill-recommendation"></div>""", unsafe_allow_html=True)
st.markdown(f"""<span id="recommendation-1z2x">Cluster {prediction}:</span><span id="cluster-label">&nbsp;&nbsp;{explanation[prediction][0]}</span>""", unsafe_allow_html=True)
st.markdown(f"""<p id="explanation-1z2x">{explanation[prediction][1]}</p>""", unsafe_allow_html=True)

# Display customer into highlighted clusters
dbscan_labels = cluster_json["color"]

def getRestOfTheClusters(prediction):
    """Function to pop a label out"""
    labels = [i for i in range(7)]
    labels.pop(prediction)
    labels = [str(i) for i in labels]
    return ", ".join(labels)

restOfTheClusters = getRestOfTheClusters(prediction)
# Move the desired label at the 0th index
if prediction:
    indexOfLabel = np.where(dbscan_labels == prediction)[0][0] # get first tuple, then get first index
    dbscan_labels[0], dbscan_labels[indexOfLabel] = dbscan_labels[indexOfLabel], dbscan_labels[0]

fig = px.scatter_3d(x = cluster_json["x"], 
                    y = cluster_json["y"], 
                    z = cluster_json["z"],
                    color = np.where(dbscan_labels == prediction, f"Cluster {prediction}", f"Cluster {restOfTheClusters}"),
                    color_discrete_sequence=["#3EFE06", "#C3C3C3"])

# display Plotly with Streamlit
st.plotly_chart(fig, use_container_width=True)
