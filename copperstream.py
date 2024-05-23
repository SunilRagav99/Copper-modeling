import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image
import json 
import requests 
from streamlit_lottie import st_lottie 
from sklearn.preprocessing import StandardScaler
import pickle
ss=StandardScaler()

with open(r'C:\Users\sunil\Downloads\Animation - 1715974125258.json', 'r') as f:
    # Load the JSON data
    data = json.load(f)

icon = Image.open(r"C:\Users\sunil\Downloads\free-search-icon-2911-thumb.png")
st.set_page_config(page_title= "Copper Modeling| By SUNIL RAGAV",
                   page_icon= icon,
                   layout= "wide",
                   initial_sidebar_state= "expanded")


markdown_text = "<h1 style='text-align: center; color: blue;'>Copper Modeling</h1>"

st.markdown(markdown_text, unsafe_allow_html=True)




df1=pd.read_csv(r"C:\Users\sunil\OneDrive\Desktop\python guvi\Copper modeling\coopermodel.csv")
df=df1.copy()
p=df[["quantity tons","country",'status','thickness','width','product_ref',"customer",'item type',"application"]]
c=df[['quantity tons','selling_price','item type','application','thickness','width','country','customer','product_ref']]

status=df["status"].unique()
Item_type=df["item type"].unique()
country=df['country'].unique()
application=df['application'].unique()

col1,col2,col3 = st.columns([4,3,4])
col1.markdown("## :blue[Domain] : Manufacturing")
col1.markdown("## :blue[Technologies used] : Python scripting, Data Preprocessing,EDA, Streamlit")
col1.markdown("## :blue[Overview] : Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.ML Regression model which predicts continuous variable .ML Classification model which predicts Status")

with col3:
   lottie_data = st_lottie(data, reverse=True, height=600, width=400, speed=15, loop=True, quality='high', key='spinner1')

st.markdown("# :blue[User Input:]")
st.markdown("### :blue[Selling Price Prediction:]")
st.markdown("#### :blue[Sample Dataframe of how the input values should look:]")
sam=p.head(1)
i=st.write(sam)
st.markdown("#### :blue[Please Enter the Valid Input:]")

col1,col2,col3=st.columns(3)
with col1:
    quantity_tons = st.text_input('Enter Quantity Tons (Max:1000000000.0)',p["quantity tons"][0])
    thickness = st.text_input('Enter thickness (Min:0.18 & Max:400)',p["thickness"][0])
    width = st.text_input('Enter width(Min:1, Max:2990)',p["width"][0])
with col2:
    customer = st.text_input('customer ID (Min:12458, Max:30408185)',p["customer"][0])
    status=st.selectbox("Select the status",status)
    item_type=st.selectbox("Select the Item type",Item_type)
with col3:
    product_ref=st.text_input("Enter the Product Ref(Min:611728 & Max:1722207579",p["product_ref"][0])
    country=st.selectbox("Select the Country",country)
    application=st.selectbox("Select the Application",application )

with open(r"C:\Users\sunil\OneDrive\Desktop\python guvi\Copper modeling\precit.pkl","rb") as f:
        predict_model=pickle.load(f)
new_sample = np.array([[np.log(float(quantity_tons)),country, status, np.log(float(thickness)), float(width), int(product_ref),float(customer), item_type, application]])

with open(r"C:\Users\sunil\OneDrive\Desktop\python guvi\Copper modeling\Scondition.pkl","rb") as ssc:
       Stand=pickle.load(ssc)


  
load_ss=Stand.transform(new_sample)


load_model=predict_model.predict(load_ss)

s=st.button("click")
if s is True:
    st.write('## :green[Predicted selling price:] ', np.exp(load_model)[0])





with open(r"C:\Users\sunil\OneDrive\Desktop\python guvi\Copper modeling\ScalerClass.pkl","rb") as sc:
       Standard=pickle.load(sc)

with open(r"C:\Users\sunil\OneDrive\Desktop\python guvi\Copper modeling\Classifier.pkl","rb") as cf:
       classification_model=pickle.load(cf)

st.markdown("### :blue[Status Prediction:]")
st.markdown("#### :blue[Sample Dataframe of how the input values should look:]")
samc=c.head(1)
i=st.write(samc)
st.markdown("#### :blue[Please Enter the Valid Input:]")
col1, col2, col3 = st.columns(3)
with col1:
            cquantity_tons = st.text_input('Enter Quantity tons (Min:611728 & Max:1722207579)',df["quantity tons"][0])
            cthickness = st.text_input('Enter Thickness (Min:0.18 & Max:400)',df["thickness"][0])
            cwidth = st.text_input('Enter Width (Min:1, Max:2990)',df["width"][0])
with col2:
            ccustomer = st.text_input('Customer ID (Min:12458, Max:30408185)',df["customer"][0])
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)",df["selling_price"][0])
            cproduct_ref=st.text_input("Enter the Product_Ref(Min:611728 & Max:1722207579",df["product_ref"][0])

Country=df["country"].unique()
Application=df["application"].unique()

with col3:
            ccountry=st.selectbox("Select the country",Country)
            capplication=st.selectbox("Select the Application ",Application )
            citem_type=st.selectbox("Select the Item_type",Item_type)

if ccountry is not None:
       ccountry=float(country)

if capplication is not None:
       capplication=int(capplication)





new_samplec = np.array([[(float(cquantity_tons)), (float(cselling)), float(citem_type),float(capplication),
                                (float(cthickness)), float(cwidth), float(ccountry), float(ccustomer), int(product_ref)]])




load_ssc=Standard.transform(new_samplec)
load_modelc=classification_model.predict(new_samplec)
# st.write(new_samplec)

# st.write(load_ssc)

# st.write(load_modelc)
sr=st.button("Click")
if sr is True:
    if (load_modelc == 7).any():
        st.write('## :green[The Status is Won] ')
    else:
        st.write('## :red[The Status is Loss] ')
