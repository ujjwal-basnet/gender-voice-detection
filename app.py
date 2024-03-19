#core pkgs
import streamlit  as st 

#eda pkgs 
import pandas as pd 
import numpy as np 

#import data visulization pkgs
import seaborn as sns 
import matplotlib.pyplot as plt 

#ml pkgs
import pickle 


#load  pkgs 
model = pickle.load(open('voice_model.pickle' , 'rb'))

#predicting Label given by users information 
def pred_gender(sd, median, Q25,Q75, IQR, skew, spent, mode,centroid, meanfun, minfun, maxfun, meandom, mindom,maxdom, modindx):
   
    
    #output 
    out_input = np.array([[sd, median, Q25,Q75, IQR, skew, spent, mode,centroid, meanfun, minfun, maxfun, meandom, mindom,maxdom, modindx]]).astype(np.float64)
    prediction = model.predict(out_input)
    return prediction


# Set page background image
st.markdown(
    """
    <style>
    body {
        background-image: url("male-vs-female.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)




st.title("Gender Voice Detection")
st.success("Detect weither the voice is Male or Female")

sd = st.text_input (label  = "sd" )
median= st.text_input (label  = "median" )
Q25= st.text_input (label  = " Q25 " )
Q75= st.text_input (label  = " Q75 " )
IQR = st.text_input (label  = " IQR " )
skew = st.text_input (label  = " skew " )
spent= st.text_input (label  = " spent " )
mode = st.text_input (label  = " mode " )
centroid = st.text_input (label  = " centroid " )
meanfun = st.text_input (label  = " meanfun " )
minfun = st.text_input (label  = " minfun " )
maxfun = st.text_input(label = "Maxfun")
meandom = st.text_input (label  = " meandom " )
mindom = st.text_input (label  = " mindom " )
maxdom = st.text_input (label  = " maxdom " )
modindx= st.text_input (label  = " modindx " )


if st.button("Predict"):
    
    output = pred_gender(sd, median, Q25,Q75, IQR, skew, spent, mode,centroid, meanfun, minfun, maxfun, meandom, mindom,maxdom, modindx)
    st.success(f"This IS {output} Voice"  )