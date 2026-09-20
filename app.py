import plotly.express as px
import streamlit as st
from core import FEATURES,generate_assets,prioritize,train_model

st.set_page_config(page_title="Predictive Maintenance",page_icon="🛠️",layout="wide")
st.title("🛠️ Warehouse Predictive Maintenance")
data=generate_assets()
model,auc=train_model(data)
current=generate_assets(180,9).drop(columns=["failure_30d"])
ranked=prioritize(model,current)
a,b,c=st.columns(3)
a.metric("Validation ROC-AUC",f"{auc:.2f}")
b.metric("Urgent assets",int((ranked.priority=="Urgent").sum()))
c.metric("Average failure risk",f"{ranked.failure_probability.mean():.1%}")
importance=sorted(zip(FEATURES,model.feature_importances_),key=lambda x:x[1])
st.plotly_chart(px.bar(x=[v for _,v in importance],y=[k for k,_ in importance],orientation="h",title="Failure-risk drivers"),use_container_width=True)
st.subheader("Maintenance priority queue")
st.dataframe(ranked,use_container_width=True)
st.download_button("Download maintenance plan",ranked.to_csv(index=False),"maintenance_priorities.csv")
