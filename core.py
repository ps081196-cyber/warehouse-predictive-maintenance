import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATURES=["temperature_c","vibration_mm_s","runtime_hours","days_since_service","error_count_7d","load_percent"]

def generate_assets(n=900,seed=42):
    r=np.random.default_rng(seed)
    df=pd.DataFrame({"temperature_c":r.normal(58,13,n).clip(20,105),"vibration_mm_s":r.gamma(2.2,1.5,n),"runtime_hours":r.integers(100,18000,n),"days_since_service":r.integers(1,500,n),"error_count_7d":r.poisson(2,n),"load_percent":r.uniform(20,110,n)})
    z=(df.temperature_c-65)/13+(df.vibration_mm_s-3)/2+df.days_since_service/260+df.error_count_7d/3+df.load_percent/100-2.2
    p=1/(1+np.exp(-z))
    df["failure_30d"]=(r.random(n)<p).astype(int)
    df.insert(0,"asset_id",[f"EQ-{i+1:04d}" for i in range(n)])
    return df.round(2)

def train_model(df):
    tr,te=train_test_split(df,test_size=.25,random_state=42,stratify=df.failure_30d)
    model=GradientBoostingClassifier(random_state=42).fit(tr[FEATURES],tr.failure_30d)
    auc=roc_auc_score(te.failure_30d,model.predict_proba(te[FEATURES])[:,1])
    return model,auc

def prioritize(model,df):
    out=df.copy()
    out["failure_probability"]=model.predict_proba(out[FEATURES])[:,1]
    out["priority"]=pd.cut(out.failure_probability,[-1,.3,.6,1],labels=["Monitor","Plan Service","Urgent"])
    return out.sort_values("failure_probability",ascending=False)
