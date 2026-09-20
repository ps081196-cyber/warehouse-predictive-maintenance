from core import generate_assets,train_model,prioritize

def test_priority_pipeline():
    data=generate_assets(250,3)
    model,auc=train_model(data)
    result=prioritize(model,data.drop(columns=["failure_30d"]))
    assert result.failure_probability.between(0,1).all()
    assert set(result.priority.astype(str))<= {"Monitor","Plan Service","Urgent"}
    assert auc>.5
