# def power_pred(x:float , power:int) -> float:
#     return x**power


import joblib
from pathlib import Path
import torch
import torch.nn as nn
model_path = Path(__file__).parent.parent /"torch_model.pt"

# class MLModel:
#     def __init__(self):
#         self.model = joblib.load(model_path)
#     def predict(self,x:float)-> float:
#         return float(self.model.predict([[x]])[0])
    
#     def predict_batch(self,xs:list[float])-> list[float]:
#         return [self.predict(x) for x in xs]
# ml_model = MLModel()


class LinearModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)
    
class MLModel:
    def __init__(self):
        self.model=LinearModel()

        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    def predict(self,x:float):
        with torch.no_grad():
            inp = torch.tensor([[x]],dtype=torch.float32)
            out= self.model(inp)
            return float(out.item())
        
    def predict_batch(self,xs:list[float])-> list[float]:
        return [self.predict(x) for x in xs]
    

ml_model = MLModel()