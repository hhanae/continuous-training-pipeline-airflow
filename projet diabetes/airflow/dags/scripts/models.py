from pydantic import BaseModel

class Diabetes(BaseModel):
    Pregnancies:int
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:float
    DPF:float
    Age:int