# api/models.py
from pydantic import BaseModel
from typing import Optional

class IndividualCreate(BaseModel):
    age: int
    fnlwgt: int
    educational_num: int
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 40
    income_greater_50k: bool = False
    gender: Optional[str] = None
    workclass: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    country: Optional[str] = None

class IndividualUpdate(BaseModel):
    age: Optional[int] = None
    fnlwgt: Optional[int] = None
    educational_num: Optional[int] = None
    capital_gain: Optional[int] = None
    capital_loss: Optional[int] = None
    hours_per_week: Optional[int] = None
    income_greater_50k: Optional[bool] = None
    gender: Optional[str] = None
    workclass: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    country: Optional[str] = None

class IncomeLogCreate(BaseModel):
    individual_id: int
    action: str

class IncomeLogUpdate(BaseModel):
    individual_id: Optional[int] = None
    action: Optional[str] = None

# MongoDB-specific models
class MongoIncomeLogCreate(BaseModel):
    individual_id: str  # String to represent MongoDB ObjectId
    action: str

class MongoIncomeLogUpdate(BaseModel):
    individual_id: Optional[str] = None  # String to represent MongoDB ObjectId
    action: Optional[str] = None