from pydantic import BaseModel
from typing import List, Optional



class Product(BaseModel):
    name: str
    brand: Optional[str] = None
    price: Optional[str] = None
    url: Optional[str] = None
    image_url: Optional[str] = None