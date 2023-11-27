from pydantic import BaseModel


class Item(BaseModel):
    features: list
