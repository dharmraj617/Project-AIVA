from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class ChatbotResult(BaseModel):
    chatbot_result: str = Field(description="Provide answer for given question about the college.")

    def to_dict(self):
        return {
            "Answer":self.Answer
        }


chatbot_result:PydanticOutputParser = PydanticOutputParser(pydantic_object=ChatbotResult)