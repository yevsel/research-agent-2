from pydantic import BaseModel, Field
from typing import List


class Reporter(BaseModel):
    name: str = Field(description="The name of the reporter")
    institution: str = Field(description="The institution the reporter is affiliated with eg. University etc")
    role: str = Field(description="Role of the reporter")
    description: str = Field(description="Description of the reporter focus, concerns and motives")
    
    @property
    def profile(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.institution}\nDescription: {self.description}"
                

class ReporterTeam(BaseModel):
    reporters: List[Reporter] = Field(description="This is a team of reporters")
    
class TavilySearchQuery(BaseModel):
    tavily_search_query: str = Field(description="Reporter interview research response")