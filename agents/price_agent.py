import modal
from agents.agent import Agent




class SpecialistAgent(Agent):

    name= "SpecialistAgent"


    def __init__(self):


        self.log("Specialist Agent is initializing - connecting to modal")
        Pricer = modal.Cls.lookup('price-service', 'Pricer')
        self.pricer  =Pricer()
        self.log("Specialist Agent is ready")

    def price(self, description: str) -> float:
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result       
    
