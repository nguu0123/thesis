import uuid 
class Agent(object):
    def __init__(self):
        self.id = uuid.uuid4
    
class Entity():
    def __init__(self, data):
        self.id = uuid.uuid4
        self.data = data
    def derive(self, other_entity: Entity):
        other_entity.wasDerivedFrom = self.id

class Activity():
    def __init__(self) -> None:
        self.id = uuid.uuid4
        self.entityUsed = []
    def use(self, dataFunction, entity):
        self.entityUsed.append(entity) 
        generatedEntity =  Entity(dataFunction(entity.data)) 
    def generate(self, entity):
        entity.wasGeneratedBy = self

