from sqlalchemy import DateTime, create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class EntitySetRegister(Base):
    __tablename__ = "EntitySetRegister"
    __table_args__ = {"schema": "_metadata"}

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    entity_id_name = Column(String, nullable=False)

    members = relationship(
        "EntitySetMember",
        back_populates="entity_set",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class EntitySetInfo:
    '''
    Info about an Entity Set - returned by the FS_client.retrieve_entity_set method
    '''
    def __init__(self, name: str, description: str, entity_id_name: str, members: list):
        self.name = name
        self.description = description
        self.entity_id_name = entity_id_name
        self.members = members

class EntitySetMember(Base):
    __tablename__ = "EntitySetMemberList"
    __table_args__ = {"schema": "_metadata"}

    id = Column(Integer, primary_key=True)
    entity_set_id = Column(Integer, ForeignKey(EntitySetRegister.id, ondelete="CASCADE"))
    member_id = Column(Integer, nullable=False) # ID of the member

    entity_set = relationship("EntitySetRegister", back_populates="members")

class ModelRegister(Base):
    __tablename__ = "ModelRegister"
    __table_args__ = {"schema": "_metadata"}

    id = Column(Integer, primary_key=True)
    model_uri = Column(String, nullable=False)  # model_uri in mlflow

class ProductionModel(Base):
    __tablename__ = "ProductionModel"
    __table_args__ = {"schema": "_metadata"}

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    model_id = Column(Integer, ForeignKey(ModelRegister.id), nullable=False)

    model = relationship("ModelRegister")

class ProductionHistory(Base):
    __tablename__ = "ProductionHistory"
    __table_args__ = {"schema": "_metadata"}

    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    model_id = Column(Integer, ForeignKey(ModelRegister.id), nullable=False)
    promoted_at = Column(DateTime, nullable=False)

    model = relationship("ModelRegister")

