import enum
from sqlalchemy import DateTime, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class SCHEMAS(enum.Enum):
    METADATA = 'metadata'
    ENTITY_SETS = 'entity_sets'
    FEATURES = 'features'
    PREDICTIONS = 'predictions'
    METRICS = 'metrics'

class EntitySetRegister(Base):
    __tablename__ = "EntitySetRegister"
    __table_args__ = {"schema": SCHEMAS.ENTITY_SETS.value}

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
    __table_args__ = {"schema": SCHEMAS.ENTITY_SETS.value}

    id = Column(Integer, primary_key=True)
    entity_set_id = Column(Integer, ForeignKey(EntitySetRegister.id, ondelete="CASCADE"))
    member_id = Column(Integer, nullable=False) # ID of the member

    entity_set = relationship("EntitySetRegister", back_populates="members")

class ModelRegister(Base):
    __tablename__ = "ModelRegister"
    __table_args__ = {"schema": SCHEMAS.METADATA.value}

    id = Column(Integer, primary_key=True)
    model_uri = Column(String, nullable=False)  # model_uri in mlflow

class ProductionModel(Base):
    __tablename__ = "ProductionModel"
    __table_args__ = {"schema": SCHEMAS.METADATA.value}

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    model_id = Column(Integer, ForeignKey(ModelRegister.id), nullable=False)

    model = relationship("ModelRegister")

class ProductionHistory(Base):
    __tablename__ = "ProductionHistory"
    __table_args__ = {"schema": SCHEMAS.METADATA.value}

    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    model_id = Column(Integer, ForeignKey(ModelRegister.id), nullable=False)
    promoted_at = Column(DateTime, nullable=False)

    model = relationship("ModelRegister")

class FeatureRegistry(Base):
    __tablename__ = "FeatureRegistry"
    __table_args__ = {"schema": SCHEMAS.METADATA.value}

    # AUTOMATIC:
    id = Column(Integer, primary_key=True)
    table_name = Column(String, unique=True, nullable=False) # name of the table where feature is stored; 

    # USER-CONTROLLED
    feature_name = Column(String, unique=True, nullable=False) # 
    entity_id_name = Column(String, nullable=False) # This is a human-readable string identifying the type of entity this features is associated with; 
    feature_type = Column(String, nullable=False) # STATES vs EVENTS
    data_type = Column(String, nullable=False) # int/bool/string/etc.
    stale_after_n_days = Column(Integer, nullable=True)  # if None, never stale

    versions = relationship("FeatureLog",back_populates="feature",cascade="delete")

class FeatureLog(Base):
    __tablename__ = "FeatureLog"
    __table_args__ = {"schema": SCHEMAS.METADATA.value}

    id = Column(Integer, primary_key=True)
    feature_id = Column(Integer, ForeignKey(FeatureRegistry.id), nullable=False)
    version = Column(Integer, nullable=False)  # version of the feature
    created_at = Column(DateTime, nullable=False)
    description = Column(String, nullable=False) # description of a feature; changing this text in FeatureMetadata object triggers new version
    version_description = Column(String, nullable=True) # description of this version change; changing this text in FeatureMetadata object triggers new version

    feature = relationship("FeatureRegistry", back_populates="versions")