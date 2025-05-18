from enum import Enum, auto
from typing import List, Union, ByteString
from dataclasses import dataclass

class RAGDataType(Enum):

    """
    Data types for RAG model.
    """    

    TEXT = auto()
    TABLE = auto()
    IMAGE = auto()


@dataclass
class DataInstance:

    """
    Data class for data instance to be stored in db.
    """

    data_type: RAGDataType
    data: Union[ByteString, str]


@dataclass
class DataSummaryInstance:

    """
    Data class for summarized data instance.
    """

    data_type: RAGDataType
    data: Union[ByteString, str]
    summary: str