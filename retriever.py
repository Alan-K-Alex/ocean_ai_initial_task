import sys
import uuid
import streamlit as st
from data_ingestor import DataInstance
from data_ingestor import DataSummaryInstance
from data_ingestor import DataIngestor
from typing import List, Union, ByteString
from langchain_chroma import Chroma # vector database to store embeddings
from langchain.storage import InMemoryStore
from langchain_core.documents.base import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from chat_models import hf_embedding


class Retriever:

    """
    Retriever class to ingest data and retrieve data from db
    """


    def __init__(self):
        """
        Initialize Retriever class.
        """        
        self.data_ingestor: DataIngestor = None
        self.vector_db: Chroma = None
        self.doc_db: InMemoryStore = None
        self.retriever: MultiVectorRetriever = None



    def ingest_data(
            self, docs_dir: str, 
            collection_name: str = "mm_rag"
        ) -> None:
        """Ingest data into vector database(here Chroma is used).

        Args:
            docs_dir (str): Path to documents directory
            collection_name (str, optional): Name of collection in db. Defaults to "mm_rag".
        """   


        #Database is setup

        self.vector_db: Chroma = Chroma(
            collection_name=collection_name,
            embedding_function=hf_embedding,
            persist_directory="./chroma_db",
            create_collection_if_not_exists=True
        )

        # original data values are stored in InMemoryStore along with the id 
        # original values corresponding to to the top k ids retrived be used

        self.doc_db = InMemoryStore()
        self.data_ingestor = DataIngestor(docs_dir)
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_db,
            docstore=self.doc_db,
        )

        # pdf is located 
        self.data_ingestor.locate_data()

        # data (in form of tables ,images,text) is extracted from the pdf
        data_instances: List[DataInstance] = self.data_ingestor.extract_text_tables_images()

        if len(data_instances) == 0:
            print("No data instances found, exiting")
            sys.exit(0)

        # summarize the data extracted from the pdf before storing in db
        data_summaries: List[DataSummaryInstance] = self.data_ingestor.summarize_text_tables_images(data_instances)

        with st.spinner("Ingesting Data"):# spinner showing ingestion step have started
            self._ingest_data_into_db(data_summaries)




    def _ingest_data_into_db(self, data_summaries: List[DataSummaryInstance]):

        ids = [str(uuid.uuid4()) for _ in data_summaries] #ids are generated for each  data instance

        summary_docs = [
            Document(
                page_content=d.summary, 
                metadata={
                    'doc_id': ids[i], 
                    "data_type": d.data_type.value
                }
            ) for i, d in enumerate(data_summaries)
        ]


        docs = [
            (ids[i], Document(
                page_content=d.data, 
                metadata={
                    'doc_id': ids[i],
                    "data_type": d.data_type.value
                }
            )) for i, d in enumerate(data_summaries)
        ]


        print("Adding documents to db")
        print("Number of summary documents to add: ", len(summary_docs))
        print("Number of documents to add: ", len(docs))

        #Documents are added to vector database
        self.retriever.vectorstore.add_documents(documents=summary_docs,ids=ids)

        #Documents are added to InMemoryStore where original data values are preserved
        self.retriever.docstore.mset(docs)
        print("Data ingested into db")