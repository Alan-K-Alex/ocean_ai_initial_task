import shutil
from retriever import Retriever
from data_classes import RAGDataType
from chat_models import chat_model
from operator import itemgetter
from typing import List, Union, ByteString
from langchain_core.runnables import RunnableLambda
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser 


class MultiModalRAG:

    """
    Multi-modal RAG class to answer queries using context documents.
    """

    def __init__(self,docs_dir=""):

        """Initialize MultiModalRAG class.

        Args:
            docs_dir (str): Path to documents directory
        """

        self.docs_dirs = docs_dir
        self.retriver: Retriever = Retriever()
        self.retriever_chain = ""

    def ingest_data(self,pdf_path):

        # data is ingested into the database once the path of pdf file is passed

        self.docs_dirs = pdf_path
        self.retriver.ingest_data(pdf_path)

    def clear(self):

        # for clearing the contents in vector database 
        # when the input pdf is changed/deleted 

        shutil.rmtree("./chroma_db", ignore_errors=True)

    @staticmethod
    def create_query_context_prompt(args) -> List[HumanMessage]:

        """
        Create query context prompt using the retrieved documents from database
        """


        print("Creating query context prompt")


        query: str = args['query']
        retrieved_data: List[Document] = args['retrieved_data']
        
        messages: List[str] = []
        text_docs: List[str] = [doc.page_content for doc in retrieved_data if doc.metadata['data_type'] == RAGDataType.TEXT.value]
        table_docs: List[str] = [doc.page_content for doc in retrieved_data if doc.metadata['data_type'] == RAGDataType.TABLE.value]
        imgs_docs: List[str] = [doc.page_content for doc in retrieved_data if doc.metadata['data_type'] == RAGDataType.IMAGE.value]
        
        print("Retrieved data: ", len(retrieved_data))
        print("Text docs: ", len(text_docs))
        print("Table docs: ", len(table_docs))
        print("Image docs: ", len(imgs_docs))


        text_message: dict = {
            'type': 'text',
            'text': f"""
You are given a query and you need to answer the query using the context documents (text, tables and images) below.
Query: {query}

Context documents:
{"\n\n".join(text_docs)}
"""
        }


        table_message: dict = {
            'type': 'text',
            'text': f"""
{"\n\n".join(table_docs)}
""" 
        }


        image_message: dict = {
            'type': 'text',
            'text': f"""
{"\n\n".join(imgs_docs)}
"""
        }

        # all types of retrievd documenst are combined into a single prompt
        messages: List[dict] = [ text_message,table_message, image_message] 


        return [HumanMessage(content=messages)]

    def answer_query(self, query: str) -> str:
        
        if self.docs_dirs=="":
            # if no pdf uploaded the following message is thrown

            return "Please Insert a proper file"
        
        # passes the query to retriver
        self.retriever_chain = (
            "" if itemgetter('query')==None else itemgetter('query')
                |
            self.retriver.retriever
        )

        # consists of the retriever chain which passes the result to the prompt
        #  where it is augmented and passed to the chat model for answer generation.
        
        generate_answer_chain = (
            {'query': itemgetter('query'), 'retrieved_data': self.retriever_chain}
                |
            RunnableLambda(MultiModalRAG.create_query_context_prompt)
                | 
            chat_model
                
        )

        
        answer: str = generate_answer_chain.invoke({'query': query})

        # Final Output is refined using StrOutputParser

        return StrOutputParser().parse(answer.content)