import os
import os.path as osp
import shutil
import streamlit as st
from typing import List, Union, ByteString
from unstructured.partition.pdf import partition_pdf 
from summarizer import Summarizer
from data_classes import DataInstance
from data_classes import DataSummaryInstance
from data_classes import RAGDataType
import torch

torch.classes.__path__ = []

import pytesseract  #crucial for performing ocr task (eg:- images maybe present in pdfs)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class DataIngestor:

    """
    Data Ingestor class to ingest data from documents.
    """


    def __init__(self, docs_dir: str, image_output_dir_path: str = 'figures'):

        """Initialize DataIngestor class.

        Args:
            docs_dir (str): Path to documents directory
            image_output_dir_path (str, optional): Directory for storing extracted images. Defaults to 'figures'.
        """

        self.docs_dir: str = docs_dir
        self.image_output_dir_path = image_output_dir_path
        self.document_paths: List[str] = []
        shutil.rmtree(self.image_output_dir_path, ignore_errors=True)

    def locate_data(self):

        """
        Locate documents in the directory.
        """

        with st.spinner("Locating"): #a spinner to show the status that file is being located
            file = self.docs_dir
            if file.endswith(".pdf"):
                    self.document_paths.append(file)
        
        print("Located documents: ", len(self.document_paths))

    def extract_text_tables_images(self) -> List[DataInstance]:

        """
        Extract text, tables and images from documents.
        """

        data_instances: List[DataInstance] = []

        # if a directory to store images doesn't exist create it
        if not os.path.exists(self.image_output_dir_path):
            os.makedirs(self.image_output_dir_path)

        with st.spinner("Extracting Data"):#spinner to show that data is being extracted
            
            # data from pdf is extracted using partition_pdf function from unstructured library

            for doc_path in self.document_paths:
                if doc_path.endswith(".pdf"):
                    pdf_elements = partition_pdf(
                        filename=doc_path,
                        extract_images_in_pdf=True,
                        infer_table_structure=True,
                        chunking_strategy="by_title",
                        strategy='hi_res',
                        mode='elements',
                        max_characters=4000,
                        image_output_dir_path=self.image_output_dir_path
                    )
                    for element in pdf_elements:
                        if element.category == 'Table':
                            data_instances.append(DataInstance(RAGDataType.TABLE, element.text))
                        elif element.category == 'CompositeElement':
                            data_instances.append(DataInstance(RAGDataType.TEXT, element.text))
                        else:
                            print('Unsupported element category: ', element.category)

            for file in os.listdir(self.image_output_dir_path):
                if file.endswith(".jpg") or file.endswith(".png"):
                    data_instances.append(DataInstance(RAGDataType.IMAGE, osp.join(self.image_output_dir_path, file)))
        
        print("Extracted data instances: ", len(data_instances))
        return data_instances
    
    def summarize_text_tables_images(self, data_instances: List[DataInstance]) -> List[str]:

        """Summarize text, tables and images.

        Args:
            data_instances (List[DataInstance]): List of data instances

        Raises:
            ValueError: Unsupported data type

        Returns:
            List[str]: List of summarized data instances
        """    


        with st.spinner("Summarizing data instances"):  #spinner to show data is being summarized
            summaries: List[DataSummaryInstance] = []
            datatype_counts = {RAGDataType.TEXT: 0, RAGDataType.TABLE: 0, RAGDataType.IMAGE: 0}
            for data_instance in data_instances:
                summary: str = None
                if data_instance.data_type == RAGDataType.TEXT:
                    summary = Summarizer.summarize_text(data_instance.data)
                elif data_instance.data_type == RAGDataType.TABLE:
                    summary = Summarizer.summarize_table(data_instance.data)
                elif data_instance.data_type == RAGDataType.IMAGE:
                    summary = Summarizer.summarize_image(data_instance.data)
                else:
                    raise ValueError("Unsupported data type")
                if summary is not None:
                    summaries.append(
                        DataSummaryInstance(
                            data_instance.data_type,
                            data_instance.data,
                            summary
                        )
                    )
                    datatype_counts[data_instance.data_type] += 1

        print("Summarized data instances: ", len(summaries))
        print("Data type counts: ", datatype_counts)

        return summaries