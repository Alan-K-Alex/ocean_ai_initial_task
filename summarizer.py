
import base64
import os.path as osp
from langchain_core.messages import HumanMessage, BaseMessage
from chat_models import chat_model
from chat_models import vision_chat_model


class Summarizer:

    """
    Summarizer class to summarize text, tables and images using chat_model and vision_chat_model of llama.
    """

    @staticmethod
    def summarize_text(text: str) -> str:

        """Summarize text using llama model.

        Args:
            text (str): Text to summarize

        Returns:
            str: Summarized text
        """


        prompt = f'''
You are given a text. Summarize it in a few sentences for semantic retrieval.
Do not include any additional words like Summary: etc.
---
Here is the text:
{text}
'''
        try:
            response: BaseMessage = chat_model.invoke([
                HumanMessage(content=[
                    {'type': 'text', 'text': prompt},
                ])
            ])
            return response.content
        except Exception as e:
            print(f"Error in Summarizer.summarize_text {e}")
            return None

        
    @staticmethod
    def encode_image(image_path: str) -> str:

    
        """Encode image to base64.

        Args:
            image_path (str): Path to image

        Returns:
            str: Base64 encoded image
        """        


        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def summarize_image(image_path: str) -> str:

        """Summarize image using  llama model(vision_chat_model) specially used for image to text tasks.

        Args:
            image_path (str): Path to image

        Returns:
            str: Summarized image
        """        



        prompt = '''
You are given a image. Summarize the image for semantic retrieval. 
Do not include words like "Summary: etc.
'''


        assert osp.exists(image_path), f"Image path does not exist {image_path}"
        base64_image = Summarizer.encode_image(image_path)
        try:
            response: BaseMessage = vision_chat_model.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ])
            ])
            return response.content
        except Exception as e:
            print(f"Error in Summarizer.summarize_image {e}")
            return None
    
    @staticmethod
    def summarize_table(table: str) -> str:


        """Summarize table using llama model.

        Args:
            table (str): Table to summarize

        Returns:
            str: Summarized table
        """    


        prompt = f'''
You are given a table. Summarize the table for semantic retrieval. 
Do not include any additional words like Summary: etc.
---
Here is the table:
{table}
'''
        
        try:
            response: BaseMessage = chat_model.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                ])
            ])
            return response.content
        except Exception as e:
            print(f"Error in Summarizer.summarize_table {e}")
            return None