# Mulitmodal RAG Pipeline for PDF Question Answering 

The diagram presented below illustrates the architecture of our Retrieval-Augmented Generation (RAG) system. The pipeline commences with an extraction phase, utilizing the Unstructured library. This library features a function known as `partition_pdf`, which effectively extracts and segregates text, images, and tables. The processing of the PDF employs a "by_title" chunking strategy. Each data instance from the various data types is subsequently summarized.

For the image data, we convert it into a corresponding textual description employing a Visual Question Answering model, specifically the Llama-4-Scout-17B-16E-Instruct. The summarized instances of data are then inserted into a vector database, for which we utilize ChromaDB.

Responses to user inquiries are generated by retrieving pertinent information from the database and integrating it with the user’s query. Additionally, if an image is provided by the user, it is first transformed into a textual description using the aforementioned VQA model before being incorporated with the other prompts.el and then included with the other prompts.

Tech Stack used : Python ,Langchain ,ChromaDB ,Streamlit

<img src="https://github.com/user-attachments/assets/7151e40e-eed7-42d6-873d-87469f383695" width="512" height="512" >

## Steps

1. Create an API token in both [HuggingFace](https://huggingface.co) and [GroqCloud](https://console.groq.com/playground) and paste in .env file
   
   Caution: The API by HuggingFace has a monthly limit; keep that in mind
   
![image](https://github.com/user-attachments/assets/dbe708ec-c26f-4314-82c7-c2a0a27b4a91)

2. Create a Python environment. I have used conda, but you are free to use any approach below. It is advised to go with a Python version==3.12.5 or above

   Approach 1:
   ```sh
   conda create -n "myenv" python=3.12.5
   conda activate myenv
   ```
      
     
   Approach 2:
```sh
python -m venv .venv
# activate the venv
# Linux/MacOS
source .venv/bin/activate
# Windows
.venv/Scripts/activate
  ```
3 . Install the dependencies
```sh
pip install -r requirements.txt
```
4. Install tesseract-ocr from this [Link](https://github.com/UB-Mannheim/tesseract/wiki) and setup, note the path to the  tesseract-ocr folder and add the path in the environmental variables,
   you can use the following [link](https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i) for reference
   Make sure that you give the correct path for the tesseract.exe file in the code shown below as it may differ for you
   ![image](https://github.com/user-attachments/assets/51481339-01cd-43ae-addb-2c7ef110124f)
   
5. Run the app using the following command
```sh
streamlit run multimodal_rag_app.py
```


## Output

You can upload a pdf and ask any question on it. 

<img src="https://github.com/user-attachments/assets/b76ff104-6cd2-4e84-bfd5-da01c6d5bc44" width="700" height="650" >

You can also upload a image /screenshot related with the pdf and it will give answers ,thanks to powerful VQA llama model 

<img src="https://github.com/user-attachments/assets/c4c3fa8b-26cf-46fe-97c7-ecdcacd06b2d" width="900" height="512" >

Below given is a response answer to the above scenario.

<img src="https://github.com/user-attachments/assets/f67cfcda-f15b-4aca-b0ba-ef473320174e" width="900" height="630" >

