import os
import tempfile
import streamlit as st
from streamlit_chat import message
from multi_modal_rag import MultiModalRAG

from multimodal_rag import Summarizer

#adds a title for the web page

st.set_page_config(page_title="Multimodal RAG Bot")

def display_messages():

    st.subheader('Chat')

    # for displaying the chat between bot and user 

    for i,(msg,is_user) in enumerate(st.session_state["messages"]):

        message(msg,is_user=is_user,key=str(i))


    st.session_state["thinking_spinner"] = st.empty()

def process_image():

    # the textual description of the image is obtained.

    st.session_state["summarized_image"] = []

    if st.session_state["image_uploader"]:
            
            for i,file in enumerate(st.session_state["image_uploader"]):

                with open("temp_"+str(i)+".jpg", "wb") as f:
                    f.write(file.read())

                st.session_state["summarized_image"].append(Summarizer.summarize_image("temp_"+str(i)+".jpg"))

                os.remove("temp_"+str(i)+".jpg")


def process_input():

    process_image() # the textual summary obtained 

    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:

        # if image has been uploaded summaries of the image 
        # is passed along with the user query

        if st.session_state["summarized_image"]: 

            summ_ = ""

            for summ in st.session_state["summarized_image"]:
                summ_+=summ

            image_prompt = f'''
                        Here is a summarized version of the provided image:{summ_}
                        '''
            txt_prompt = f'''
                        Here is the user query : {st.session_state["user_input"].strip()}

                        '''
            
            user_text = image_prompt + txt_prompt
        

        else:

            user_text = st.session_state["user_input"].strip()


        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].answer_query(user_text)

        st.session_state["messages"].append((user_text,True))
        st.session_state["messages"].append((agent_text,False))


def read_and_save_file():

    st.session_state["assistant"].clear() # database cleared when new pdf inserted


    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    st.session_state["p_id"]=""

    for i,file in enumerate(st.session_state["file_uploader"]):


        # open the pdf file and store the pdf file in a temporary location
        with open("temp_"+str(i)+".pdf", "wb") as f:
            f.write(file.read())

        
    

        # insert the data in the pdf file into database
        st.session_state["assistant"].ingest_data("temp_"+str(i)+".pdf")

        #remove the pdf file once the data has been inserted
        os.remove("temp_"+str(i)+".pdf")

def page():

    if len(st.session_state)==0:

        st.session_state["messages"] = []
        st.session_state["assistant"] = MultiModalRAG()

    st.header("MultiModal RAG Chatbot")

    st.subheader("Upload a pdf file")

    st.file_uploader(
        "Upload pdf",
        type = "pdf",
        key = "file_uploader",
        on_change = read_and_save_file,
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()

    st.file_uploader(
        "Upload image file",
        type = ["jpg","png"],
        key = "image_uploader",
        accept_multiple_files=True,
    )

    st.text_input("Message",key="user_input",on_change=process_input)


if __name__ == "__main__":
    page()


        