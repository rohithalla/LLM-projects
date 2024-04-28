import streamlit as st
import utilities
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from prompt_util import return_map_reduce_llm_response
import os
os.environ['OPENAI_API_KEY'] = 'YOUR-OPENAI-API-KEY'

def langchain_model(model_name,temperature, top_p, max_output_tokens):
  model = OpenAI(model_name=model_name,temperature =temperature,top_p=top_p, max_tokens = max_output_tokens)
  return model

response = []


def fulfillments_generate(split_docs, llm_model):
    global response
    prompt_template = """A document has been submitted. 
    Use information only from below text. 
    Your job is to Extract the information, outline or the summary from the following Document Text """

    prompt_template += """Document Text: 
    ```"{docs}"``` 
    """
    
    map_prompt = PromptTemplate.from_template(prompt_template)

    reduce_template = ("Your job is to create a final summary from the \
    following statements \n"
    "------------\n"
    "{doc_summaries}\n"
    "------------\n")
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    output = return_map_reduce_llm_response(llm_model, map_prompt, 
                                            reduce_prompt, split_docs)
    return output
    

def main():
    global response
    utilities.set_page_config("Extract fulfillments")
    st.title("Upload documents to get summary")
    
    rfp_input_text = ''
    llm_model = langchain_model(model_name="gpt-3.5-turbo-instruct", temperature=0, top_p=0.9, 
                                max_output_tokens=1024)
    print(llm_model)
    uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf", "docx"], 
                                      accept_multiple_files=True)
    st.write(f"**************No of files :{len(uploaded_files)}*************")

    if len(uploaded_files) > 0:
        uploaded_file_gcs_list = []

        for uploaded_file in uploaded_files:
            uploaded_file_gcs_list.append(uploaded_file)
            rfp_input_text = ''
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension == "pdf":
                rfp_input_text = utilities.read_pdf_file(rfp_input_text, uploaded_file)
            if file_extension == "docx":
                rfp_input_text = utilities.read_docx_file(rfp_input_text, uploaded_file)

        # Splitting the text
        split_docs = utilities.get_split_texts(rfp_input_text)

        response = fulfillments_generate(split_docs, llm_model)
        st.write(response)


if __name__ == "__main__":
    main()
