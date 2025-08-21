from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your pdf")
    st.header("Ask Your PDF")
    #upload the pdf
    pdf=st.file_uploader("Upload your PDF",type="pdf")
    #extract the text from pdf
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        #split the text into chunks
        text_splitter=CharacterTextSplitter(
            separator="\n", 
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text)
        #create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        knowledge_base=FAISS.from_texts(chunks,embeddings)
        #input the question
        user_question=st.text_input('Ask a Question from PDF')
        if user_question:
            docs=knowledge_base.similarity_search(user_question)
            docs_with_scores = knowledge_base.similarity_search_with_score(user_question, k=3)
            if not docs_with_scores or docs_with_scores[0][1] > 1.5: 
                st.warning("I couldn't find anything relevant to your question in the PDF.")
                return
            model_name = "google/flan-t5-small"  # Free, lightweight LLM
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            hf_pipeline = pipeline("text2text-generation",model=model,tokenizer=tokenizer,max_length=512)
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            chain=load_qa_chain(llm,chain_type="stuff")
            response=chain.run(input_documents=docs,question=user_question)
            st.write(response)
            

        
    
        

    





if __name__=="__main__":
    main()
