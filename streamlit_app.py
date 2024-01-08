import streamlit as st
import whisper
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Function to load Whisper model
def load_whisper_model(model_name):
    return whisper.load_model(model_name)

# Function to transcribe audio
def transcribe_audio(model, audio):
    with open(audio, "wb") as f:
        f.write(audio_file.getvalue())
    return model.transcribe(audio, fp16=False)

# Function to process query
def process_query(transcription, query, audio):
    transcript_doc = Document(page_content=transcription, metadata={"source": audio})
    alt_rag_template = """
        You answer questions about the contents of a transcribed audio file. 
        Use only the provided audio file transcription as context to answer the question. 
        Do not use any additional information.
        If you don't know the answer, just say that you don't know. Do not use external knowledge. 
        Use three sentences maximum and keep the answer concise. 
        Make sure to cite references by referencing quotes of the provided context. Do not use any other knowledge.
        
        \nQuestion: {question} \nContext: {context} \nAnswer:
        """
    alt_prompt = PromptTemplate.from_template(alt_rag_template)
    alt_rag_prompt = alt_prompt.format(context=transcript_doc.page_content, question=query)
    return llm(alt_rag_prompt)

# Initialize models
model_name = st.sidebar.selectbox("Select Whisper Model", ["base", "small", "medium", "large"])
model = load_whisper_model(model_name)
llm = Ollama(model='llama2', temperature=0)

# Streamlit UI Setup
st.title("Audio Transcription and Query Answering App")

audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "flac"])

if audio_file is not None:
    audio = audio_file.name
    result = transcribe_audio(model, audio)
    transcription = result["text"]
    # Use a dropdown (expander) for transcription
    with st.expander("View Transcription"):
        st.write(transcription)

    query = st.text_input("Enter your query about the text")
    if query:
        answer_alt = process_query(transcription, query, audio)
        st.write("Answer:", answer_alt)
