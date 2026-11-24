# Search-Audio-LLM

Upload an audio file and ask questions about what was said. Everything runs on
your machine: no audio or text leaves your computer.

## How it works
- Transcribes the audio locally with OpenAI's open-source Whisper. You choose
  the model size (`base` to `large`) in the sidebar.
- Answers questions with Llama 2 running on Ollama, using only the transcript as
  context. Answers are short and quote the parts of the transcript they rely on.

## Run locally
Requires [Ollama](https://ollama.com) with the Llama 2 model:
```bash
ollama pull llama2
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Stack
Python · Streamlit · Whisper · LangChain · Ollama (Llama 2)

Built in January 2024.
