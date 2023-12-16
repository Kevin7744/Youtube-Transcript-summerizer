# Youtube trancript insights
This project covers how to load documents from youtube transcripts, which is the text version of the youtube video.
Use text splitters to remove any overlaps and clean the data. Use vector to store the transcript in numerical format that can be used as the knowledge base.

## Prerequisites and packages
- Python version 3.8 or *higher*
- Langchain, openai, streamlit *(For building the interfaces)*, python-dotenv *(to securely access the .env file where the API keys reside)*.
    *pip install openai streamlit python-dontenv*

**API reference for this tasks**
- Langchain youtube transcript API loader
    *--pip install youtube-transcript-api--*
  
- Python virtual enviroment
    *python -m venv .venv


### How it works?
Paste the video url of the youtube video you want to use as the knowledge base and ask the any it question concerning the video and thats the ouput.


