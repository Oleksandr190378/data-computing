# Chat with AI

This project is a web application that allows users to chat with an AI assistant using the context from an uploaded PDF file. The application is written in Python using FastAPI for the web server and LangChain for language processing.

## Features

- Upload a PDF file and extract text from it.
- Create a vector database for fast advanced retrieval of relevant documents.
- Chat with an AI assistant that uses the context from the uploaded PDF file.
- Save and view chat history.

## Requirements

- Python 3.8 or newer
- FastAPI
- LangChain
- OpenAI API key
- Cohere API key
- rank_bm25

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/Chat_with_AI.git
   cd Chat_with_AI
Create a virtual environment and activate it:

sh
Copy code
python -m venv .venv
source .venv/bin/activate  # For Linux/MacOS
.venv\Scripts\activate  # For Windows
Install the required dependencies:

sh
Copy code
pip install -r requirements.txt
Install rank_bm25:

sh
Copy code
pip install rank_bm25
Create a .env file in the root directory of the project and add your API keys:

env
Copy code
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
Running the Application
Start the FastAPI server:

sh
Copy code
uvicorn app.main:app --host 0.0.0.0 --port 8000
Open your browser and go to:

Copy code
http://127.0.0.1:8000
Usage

Chat with AI Assistant

Go to the /chat_with_history page.

Enter your message and click "Send"

The AI assistant will respond using the `ConversationSummaryBufferMemory` with a `max_token_limit` of 1000, allowing it to remember previous interactions.


Upload a PDF file:

Go to the /upload page.

Select a PDF file to upload.

After a successful upload, you will be redirected to the /upload_success page.

Chat with the AI assistant:

Go to the /chat_with_context page.

Enter your message and click "Send".

The AI assistant will respond using the context from the uploaded PDF file and using the `ConversationSummaryBufferMemory`.

View chat history:

Go to the /history page.

You can view your chat history and delete it if needed.

Project Structure
Copy code
Chat_with_AI/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── chat.py
│   ├── pdf_processing.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── history.html
│   │   ├── upload.html
│   │   ├── upload_success.html
│   │   ├── chat_with_context.html
├── .env
├── requirements.txt
├── README.md
Logging
The project is configured to use logging to track the progress of code execution and detect errors. Logs can be found in the console where the FastAPI server is running.

Contributing
If you would like to contribute to the project, please fork the repository and submit a pull request with your changes. We would be happy to review your suggestions!

License
This project is licensed under the MIT License.

