import tempfile
from io import BytesIO
import logging

from fastapi import FastAPI, Request, Form,  File, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import io
from fastapi.responses import JSONResponse
from app.pdf_processing import load_pdf_file, create_vector_store, load_vector_store, create_ensemble_retriever, \
    create_compression_retriever, vector_store, ensemble_retriever, compression_retriever
from .chat import chat_with_history, save_history_to_file, load_history_from_file, delete_history_file, \
    chat_with_history_context

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    history_text = load_history_from_file()
    return templates.TemplateResponse("index.html", {"request": request, "history": history_text})


@app.post("/chat")
async def chat(request: Request, query: str = Form(...)):
    response_text, history_text = chat_with_history(query)
    save_history_to_file(history_text)
    return templates.TemplateResponse("index.html", {"request": request, "response": response_text, "history": history_text})


@app.get("/history", response_class=HTMLResponse)
async def read_history(request: Request):
    history_text = load_history_from_file()
    return templates.TemplateResponse("history.html", {"request": request, "history": history_text})


@app.post("/delete_history")
async def delete_history(request: Request):
    delete_history_file()
    return templates.TemplateResponse("history.html", {"request": request, "history": ""})


@app.get("/upload", response_class=HTMLResponse)
async def read_upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload_pdf", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        logger.info("Starting file upload process")
        file_content = await file.read()

        # Validate file type (optional but recommended)
        if not file.content_type.startswith("application/pdf"):
            return JSONResponse(content={"message": "Invalid file type. Please upload a PDF file"}, status_code=400)

        # Create a temporary file for PDF processing (more robust)
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.seek(0)

            # Call load_pdf_file with the temporary file path
            documents = load_pdf_file(temp_file.name)

        logger.info("Creating vector store")
        create_vector_store(documents)
        logger.info("File uploaded and processed successfully")
        return RedirectResponse(url="/upload_success", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return JSONResponse(content={"message": "File upload failed!", "error": str(e)}, status_code=500)


@app.get("/upload_success", response_class=HTMLResponse)
async def read_upload_success(request: Request):
    return templates.TemplateResponse("upload_success.html", {"request": request})


@app.get("/chat_with_context", response_class=HTMLResponse)
async def read_chat_with_context(request: Request):
    return templates.TemplateResponse("chat_with_context.html", {"request": request, "response": "", "history": ""})


@app.post("/chat_with_context")
async def chat_with_context(request: Request, query: str = Form(...)):
    global vector_store, ensemble_retriever, compression_retriever

    if vector_store is None:
        vector_store = load_vector_store()
    if ensemble_retriever is None:
        ensemble_retriever = create_ensemble_retriever(vector_store)
    if compression_retriever is None:
        compression_retriever = create_compression_retriever(ensemble_retriever)

    compressed_docs = compression_retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in compressed_docs])

    response_text, history_text = chat_with_history_context(query, context)
    save_history_to_file(history_text)
    return templates.TemplateResponse("chat_with_context.html", {"request": request, "response": response_text, "history": history_text})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
