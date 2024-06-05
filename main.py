import gradio as gr
import fitz  # PyMuPDF
from transformers import pipeline

# Load the QA and summarization pipelines
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_document = fitz.open(pdf_file)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def answer_question(pdf_file, question):
    """Extract text from PDF and answer the given question."""
    text = extract_text_from_pdf(pdf_file)
    answer = qa_pipeline(question=question, context=text)
    return answer["answer"]

def summarize_pdf(pdf_file):
    """Extract text from PDF and summarize it."""
    text = extract_text_from_pdf(pdf_file)
    # BART model has a maximum token limit, so we'll summarize in chunks if needed
    max_chunk = 512  # BART's token limit
    inputs = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summaries = summarization_pipeline(inputs)
    summarized_text = " ".join([summary['summary_text'] for summary in summaries])
    return summarized_text

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# PDF Question Answering and Summarization Tool")

    with gr.Tab("Question Answering"):
        with gr.Row():
            with gr.Column():
                pdf_file = gr.File(label="Upload PDF")
                question = gr.Textbox(label="Enter your question")
                answer_btn = gr.Button("Get Answer")
            with gr.Column():
                answer_output = gr.Textbox(label="Answer", interactive=False)

        answer_btn.click(answer_question, inputs=[pdf_file, question], outputs=answer_output)

    with gr.Tab("Summarization"):
        with gr.Row():
            with gr.Column():
                pdf_file_sum = gr.File(label="Upload PDF")
                summarize_btn = gr.Button("Summarize PDF")
            with gr.Column():
                summary_output = gr.Textbox(label="Summary", interactive=False, lines=10)

        summarize_btn.click(summarize_pdf, inputs=pdf_file_sum, outputs=summary_output)

demo.launch()
