import gradio as gr
import os
api_token = os.getenv("HF_TOKEN")

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
import torch

list_llm = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]  
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Load and split PDF document
def load_doc(list_file_path):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, 
        chunk_overlap=64
    )  
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Create vector database
def create_db(splits):
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb

# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        huggingfacehub_api_token=api_token,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )

    retriever = vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

# Initialize database
def initialize_database(list_file_obj, progress=gr.Progress()):
    list_file_path = [x.name for x in list_file_obj if x is not None]
    doc_splits = load_doc(list_file_path)
    vector_db = create_db(doc_splits)
    return vector_db, "✅ Database created!"

# Initialize LLM
def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    llm_name = list_llm[llm_option]
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "✅ QA chain initialized. Chatbot is ready!"

def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    response = qa_chain.invoke({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if "Helpful Answer:" in response_answer:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    new_history = history + [(message, response_answer)]
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page

# Enhanced user interface with better layout and styling
def app():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green", neutral_hue="slate")) as interface:
        vector_db = gr.State()
        qa_chain = gr.State()
        
        gr.HTML("<center><h1> App Genie</h1></center>")
        gr.Markdown("""
        <center><b>Query your PDF documents using advanced AI technology!</b></center>
        <br><center><i>This tool uses Retrieval Augmented Generation (RAG) to extract meaningful insights from your documents. <br>
        Please ensure the PDFs uploaded do not contain sensitive information.</i></center>
        <hr>
        """)

        with gr.Row():
            with gr.Column(scale=60):
                gr.Markdown("### Step 1: Upload Your PDF Documents")
                document = gr.Files(file_count="multiple", file_types=["pdf"], label="📂 Upload PDF Files", interactive=True)
                db_btn = gr.Button("⚙️ Create Vector Database", variant="primary")
                db_progress = gr.Textbox(value="Status: Not initialized", show_label=False)

                gr.Markdown("### Step 2: Choose Your Language Model & Parameters")
                llm_btn = gr.Radio(list_llm_simple, label="🧠 Choose a Language Model", value=list_llm_simple[0], type="index")
                
                with gr.Accordion("LLM Advanced Settings", open=False):
                    slider_temperature = gr.Slider(minimum=0.01, maximum=1.0, value=0.5, step=0.1, label="🔧 Temperature")
                    slider_maxtokens = gr.Slider(minimum=128, maximum=9192, value=4096, step=128, label="🔧 Max Tokens")
                    slider_topk = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="🔧 Top-k")
                
                qachain_btn = gr.Button("🚀 Initialize Chatbot", variant="primary")
                llm_progress = gr.Textbox(value="Status: Not initialized", show_label=False)

            with gr.Column(scale=40):
                gr.Markdown("### Step 3: Chat with Your Documents")
                chatbot = gr.Chatbot(height=400, label="💬 Chat Window", show_label=False)
                
                with gr.Accordion("🔍 Relevant Document References", open=False):
                    doc_source1 = gr.Textbox(label="🔗 Reference 1", lines=2)
                    source1_page = gr.Number(label="📄 Page 1", interactive=False)
                    doc_source2 = gr.Textbox(label="🔗 Reference 2", lines=2)
                    source2_page = gr.Number(label="📄 Page 2", interactive=False)
                    doc_source3 = gr.Textbox(label="🔗 Reference 3", lines=2)
                    source3_page = gr.Number(label="📄 Page 3", interactive=False)
                
                msg = gr.Textbox(placeholder="💬 Ask a question about the documents", container=True, show_label=False)
                submit_btn = gr.Button("Submit", variant="secondary")
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear Chat")

        # Event bindings
        db_btn.click(initialize_database, inputs=[document], outputs=[vector_db, db_progress])
        qachain_btn.click(initialize_LLM, inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], outputs=[qa_chain, llm_progress])
        msg.submit(conversation, inputs=[qa_chain, msg, chatbot], outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page])
        submit_btn.click(conversation, inputs=[qa_chain, msg, chatbot], outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page])
        clear_btn.click(lambda: [None, "", 0, "", 0, "", 0], outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page])

    interface.queue().launch(debug=True)

if __name__ == "__main__":
    app()