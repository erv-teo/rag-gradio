import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import bs4
import json
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
# import langgraph.prebuilt.ToolNode 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
import uuid
import torch
import gradio as gr

torch.set_num_threads(1)
load_dotenv()

# llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def isStoreEmpty(vector_store):
    return vector_store.index.ntotal == 0

def addPDF(pdfs):
    if (isinstance(pdfs, str)):
        pdfs = [pdfs]
    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        _ = vector_store.add_documents(documents=all_splits)
    return pdfs

def addURL(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=strainer)
        )
    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.aadd_documents(documents=all_splits)
    return f'Loaded {url}'


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
            )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    trimmer = trim_messages(
        strategy="last",
        token_counter=len,
        max_tokens=12,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
            )
    chain = trimmer|llm_with_tools
    response = chain.invoke(state["messages"])
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
            # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    
    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use only the provided pieces of retrieved context to form your response."
            "If you don't know the answer, say that you don't know the information."
            "If the answer cannot be found in the context, say that the information is not available, or provide a general answer."
            "\n\n"
            f"{docs_content}"
            )


    conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human","system")
            # if message.type == "human"
            or (message.type == "ai" and not message.tool_calls)
            ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages
    
    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

memory = MemorySaver()
graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile(checkpointer=memory)
thread_id = str(uuid.uuid4())
print(thread_id)
config = {"configurable": {"thread_id": thread_id}}


# pdfs = './bacteria.pdf'
# url = ''
# addPDF(pdfs)
# addURL(url)
# if vector_store.index.ntotal == 0:
#     print("Vector store is empty")
# else:
#     print("Documents Loaded")

print("RAG ready")
# while True:
#     input_message = input() 
#     for step in graph.stream(
#             {"messages": [{"role": "user", "content": input_message}]},
#             stream_mode="values",
#             config=config
#             ):
#         if (step["messages"][-1].type == "ai"):
#             step["messages"][-1].pretty_print()


# def initialize_instance(request: gr.Request):
#     instances[request.session_hash] = #TODO
#     return "Session initialized!"

# def cleanup_instance(request: gr.Request):
#     if request.session_hash in instances:
#         del instances[request.session_hash]

# def increment_counter(request: gr.Request):
#     if request.session_hash in instances:
#         instance = instances[request.session_hash]
#         return instance.increment()
#     return "Error: Session not initialized"

def refresh():
    global config
    # vector_store.index.reset()
    global index
    global vector_store
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    return None 



with gr.Blocks() as demo:

    gr.Markdown(
    """
    # Custom GenAI Demo 
    Start by uploading your PDFs! 
    """)

    file_input = gr.Files()
    gr.Markdown("Or input your URL below!")
    url_input = gr.Textbox(
        submit_btn=True,
        placeholder="Input URL",
        show_label=False,
    )
    
    gr.Markdown("Start asking questions here!")
    chatbot = gr.Chatbot(type="messages")
    
    msg = gr.Textbox(
        submit_btn=True,
        placeholder="Ask something",
        show_label=False,
        
    )
    clear = gr.Button("Clear")
    
    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history: list):
        
        question = history[-1]['content']
        bot_message = ""
        for step in graph.stream( 
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            config=config
        ):
            if step["messages"][-1].type == "ai":
                bot_message = step["messages"][-1].text()

                history.append({"role": "assistant", "content": bot_message})
        yield history

    file_input.upload(addPDF, file_input, file_input)
    url_input.submit(addURL, url_input, url_input)
    user_msg = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    bot_msg = user_msg.then(bot, chatbot, chatbot)
    
    clear.click(lambda: None, None, chatbot, queue=False)

    # demo.load(initialize_instance, inputs=None, outputs=output) 
    demo.load(refresh)
    demo.unload(refresh) 
    # demo.close(refresh)   

demo.css = """
    .gradio-container {
        width: 1140px;
        margin: 0 auto;
    }
"""
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
