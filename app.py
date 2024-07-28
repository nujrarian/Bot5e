import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from agents import ChatbotAgent, PDFQAAgent
from classifier import classify_query
from langchain.agents import AgentExecutor, Tool

# Initialize agents
general_agent = ChatbotAgent()
pdf_agent = PDFQAAgent(pdf_path='SRD-OGL_V5.1.pdf')

# Define LangChain tools for the agents
def general_agent_tool(question, history):
    return general_agent.handle_query(question, history)

def pdf_agent_tool(question, history):
    return pdf_agent.handle_query(question, history)

general_tool = Tool(
    name="General Chatbot",
    func=general_agent_tool,
    description="Handles general conversation queries.",
)

pdf_tool = Tool(
    name="PDF Q&A",
    func=pdf_agent_tool,
    description="Handles queries related to the PDF content.",
)

tools = [general_tool, pdf_tool]

# Set up Streamlit interface
st.title("DnD 5e Bot")

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a friendly and helpful Dungeons and Dragons chatbot designed to answer questions about Dungeons and Dragons 5th Edition. Always respond in a friendly and helpful manner to user queries."},
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":  # Skip displaying the system message
        with st.chat_message(message["role"]):
            st.write(message["content"])

def get_full_history(messages):
    history = ""
    for message in messages:
        if message["role"] != "system":  # Exclude the system message from the displayed history
            role = "User" if message["role"] == "user" else "Assistant"
            history += f"{role}: {message['content']}\n"
        else:
            history += f"System: {message['content']}\n"  # Include the system message in the context
    return history

if question := st.chat_input("Ask 5eBot"):
    # Append user input to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Create full history for context
    full_history = get_full_history(st.session_state.messages)

    # Classify the query
    agent_name = classify_query(question)
    if agent_name == "general":
        response_content = general_agent_tool(question, full_history)
    else:
        response_content = pdf_agent_tool(question, full_history)

    st.session_state.messages.append({"role": "assistant", "content": response_content})
    # Display assistant response immediately
    with st.chat_message("assistant"):
        st.markdown(response_content)
