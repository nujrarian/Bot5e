import streamlit as st
from agents import ChatbotAgent, PDFQAAgent
from classifier import classify_query
from config import config
from logger import setup_logger

logger = setup_logger(__name__)

# Page config
st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon
)

# Initialize agents
@st.cache_resource
def load_agents():
    """Cache agents to avoid reloading on every interaction"""
    try:
        logger.info("Initializing agents...")
        general_agent = ChatbotAgent()
        pdf_agent = PDFQAAgent()
        logger.info("Agents initialized successfully")
        return general_agent, pdf_agent
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        st.error(f"Failed to initialize chatbot: {e}")
        st.stop()

general_agent, pdf_agent = load_agents()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    show_agent = st.checkbox("Show active agent", value=config.show_agent_by_default)
    show_context = st.checkbox("Show retrieved context (debug)", value=config.show_context_by_default)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "system", "content": "You are a friendly and helpful Dungeons and Dragons chatbot designed to answer questions about Dungeons and Dragons 5th Edition. Always respond in a friendly and helpful manner to user queries."},
            {"role": "assistant", "content": "How may I assist you today?"}
        ]
        st.rerun()
    
    st.divider()
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    **General:**
    - Tell me about the history of D&D
    - What's a good class for beginners?
    
    **Rules:**
    - How does divine smite work?
    - What are the conditions in 5e?
    - Explain advantage and disadvantage
    """)

# Title
st.title("Bot5e - D&D 5e Assistant")
st.caption("Your friendly D&D 5th Edition rules companion")

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a friendly and helpful Dungeons and Dragons chatbot designed to answer questions about Dungeons and Dragons 5th Edition. Always respond in a friendly and helpful manner to user queries."},
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Show agent badge if it exists
            if "agent" in message and show_agent:
                if message["agent"] == "general":
                    st.caption("💬 General Chat")
                else:
                    st.caption("📖 Rules Lookup")

def get_full_history(messages):
    history = ""
    for message in messages:
        if message["role"] != "system":  # Exclude the system message from the displayed history
            role = "User" if message["role"] == "user" else "Assistant"
            history += f"{role}: {message['content']}\n"
        else:
            history += f"System: {message['content']}\n"  # Include the system message in the context
    return history

if question := st.chat_input("Ask about D&D 5e rules or chat about the game..."):
    # Input validation
    if not question or not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Sanitize input (remove excessive whitespace)
    question = question.strip()

    # Limit question length
    max_question_length = 1000
    if len(question) > max_question_length:
        st.warning(f"Question is too long. Please limit to {max_question_length} characters.")
        st.stop()

    # Append user input to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Create full history for context
    full_history = get_full_history(st.session_state.messages)

    # Show "thinking" indicator
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # Classify the query
                agent_name = classify_query(question)

                # Route to appropriate agent
                if agent_name == "general":
                    if show_agent:
                        st.caption("💬 Using General Chat")
                    response_content = general_agent.handle_query(question, full_history)
                else:
                    if show_agent:
                        st.caption("📖 Searching D&D Rules...")
                    response_content = pdf_agent.handle_query(question, full_history)

                # Display response
                st.markdown(response_content)

                # Show agent badge
                if show_agent:
                    if agent_name == "general":
                        st.caption("💬 General Chat")
                    else:
                        st.caption("📖 Rules Lookup")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            st.error("I encountered an error processing your request. Please try again or rephrase your question.")
            response_content = "I apologize, but I encountered an error. Please try again."
            agent_name = "error"

    # Save to history with agent info
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content,
        "agent": agent_name
    })
    st.rerun()