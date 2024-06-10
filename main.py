import os
import sys
from taipy.gui import Gui, State, notify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

client = None
context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today? "
conversation = {
    "Conversation": [["Hi! How can I help you today?"]]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
chat_history = []
conversation_chain = []

def on_init(state: State) -> None:
    """
    Initialize the app.
    """
    print("Initializing state...")
    state.context = context
    state.conversation = conversation
    state.current_user_message = ""
    state.past_conversations = []
    state.selected_conv = None
    state.selected_row = [1]
    company = "Lec1"
    vectorstore_path = f"./annual_reports/Transcripts/{company}/faiss_index"
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(vectorstore_path, allow_dangerous_deserialization=True, embeddings=embeddings)
    state.conversation_chain = get_conversation_chain(vectorstore)
    print("State initialized")

def get_conversation_chain(vectorstore):
    """
    Create the conversation chain.
    """
    print("Creating conversation chain...")
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    print("Conversation chain created.")
    return conversation_chain

def request(state: State, prompt: str) -> str:
    """
    Send a prompt to the conversation chain and return the response.
    """
    try:
        response = state.conversation_chain({'question': prompt})
        state.chat_history = response['chat_history']
        print("Response received:", response)
        # Fetch the last answer from the conversation chain
        return state.chat_history[-1].content if state.chat_history else "No response"
    except Exception as e:
        print("Error in request function:", e)
        return "Error"

def update_context(state: State) -> None:
    """
    Update the context with the user's message and the AI's response.
    """
    # Add user's message to the context
    state.context += f"Human: {state.current_user_message}\n\nAI:"
    # Fetch the AI's response
    answer = request(state, state.context).replace("\n", "")
    # Add AI's response to the context
    state.context += answer
    # Update selected row for UI
    state.selected_row = [len(state.conversation["Conversation"]) + 1]
    return answer

def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the context.
    """
    notify(state, "info", "Sending message...")
    answer = update_context(state)
    # Append the user's message and the AI's response to the conversation
    conv = state.conversation.copy()  # Use a shallow copy if conversation is a dict, or just direct assignment if it's a list
    conv["Conversation"].append(state.current_user_message)  # User message
    conv["Conversation"].append(answer)  # AI response
    # Reset the current user message
    state.current_user_message = ""
    state.conversation = conv
    notify(state, "success", "Response received!")

def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation messages based on the index.
    """
    if idx % 2 == 0:
        return "user_message"  # Style for user messages
    else:
        return "gpt_message"  # Style for AI messages

def on_exception(state, function_name: str, ex: Exception) -> None:
    """
    Catches exceptions and notifies user in Taipy GUI.
    """
    notify(state, "error", f"An error occurred in {function_name}: {ex}")

def reset_chat(state: State) -> None:
    """
    Reset the chat by clearing the conversation.
    """
    state.past_conversations.append([len(state.past_conversations), state.conversation])
    state.conversation = {"Conversation": [["Who are you?", "Hi! I am GPT-4. How can I help you today?"]]}
    state.chat_history = []
    print("Chat reset. Past conversations:", state.past_conversations)

def tree_adapter(item: list) -> [str, str]:
    """
    Converts element of past_conversations to id and displayed string.
    """
    identifier = item[0]
    if len(item[1]["Conversation"]) > 1:
        return (identifier, item[1]["Conversation"][1][0][:50] + "...")
    return (item[0], "Empty conversation")

def select_conv(state: State, var_name: str, value) -> None:
    """
    Selects conversation from past_conversations.
    """
    state.conversation = state.past_conversations[value[0][0]][1]
    state.selected_row = [len(state.conversation["Conversation"])]
    print("Conversation selected:", state.conversation)

page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# **Lecture Chat**{: .gradient-text} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>
### Chat History ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
"""

if __name__ == "__main__":
    if "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
    elif len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        raise ValueError(
            "Please provide the OpenAI API key as an environment variable OPENAI_API_KEY or as a command line argument."
        )
    
    stylekit = {
    "color_primary": "#7476C3",
    "color_secondary": "#7476C3",
    "color_background_dark": "#131314",
    "color_paper_dark": "#1C1C1D"
    }
    Gui(page).run(debug=True, stylekit=stylekit, dark_mode=True, use_reloader=True, title="Lecture Chat", port=5001)
