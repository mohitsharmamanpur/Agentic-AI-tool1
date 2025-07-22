import streamlit as st
import os
from langchain_core.tools import tool
from langchain.agents import tool as agent_tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyABUqgYzp7ekmlyKErgG5hu_-H0JIAPB1A"  # Replace this with your Gemini API key

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def refine_idea(input: str) -> str:
    """Refines a vague startup idea into a clear problem-solution statement."""
    prompt = f"Refine this vague startup idea into a specific, innovative, and clear problem-solution format:\n\nIdea: {input}"
    return llm.invoke(prompt)

@tool
def market_research(input: str) -> str:
    """Performs basic market research including market size, trends, and key competitors."""
    prompt = f"Do basic market research for this startup idea:\n\n{input}\n\nInclude market size, competitors, and trends."
    return llm.invoke(prompt)

@tool
def business_model(input: str) -> str:
    """Generates a Business Model Canvas for the startup."""
    prompt = f"Generate a Business Model Canvas for this startup:\n\n{input}"
    return llm.invoke(prompt)

@tool
def pitch_deck(input: str) -> str:
    """Creates a concise pitch deck outline with key startup sections."""
    prompt = f"Generate a concise pitch deck outline for this startup idea:\n\n{input}\n\nInclude sections like Problem, Solution, Market Size, Business Model, and Team."
    return llm.invoke(prompt)

@tool
def elevator_pitch(input: str) -> str:
    """Generates a short, persuasive 30-second elevator pitch."""
    prompt = f"Write a 30-second elevator pitch for this startup:\n\n{input}"
    return llm.invoke(prompt)


tools = [refine_idea, market_research, business_model, pitch_deck, elevator_pitch]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=False)


st.set_page_config(page_title="Startup Assistant", page_icon="🚀")
st.title(" AI-Powered Startup Assistant")
st.write("Built with LangChain + Gemini 2.5 Flash")

user_input = st.text_area("💡 Describe your startup idea", height=150, placeholder="e.g. An AI tool that helps students learn based on their emotions...")


selected_tool = st.selectbox(
    "🛠 What would you like help with?",
    (
        "Refine Startup Idea",
        "Market Research",
        "Business Model Canvas",
        "Pitch Deck Generator",
        "Elevator Pitch"
    )
)


if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter a startup idea first.")
    else:
        st.info("Generating response using Gemini 2.5 Flash...")
        try:
            if selected_tool == "Refine Startup Idea":
                result = refine_idea.run(user_input)
            elif selected_tool == "Market Research":
                result = market_research.run(user_input)
            elif selected_tool == "Business Model Canvas":
                result = business_model.run(user_input)
            elif selected_tool == "Pitch Deck Generator":
                result = pitch_deck.run(user_input)
            elif selected_tool == "Elevator Pitch":
                result = elevator_pitch.run(user_input)
            else:
                result = agent.run(user_input)

            st.success(" Output")
            st.markdown(result)

        except Exception as e:
            st.error(f" An error occurred: {e}")
