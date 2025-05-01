# Streamlit-based LangGraph app for automating insurance claim decisions
# This application processes insurance claims using a graph-based workflow
# combining multiple AI agents for risk assessment, validity checking, and decision making

import os
import re
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings

# --- Initialize OpenAI Configuration ---
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI()

# --- Setup Vector Store for Claim History ---
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="claims_history",
    embedding_function=embedding_model,
    persist_directory="./chroma_store",
    client_settings=Settings(anonymized_telemetry=False)
)

# --- Define State Structure ---
class ClaimState(TypedDict):
    claim: str
    risk_score: str
    validity: str
    decision: str
    response: str

def extract_numeric_score(text: str) -> str:
    matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    for match in matches:
        score = float(match)
        if score <= 10:
            return str(score)
    return "10"

# --- Define Processing Nodes ---
def evaluate_risk(state: ClaimState) -> ClaimState:
    claim_text = state["claim"]
    similar_docs = vectorstore.similarity_search(claim_text, k=3)
    similar_examples = "\n\n---\n\n".join(doc.page_content for doc in similar_docs)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an insurance claims analyst expert.

        Based on the following new insurance claim and previously processed similar claims,
        assign a risk score from 0 to 10 (higher = more suspicious). Respond with a single number.

        New Claim:
        {claim}

        Similar Past Claims:
        {examples}
        """
    )
    chain = prompt | llm
    response = chain.invoke({
        "claim": claim_text,
        "examples": similar_examples
    }).content.strip()

    risk_score = extract_numeric_score(response)

    print(f"Risk Score (raw): {response}")
    print(f"Risk Score (parsed): {risk_score}")

    return {"risk_score": risk_score, "response": response}

def check_validity(state: ClaimState) -> ClaimState:
    prompt = ChatPromptTemplate.from_template(
        """
        Determine if this insurance claim is complete and valid.
        Respond with 'Valid' or 'Invalid'.
        Claim: {claim}
        """
    )
    chain = prompt | llm
    validity = chain.invoke({"claim": state["claim"]}).content.strip()

    print(f"Validity: {validity}")

    return {"validity": validity}

def approve_claim(state: ClaimState) -> ClaimState:
    vectorstore.add_texts([state["claim"]])
    return {"decision": "✅ Claim Approved"}

def reject_claim(state: ClaimState) -> ClaimState:
    vectorstore.add_texts([state["claim"]])
    return {"decision": "❌ Claim Rejected"}

def send_for_review(state: ClaimState) -> ClaimState:
    vectorstore.add_texts([state["claim"]])
    return {"decision": "Sent to Claims for further review"}

def route_claim(state: ClaimState) -> str:
    try:
        score = float(state.get("risk_score", 10))
    except:
        score = 10
    validity = state.get("validity", "Invalid")

    if validity == "Invalid":
        print(f"Routing Claim: {validity}, Score: {score}")
        return "reject_claim"
    elif score < 3:
        print(f"Routing Claim: {validity}, Score: {score}")
        return "approve_claim"
    elif score > 7:
        print(f"Routing Claim: {validity}, Score: {score}")
        return "reject_claim"
    else:
        print(f"Routing Claim: {validity}, Score: {score}")
        return "send_for_review"

# --- Define Workflow Graph ---
workflow = StateGraph(ClaimState)
workflow.add_node("evaluate_risk", evaluate_risk)
workflow.add_node("check_validity", check_validity)
workflow.add_node("approve_claim", approve_claim)
workflow.add_node("reject_claim", reject_claim)
workflow.add_node("send_for_review", send_for_review)
workflow.add_edge(START, "evaluate_risk")
workflow.add_edge("evaluate_risk", "check_validity")
workflow.add_conditional_edges("check_validity", route_claim)
workflow.add_edge("approve_claim", END)
workflow.add_edge("reject_claim", END)
workflow.add_edge("send_for_review", END)

app = workflow.compile()

# --- Streamlit User Interface ---
st.title("\U0001F4C4 Insurance Claim Processor (LangGraph AI Agent)")
claim_text = st.text_area("Enter insurance claim details:", height=200)

if st.button("\U0001F9E0 Evaluate Claim") and claim_text:
    with st.spinner("Processing..."):
        result = app.invoke({"claim": claim_text})
        st.success("Decision Made")
        st.write(f"**AI Explanation:** {result.get('response', 'N/A')}")
        st.write(f"**Risk Score:** {result['risk_score']}")
        st.write(f"**Validity:** {result['validity']}")
        st.write(f"**Decision:** {result['decision']}")
