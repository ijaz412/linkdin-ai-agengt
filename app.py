"""
Job Finding AI Agent — Streamlit front-end.

Upload a LinkedIn "Save to PDF" profile export, tell the agent whether you
want remote or onsite work (and where), and a single CrewAI agent (powered
by Groq's free `openai/gpt-oss-120b` model) searches the live web for real,
matching job openings.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from agent import GROQ_MODEL_NAME, build_job_search_crew
from pdf_parser import extract_text_from_pdf

load_dotenv()  # allows a local .env file with GROQ_API_KEY for local runs

st.set_page_config(page_title="Job Finding AI Agent", page_icon="🧑‍💼", layout="centered")

st.title("🧑‍💼 Job Finding AI Agent")
st.caption(
    "Upload your LinkedIn profile PDF, set your job preferences, and let a "
    "single CrewAI agent search the live web for real matching jobs."
)

# --------------------------------------------------------------------------
# Sidebar — Groq API key
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    default_key = os.getenv("GROQ_API_KEY", "")
    try:
        # st.secrets raises if no secrets.toml exists locally — that's fine.
        default_key = st.secrets.get("GROQ_API_KEY", default_key)
    except Exception:
        pass

    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Paste your free Groq API key. It is only used for this session "
        "and is never stored.",
    )
    st.markdown("🔑 [Get a free Groq API key](https://console.groq.com/keys)")
    st.divider()
    st.markdown(f"**Model:** `{GROQ_MODEL_NAME}` (via Groq, free tier)")
    st.markdown("**Search:** DuckDuckGo (free, no key) + free web scraping")

# --------------------------------------------------------------------------
# Step 1 — Upload LinkedIn PDF
# --------------------------------------------------------------------------
st.subheader("1. Upload your LinkedIn profile PDF")
st.caption(
    "On your LinkedIn profile page: click **More → Save to PDF**, then upload "
    "that file here."
)

uploaded_file = st.file_uploader("LinkedIn profile PDF", type=["pdf"])

if "profile_text" not in st.session_state:
    st.session_state.profile_text = None

if uploaded_file is not None:
    with st.spinner("Reading your profile..."):
        try:
            st.session_state.profile_text = extract_text_from_pdf(uploaded_file)
            st.success("Profile loaded ✅")
        except Exception as exc:
            st.session_state.profile_text = None
            st.error(f"Could not read this PDF: {exc}")

    if st.session_state.profile_text:
        with st.expander("Preview extracted profile text"):
            preview = st.session_state.profile_text[:3000]
            suffix = "..." if len(st.session_state.profile_text) > 3000 else ""
            st.text(preview + suffix)

# --------------------------------------------------------------------------
# Step 2 — Job preferences
# --------------------------------------------------------------------------
st.subheader("2. Job preferences")

work_mode = st.radio(
    "Are you looking for a Remote or Onsite job?",
    options=["Remote", "Onsite"],
    horizontal=True,
)

location = None
if work_mode == "Onsite":
    location_choice = st.radio(
        "Where should the agent search?",
        options=["My own location", "Anywhere in England", "Anywhere in the United States"],
    )
    if location_choice == "My own location":
        location = st.text_input(
            "Enter your city / country", placeholder="e.g. Lahore, Pakistan"
        )
    else:
        location = location_choice.replace("Anywhere in ", "")

# --------------------------------------------------------------------------
# Step 3 — Run the agent
# --------------------------------------------------------------------------
st.subheader("3. Find matching jobs")

run_clicked = st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True)

if run_clicked:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
    elif not st.session_state.profile_text:
        st.error("Please upload your LinkedIn profile PDF first.")
    elif work_mode == "Onsite" and not location:
        st.error("Please provide a location for onsite jobs.")
    else:
        with st.spinner(
            "Your AI agent is searching the web for matching jobs. "
            "This can take one to two minutes..."
        ):
            try:
                crew = build_job_search_crew(
                    profile_text=st.session_state.profile_text,
                    work_mode=work_mode,
                    location=location,
                    groq_api_key=groq_api_key,
                )
                result = crew.kickoff()

                st.success("Done! Here are your matching jobs:")
                st.markdown(str(result))
            except Exception as exc:
                st.error(f"Something went wrong while running the agent: {exc}")

st.divider()
st.caption(
    "Built with CrewAI, Streamlit and Groq. All tools used (DuckDuckGo search "
    "and web scraping) are free and require no paid API keys."
)
