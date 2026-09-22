"""
Defines the single CrewAI agent that researches and finds job openings
that match a candidate's LinkedIn profile, using free web search + web
scraping tools, powered by Groq's `openai/gpt-oss-120b` model.

Everything (the custom search tool + the agent/task/crew) lives in this one
file on purpose, so there is no `tools/` or `utils/` package folder to get
wrong when setting up the GitHub repo.
"""

from typing import Type

import requests
from bs4 import BeautifulSoup
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field

# Maximum characters of profile text / scraped page text we send to the LLM.
# Groq's free tier has a daily token budget, and raw scraped pages / long
# profiles can burn through it fast, so we keep everything on a tight leash.
MAX_PROFILE_CHARS = 4000
MAX_SCRAPE_CHARS = 2000

# --------------------------------------------------------------------------
# Free, no-API-key web search tool (DuckDuckGo via the `ddgs` package).
# We use this instead of CrewAI's built-in SerperDevTool because that one
# requires a paid/limited Serper.dev API key.
# --------------------------------------------------------------------------


class DuckDuckGoSearchInput(BaseModel):
    """Input schema for DuckDuckGoSearchTool."""

    query: str = Field(
        ...,
        description=(
            "The search query to run, e.g. "
            "'remote Python developer jobs 2026' or "
            "'site:indeed.com data analyst jobs London'."
        ),
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Searches the public web using DuckDuckGo (completely free, no API key). "
        "Use this to find job postings, job board listings (LinkedIn, Indeed, "
        "Glassdoor, etc.), company career pages, and news articles about hiring. "
        "Pass a clear, specific search query as input."
    )
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region="wt-wt",
                        safesearch="off",
                        max_results=8,
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive network handling
            return f"Search failed for query '{query}'. Error: {exc}"

        if not results:
            return f"No results found for query: '{query}'. Try a different phrasing."

        formatted_results = []
        for i, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            link = result.get("href") or result.get("link", "No link")
            snippet = result.get("body") or result.get("snippet", "")
            formatted_results.append(
                f"{i}. {title}\n   Link: {link}\n   Snippet: {snippet}"
            )

        return "\n\n".join(formatted_results)


# --------------------------------------------------------------------------
# Free, no-API-key page reading tool with a hard length cap (to save tokens).
# We use a small custom tool instead of crewai_tools' ScrapeWebsiteTool
# because that one returns the entire page text with no size limit, which
# can burn through a lot of Groq's daily free-tier token budget.
# --------------------------------------------------------------------------


class ScrapePageInput(BaseModel):
    """Input schema for LightScrapeTool."""

    url: str = Field(..., description="The full URL of the web page to read.")


class LightScrapeTool(BaseTool):
    name: str = "Read Web Page"
    description: str = (
        "Fetches a web page and returns its main visible text (capped in "
        "length to keep things efficient). Use this to confirm a job "
        "posting is real and to read its details."
    )
    args_schema: Type[BaseModel] = ScrapePageInput

    def _run(self, url: str) -> str:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (JobFindingAIAgent/1.0)"},
                timeout=10,
            )
            response.raise_for_status()
        except Exception as exc:
            return f"Could not fetch '{url}'. Error: {exc}"

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())

        if not text:
            return f"No readable text found on '{url}'."

        truncated = text[:MAX_SCRAPE_CHARS]
        if len(text) > MAX_SCRAPE_CHARS:
            truncated += " ... [truncated]"

        return truncated


# --------------------------------------------------------------------------
# LLM setup — Groq's OpenAI-compatible endpoint.
# We route through the "openai/" prefix (CrewAI's native OpenAI-compatible
# code path) instead of "groq/" because, as of CrewAI 1.x, the LiteLLM-based
# "groq/" route has a known bug with some Groq model names. Using base_url +
# the "openai/" prefix avoids it completely while still calling Groq.
# --------------------------------------------------------------------------

GROQ_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_MODEL_STRING = f"openai/{GROQ_MODEL_NAME}"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_groq_llm(groq_api_key: str) -> LLM:
    """Build a CrewAI LLM object pointed at Groq's free-tier API."""
    return LLM(
        model=GROQ_MODEL_STRING,
        api_key=groq_api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.4,
    )


# --------------------------------------------------------------------------
# The single agent, its task, and the crew that runs it.
# --------------------------------------------------------------------------


def build_job_search_crew(
    profile_text: str,
    work_mode: str,
    location: str | None,
    groq_api_key: str,
) -> Crew:
    """
    Build a single-agent CrewAI crew that searches the web for jobs matching
    the given LinkedIn profile and preferences.

    Args:
        profile_text: Text extracted from the candidate's LinkedIn PDF.
        work_mode: "Remote" or "Onsite".
        location: Location string when work_mode == "Onsite" (city/country,
            "England", or "United States"). Ignored for Remote.
        groq_api_key: The user's free Groq API key.

    Returns:
        A ready-to-run CrewAI Crew (call `.kickoff()` on it).
    """
    llm = get_groq_llm(groq_api_key)

    search_tool = DuckDuckGoSearchTool()
    scrape_tool = LightScrapeTool()

    job_search_agent = Agent(
        role="Senior Job Search Specialist",
        goal=(
            "Find real, currently open job postings that closely match the "
            "candidate's skills, experience, and stated work preferences, by "
            "actively searching the live web and reading job pages/articles."
        ),
        backstory=(
            "You are an experienced technical recruiter and career coach. "
            "You are excellent at reading a resume or LinkedIn profile, "
            "identifying a candidate's strongest skills and seniority level, "
            "and then finding genuine matching job openings across job "
            "boards (LinkedIn Jobs, Indeed, Glassdoor, company career pages), "
            "tech blogs, and news articles. You never invent companies, job "
            "titles, or links — you only report what you actually found "
            "through search and page reading."
        ),
        tools=[search_tool, scrape_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        # Memory is OFF on purpose: CrewAI's default memory=True tries to
        # create an embedder (defaults to OpenAI embeddings), which will
        # hang/retry for a long time if you only have a GROQ_API_KEY set
        # and no OPENAI_API_KEY. We don't need memory for a single-shot
        # job search anyway.
        memory=False,
        # Hard safety caps so a confused agent can never spin forever:
        # stop after at most 8 tool-call rounds, or 3 minutes, whichever
        # comes first.
        max_iter=6,
        max_execution_time=180,
    )

    # Keep the profile short — LinkedIn PDF exports repeat a lot of
    # boilerplate, and every extra character here gets re-sent to the LLM
    # on every tool-call round, which adds up fast against Groq's daily
    # token budget.
    trimmed_profile = profile_text[:MAX_PROFILE_CHARS]
    if len(profile_text) > MAX_PROFILE_CHARS:
        trimmed_profile += "\n... [profile truncated for length]"

    if work_mode == "Remote":
        preference_text = "The candidate wants REMOTE jobs only (work from anywhere)."
    else:
        preference_text = (
            f"The candidate wants ONSITE / hybrid jobs located in or near: {location}."
        )

    task_description = f"""
You are helping a job seeker find real, currently available job openings.

Here is the candidate's LinkedIn profile, extracted from their PDF export:
---
{trimmed_profile}
---

Preference: {preference_text}

Follow these steps:
1. Carefully read the profile above and identify: the candidate's top hard
   skills, most recent job titles, approximate seniority level (junior /
   mid / senior), and industry or domain.
2. Run MULTIPLE different web searches (vary the phrasing) to find current
   job postings that match those skills and the stated work-mode/location
   preference. Try things like:
   - "<top skill> <job title> jobs" plus the work mode / location
   - "site:linkedin.com/jobs <job title>"
   - "site:indeed.com <job title> <location or remote>"
   - "site:glassdoor.com <job title>"
   - relevant company career pages or hiring news articles
3. Use the web scraping tool on the most promising result links to confirm
   each posting is real and to capture its exact job title, company name,
   location, and a short description.
4. From everything you found, select the 8 to 12 BEST matching jobs.

Formatting rules for your final answer (Markdown):
- One entry per job, using this pattern:
  ### <Job Title> — <Company Name>
  - **Location:** <Remote / City, Country>
  - **About the role:** 1-2 sentence summary
  - **Why it fits you:** short reason tied to the candidate's profile
  - **Apply / view here:** <direct URL>
- Only include jobs for which you have a real URL you found through search
  or scraping. NEVER invent a company, job title, or link.
- If you could not find enough strong matches, say so honestly and list
  fewer jobs rather than making some up.
"""

    job_search_task = Task(
        description=task_description,
        expected_output=(
            "A Markdown-formatted list of 8-12 real, verifiable job postings "
            "matching the candidate's profile and preferences, each with a "
            "title, company, location, short summary, match reason, and a "
            "working link."
        ),
        agent=job_search_agent,
    )

    crew = Crew(
        agents=[job_search_agent],
        tasks=[job_search_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
