"""
A free, no-API-key web search tool for CrewAI, built on top of the `ddgs`
package (DuckDuckGo Search). We use this instead of CrewAI's built-in
SerperDevTool because SerperDevTool requires a paid/limited Serper.dev API
key, and the goal of this project is to stay 100% free.
"""

from typing import Type

from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import BaseModel, Field


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
