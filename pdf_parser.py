"""
Helper to extract plain text from a LinkedIn "Save to PDF" profile export
that the user uploads through Streamlit's file_uploader.
"""

from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract and clean text from an uploaded PDF file.

    Args:
        uploaded_file: A Streamlit UploadedFile object (file-like), e.g. from
            st.file_uploader(..., type=["pdf"]).

    Returns:
        Cleaned plain text extracted from every page of the PDF.
    """
    reader = PdfReader(uploaded_file)

    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)

    # Drop empty/blank lines and extra whitespace so the LLM gets a cleaner
    # profile summary instead of raw PDF layout noise.
    cleaned_lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    cleaned_text = "\n".join(cleaned_lines)

    if not cleaned_text:
        raise ValueError(
            "No readable text was found in this PDF. Make sure you uploaded a "
            "text-based LinkedIn 'Save to PDF' export, not a scanned image."
        )

    return cleaned_text
