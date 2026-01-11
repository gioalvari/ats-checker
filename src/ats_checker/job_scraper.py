"""Job description scraper for LinkedIn and other job sites."""

import logging
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Headers to mimic a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def is_valid_url(url: str) -> bool:
    """Check if string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_job_site(url: str) -> str | None:
    """Identify the job site from URL."""
    domain = urlparse(url).netloc.lower()

    if "linkedin.com" in domain:
        return "linkedin"
    elif "indeed.com" in domain:
        return "indeed"
    elif "glassdoor.com" in domain:
        return "glassdoor"
    elif "monster.com" in domain:
        return "monster"
    else:
        return "generic"


def scrape_linkedin_job(url: str) -> str:
    """
    Scrape job description from LinkedIn.

    LinkedIn public job pages have the description in a specific div.
    """
    try:
        # Clean the URL - remove tracking parameters
        base_url = url.split("?")[0]

        response = requests.get(base_url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Try multiple selectors for LinkedIn job descriptions
        selectors = [
            # Public job page
            {"class": "show-more-less-html__markup"},
            {"class": "description__text"},
            {"class": "jobs-description__content"},
            {"class": "jobs-box__html-content"},
            # Alternative selectors
            {"data-job-description": True},
            {"class": "job-details"},
        ]

        description_text = ""

        for selector in selectors:
            elements = soup.find_all("div", selector)
            if elements:
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True)
                    if len(text) > len(description_text):
                        description_text = text

        # If no specific div found, try to extract from script tags (JSON-LD)
        if not description_text:
            scripts = soup.find_all("script", type="application/ld+json")
            for script in scripts:
                try:
                    import json

                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        desc = data.get("description", "")
                        if desc:
                            # Clean HTML from description
                            desc_soup = BeautifulSoup(desc, "html.parser")
                            description_text = desc_soup.get_text(separator="\n", strip=True)
                            break
                except Exception:
                    continue

        # Also try to get job title and company
        job_title = ""
        company = ""

        # Title
        title_elem = soup.find("h1", class_=re.compile(r"job.*title|title", re.I))
        if title_elem:
            job_title = title_elem.get_text(strip=True)
        else:
            title_elem = soup.find("h1")
            if title_elem:
                job_title = title_elem.get_text(strip=True)

        # Company
        company_elem = soup.find("a", class_=re.compile(r"company", re.I))
        if company_elem:
            company = company_elem.get_text(strip=True)

        # Build final text
        parts = []
        if job_title:
            parts.append(f"Job Title: {job_title}")
        if company:
            parts.append(f"Company: {company}")
        if parts:
            parts.append("")  # Empty line
        if description_text:
            parts.append(description_text)

        final_text = "\n".join(parts)

        if not final_text.strip():
            raise ValueError(
                "Could not extract job description. LinkedIn may require login "
                "for this job posting. Please copy the description manually."
            )

        return final_text

    except requests.RequestException as e:
        logger.error(f"Request error scraping LinkedIn: {e}")
        raise ValueError(f"Failed to fetch LinkedIn page: {e}")
    except Exception as e:
        logger.error(f"Error scraping LinkedIn: {e}")
        raise ValueError(f"Failed to parse LinkedIn job: {e}")


def scrape_generic_job(url: str) -> str:
    """
    Generic job description scraper for any website.

    Tries to extract main content from the page.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()

        # Try common job description containers
        content_selectors = [
            {"class": re.compile(r"job.*description|description.*job", re.I)},
            {"class": re.compile(r"job.*content|content.*job", re.I)},
            {"class": re.compile(r"job.*details|details.*job", re.I)},
            {"id": re.compile(r"job.*description|description", re.I)},
            {"role": "main"},
            {"class": "content"},
            {"class": "main"},
        ]

        for selector in content_selectors:
            elements = soup.find_all(["div", "article", "section", "main"], selector)
            if elements:
                text = elements[0].get_text(separator="\n", strip=True)
                if len(text) > 200:  # Reasonable minimum length
                    return _clean_text(text)

        # Fallback: get body text
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
            return _clean_text(text)

        raise ValueError("Could not extract content from page")

    except requests.RequestException as e:
        logger.error(f"Request error: {e}")
        raise ValueError(f"Failed to fetch page: {e}")
    except Exception as e:
        logger.error(f"Error scraping page: {e}")
        raise ValueError(f"Failed to parse page: {e}")


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive whitespace
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    # Remove duplicate consecutive lines
    cleaned_lines = []
    prev_line = None
    for line in lines:
        if line != prev_line:
            cleaned_lines.append(line)
        prev_line = line

    return "\n".join(cleaned_lines)


def scrape_job_description(url: str) -> str:
    """
    Main entry point - scrape job description from URL.

    Args:
        url: URL of job posting

    Returns:
        Extracted job description text

    Raises:
        ValueError: If scraping fails
    """
    if not is_valid_url(url):
        raise ValueError("Invalid URL format")

    site = get_job_site(url)

    if site == "linkedin":
        return scrape_linkedin_job(url)
    else:
        return scrape_generic_job(url)
