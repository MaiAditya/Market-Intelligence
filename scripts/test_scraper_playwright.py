import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingestion.web_scraper import WebScraper
import logging

logging.basicConfig(level=logging.INFO)

def test_scraper():
    scraper = WebScraper()
    url = "https://medium.com/about"
    
    print(f"Scraping {url} (Standard)...")
    # This should use requests (status 200) or newspaper
    result = scraper.scrape(url, use_newspaper=False)
    if result:
        print(f"Success: {result.title}")
    else:
        print("Failed")

    print("\nTesting Playwright directly...")
    # This calls _scrape_with_playwright
    result_pw = scraper._scrape_with_playwright(url)
    if result_pw:
        print(f"Success (Playwright): {result_pw.title}")
    else:
        print("Failed (Playwright)")

if __name__ == "__main__":
    test_scraper()
