import logging
import os

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig()


def scrape_dtd_data(
    start_year,
    end_year,
    path="",
):
    if not os.path.exists(path):
        os.mkdir(path)

    for year in range(start_year, end_year):
        logger.info(f"Scraping DTD data for {year}")
        prosports = []
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        while True:
            url = f"https://www.prosportstransactions.com/baseball/Search/SearchResults.php?Player=&Team=&BeginDate={start}&EndDate={end}&DLChkBx=yes&prosportsChkBx=yes&submit=Search&start={len(prosports)*25}"
            r = requests.get(url)
            scraped_df = pd.read_html(r.text, header=0)[0]
            if scraped_df.empty:
                break
            prosports.append(scraped_df)

        prosports_df = pd.concat(prosports)
        prosports_df.to_csv(
            os.path.join(path, f"prosports_{year}.csv"), index=False
        )
