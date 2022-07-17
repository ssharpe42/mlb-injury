import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig()


def scrape_team_info(year):
    teams_url = "https://statsapi.mlb.com/api/v1/teams?sportId=1&season={year}"
    mlb_teams = requests.get(teams_url.format(year=year)).json()["teams"]
    teams_yr = pd.DataFrame(
        [
            {
                k: v
                for k, v in team.items()
                if k in ["id", "teamName", "abbreviation"]
            }
            for team in mlb_teams
        ]
    ).rename(
        columns={
            "abbreviation": "team",
            "teamName": "team_name",
            "id": "team_id",
        }
    )
    teams_yr["year"] = year
    return teams_yr


def scrape_il_data(start_year, end_year):

    url = "https://statsapi.mlb.com/api/v1/transactions?startDate={start}&endDate={end}"

    all_status_changes = []
    teams_list = []
    for year in range(start_year, end_year):
        logger.info(f"Scraping IL data for {year}")

        start = f"{year}-01-01"
        end = f"{year}-12-31"
        results = requests.get(url.format(start=start, end=end))
        results = results.json()["transactions"]
        status_changes = [x for x in results if x["typeCode"] == "SC"]

        for trxn in status_changes:
            if "person" in trxn and "description" in trxn:
                temp = {
                    k: v
                    for k, v in trxn.items()
                    if k
                    in [
                        "id",
                        "date",
                        "effectiveDate",
                        "resolutionDate",
                        "description",
                    ]
                }
                temp["player_id"] = trxn["person"]["id"]
                temp["team_id"] = trxn["toTeam"]["id"]
                all_status_changes.append(temp)

        # Get teams
        teams_list.append(scrape_team_info(year))

    teams = pd.concat(teams_list)
    status_changes = pd.DataFrame(all_status_changes)
    status_changes[
        ["date", "effectiveDate", "resolutionDate"]
    ] = status_changes[["date", "effectiveDate", "resolutionDate"]].apply(
        pd.to_datetime
    )
    status_changes["year"] = status_changes.date.dt.year
    status_changes = status_changes.merge(
        teams, on=["team_id", "year"], how="inner"
    )

    return status_changes, teams
