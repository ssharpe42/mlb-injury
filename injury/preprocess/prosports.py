"""Align and clean prosports data to combine with MLB IL transactions"""
from itertools import product

import numpy as np
import pandas as pd


def remove_accents(a):
    return (
        a.str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )


class ProsportsCleaner:
    def __init__(self, prosports, teams):
        self.prosports = prosports
        self.teams = teams

    def _clean_cols(self, prosports):
        # Clean up columns
        prosports.columns = prosports.columns.str.lower()
        prosports = prosports.rename(columns={"team": "team_name"})

        prosports = prosports.assign(
            # Add date cols
            date=pd.to_datetime(prosports["date"]),
            year=lambda x: x["date"].dt.year,
            # Add name column based on acquired/relinquished
            activated=prosports["relinquished"].isnull(),
            name=np.where(
                prosports["acquired"].isnull(),
                prosports["relinquished"],
                prosports["acquired"],
            ),
            # Day to day
            dtd=prosports["notes"].str.contains("DTD"),
        )
        # # Add date cols
        # prosports["date"] = pd.to_datetime(prosports["date"])
        # prosports["year"] = prosports["date"].dt.year

        # # Add name column based on acquired/relinquished
        # prosports["activated"] = prosports["relinquished"].isnull()
        # prosports["name"] = np.where(
        #     prosports["acquired"].isnull(),
        #     prosports["relinquished"],
        #     prosports["acquired"],
        # )
        # # Day to day
        # prosports["dtd"] = prosports.notes.str.contains("DTD")

        return prosports.drop(columns=["acquired", "relinquished"])

    def _clean_player_names(self, prosports):

        prosports["name"] = (
            prosports["name"]
            .str.replace(r"[\(\[].*?[\)\]]", "", regex=True)
            .str.replace("•", "", regex=False)
        )
        # prosports["name"] = prosports["name"].str.replace("•", "", regex=False)

        # remove non-names & empty notes
        prosports = prosports[
            prosports["name"].notnull()
            & prosports["name"].str.contains("[a-z]")
            & prosports["notes"].notnull()
        ]

        # replace abbrev
        has_abbrev = prosports.name.str.contains("([A-Z]|r)\.")
        prosports.loc[has_abbrev, "name"] = prosports.loc[
            has_abbrev, "name"
        ].str.replace("\.", "", regex=True)

        # lower/remove accents
        prosports["name"] = remove_accents(prosports["name"]).str.lower()

        # split names
        prosports[["name", "name2"]] = prosports["name"].str.split(
            "/", expand=True
        )
        prosports[["name", "name2"]] = prosports[["name", "name2"]].apply(
            lambda x: x.str.strip()
        )
        return prosports

    def _merge_teams(self, prosports, teams):

        teams.loc[teams.team == "ARI", "team_name"] = "Diamondbacks"
        return prosports.merge(teams, on=["year", "team_name"], how="left")

    def _finalize_prosports(self, prosports):

        prosports["id"] = list(prosports.index)
        return prosports[prosports["dtd"]]

    def clean(self):

        return (
            self.prosports.pipe(self._clean_cols)
            .pipe(self._clean_player_names)
            .pipe(self._merge_teams, teams=self.teams)
            .pipe(self._finalize_prosports)
        )


class AlignProsportsMLB:
    def __init__(self, prosports, mlb_players):
        """Align prosports injuries with MLB ids

        Args:
            prosports: cleaned prosports dataframe
            mlb_players: dataframe of all mlb player games
        """
        self.prosports = prosports
        self.players = mlb_players

    def _clean_player_names(self, players):

        # Normalize player names for matching
        players["full_name"] = remove_accents(
            players.name.apply(lambda x: x.split(", ")[1])
            + " "
            + players.name.apply(lambda x: x.split(", ")[0])
        ).str.replace("\.", "", regex=True)

        # remove middle initial
        players.loc[
            players.full_name.str.contains("[A-Za-z]+ [A-Z] [A-Za-z]+"),
            "full_name",
        ] = players.full_name[
            players.full_name.str.contains("[A-Za-z]+ [A-Z] [A-Za-z]+")
        ].apply(
            lambda x: x.split(" ")[0] + " " + x.split(" ")[2]
        )

        players["lower_full_name"] = players["full_name"].str.lower()

        return players

    def _get_injured_player_info(self, prosports, players):

        injured_players = players[
            players["lower_full_name"].isin(prosports["name"])
            | players["lower_full_name"].isin(prosports["name2"])
        ]

        # All players x all dates in range
        date_range = pd.date_range(
            players["game_date"].min(), players["game_date"].max()
        )
        player_dt_rng = pd.DataFrame(
            list(product(date_range, injured_players.player_id.unique())),
            columns=["game_date", "player_id"],
        )

        injured_players_by_date = player_dt_rng.merge(
            injured_players, how="left", on=["player_id", "game_date"]
        )

        # Fill back team and name information for all empty dates before a game
        injured_players_by_date[
            ["name", "team", "lower_full_name", "full_name"]
        ] = injured_players_by_date.groupby(["player_id"])[
            ["name", "team", "lower_full_name", "full_name"]
        ].bfill()

        # Fill forward
        injured_players_by_date[
            ["name", "team", "lower_full_name", "full_name"]
        ] = injured_players_by_date.groupby(["player_id"])[
            ["name", "team", "lower_full_name", "full_name"]
        ].ffill()

        # Get all teams each player played games for during time span
        all_teams = (
            injured_players.groupby(["player_id", "lower_full_name"])["team"]
            .apply(lambda x: list(set(x)))
            .reset_index(name="all_teams")
        )
        return injured_players_by_date.merge(
            all_teams, on=["player_id", "lower_full_name"], how="left"
        ).drop_duplicates(["game_date", "player_id"])

    def _match_injuries_and_players(self, prosports, injured_players_by_date):

        player_cols = [
            "game_date",
            "player_id",
            "lower_full_name",
            "full_name",
            "team",
            "all_teams",
        ]
        matched_prosports = prosports.merge(
            injured_players_by_date[player_cols].rename(
                columns={"team": "team_x"}
            ),
            left_on=["date", "name"],
            right_on=["game_date", "lower_full_name"],
            how="left",
        ).merge(
            injured_players_by_date[player_cols].rename(
                columns={"team": "team_y"}
            ),
            left_on=["date", "name2"],
            right_on=["game_date", "lower_full_name"],
            how="left",
        )
        matched_prosports = matched_prosports.assign(
            player_id=matched_prosports["player_id_x"].fillna(
                matched_prosports["player_id_y"]
            ),
            full_name=matched_prosports["full_name_x"].fillna(
                matched_prosports["full_name_y"]
            ),
            team_match=(
                (matched_prosports["team"] == matched_prosports["team_x"])
                | (matched_prosports["team"] == matched_prosports["team_y"])
            ).fillna(False),
            total_same_team=lambda x: x.groupby(["id"])[
                "team_match"
            ].transform(sum),
            total_player_id=lambda x: x.groupby(["id"])["player_id"].transform(
                "nunique"
            ),
            all_teams_x=matched_prosports["all_teams_x"].apply(
                lambda d: d if isinstance(d, list) else []
            ),
            all_teams_y=matched_prosports["all_teams_y"].apply(
                lambda d: d if isinstance(d, list) else []
            ),
        )

        # Does the players team from prosports match with any team they played
        # for during the timespan
        matched_prosports["career_team_match"] = matched_prosports.apply(
            lambda x: x["team"] in x["all_teams_x"], axis=1
        ) | matched_prosports.apply(
            lambda x: x["team"] in x["all_teams_y"], axis=1
        )

        return matched_prosports

    def _remove_duplicates(self, matched_prosports):

        exact_match = matched_prosports.query("total_player_id==1")
        no_match = matched_prosports.query("total_player_id==0")

        # Multiple matches on a player name
        multi_match = matched_prosports.query("total_player_id>1")

        # One player id matches the corresponding team
        exact_match_team = (
            multi_match.query("team_match")
            .groupby("id")
            .filter(lambda x: x.player_id.nunique() == 1)
        )

        # Multiple matches on player name with no team matches
        multi_noteam_match = multi_match[
            ~multi_match.id.isin(exact_match_team.id) & ~multi_match.team_match
        ]
        # No team matches, but one player matches on team during whole career
        exact_match_career = (
            multi_noteam_match.query("career_team_match")
            .groupby("id")
            .filter(lambda x: x.player_id.nunique() == 1)
        )
        return (
            pd.concat([exact_match, exact_match_team, exact_match_career])
            .drop(
                columns=[
                    "name",
                    "name2",
                    "game_date_x",
                    "player_id_x",
                    "lower_full_name_x",
                    "full_name_x",
                    "team_x",
                    "all_teams_x",
                    "game_date_y",
                    "player_id_y",
                    "lower_full_name_y",
                    "full_name_y",
                    "team_y",
                    "all_teams_y",
                    "team_match",
                    "total_same_team",
                    "total_player_id",
                    "career_team_match",
                ]
            )
            .rename(columns={"full_name": "name"})
        )

    def run(self):

        clean_players = self._clean_player_names(self.players)
        injured_players_by_date = self._get_injured_player_info(
            self.prosports, clean_players
        )
        matched_prosports = self._match_injuries_and_players(
            self.prosports, injured_players_by_date
        )
        return self._remove_duplicates(matched_prosports)
