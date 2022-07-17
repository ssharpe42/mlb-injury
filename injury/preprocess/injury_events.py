import numpy as np
import pandas as pd

from .injury_map import injury_priority
from .location_map import location_priority


def add_cols_player_games(player_data):

    # Get all player games
    return (
        player_data.assign(
            date=player_data["game_date"],
            year=player_data["game_date"].dt.year,
            game=1,
        )
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    # player_data["cum_games"] = player_data.groupby("player_id")["game"].transform(
    #     "cumsum"
    # )
    # return player_data


def cum_season_days(data):
    """Get cumulative season days (days inside season start and end)
    for a dataset containing all games in a certain time span

    Args:
        data: pandas dataframe containing year and game_date for all games
    Returns:
        dataframe of dates and cumulative season days since min date
    """

    # Max and min dates by season
    season_dates = (
        data.groupby(["year"], as_index=False)
        .agg(season_max=("game_date", "max"))
        .merge(
            data.groupby("year", as_index=False).agg(
                season_min=("game_date", "min")
            ),
            on="year",
            how="left",
        )
    )

    # Full date range
    dt_range = (
        pd.DataFrame(
            {
                "date": pd.date_range(
                    start=data.game_date.min(),
                    end=data.game_date.max(),
                )
            }
        )
        .assign(year=lambda x: x["date"].dt.year)
        .merge(season_dates, on=["year"])
    )

    # Indicator if date is within season
    dt_range["in_season"] = dt_range["date"].between(
        dt_range["season_min"], dt_range["season_max"]
    )
    # Cumulative season days
    dt_range["season_days"] = dt_range["in_season"].cumsum()
    return dt_range[["date", "season_days"]]


class InjuryEvents:
    def __init__(self, injury_data, mlb_players):
        self.injury_data = injury_data
        self.mlb_players = mlb_players

    def _combine_injuries_and_games(self, injury_data, player_data):
        """Combine player games and injuries to determine injury time spans
        and active player windows.

        Args:
            injury_data: dataframe of injuries
            player_data: dataframe of player games
        Returns:
            unioned dataframe of all events (injury/games)
        """

        injury_data = injury_data.assign(
            game=0,
            injury_id=injury_data.groupby("player_id").cumcount(),
            injury_date=injury_data["date"],
        )

        # combine injuries and players
        data = pd.concat([injury_data, player_data]).reset_index(drop=True)

        # Use first name from injury data otherwise player data
        data["name"] = data.groupby("player_id")["name"].transform("first")

        return data.sort_values("date")

    def _game_indicators(self, data):

        # Get indicators and dates for prev/next games
        data = data.assign(
            prev_is_game=data.groupby("player_id")["game"].shift(1),
            next_is_game=data.groupby("player_id")["game"].shift(-1),
            prev_game_date=data["game_date"],
            next_game_date=data["game_date"],
        )

        data = data.assign(
            next_game_date=data.groupby("player_id")["next_game_date"].bfill(),
            prev_game_date=data.groupby("player_id")["prev_game_date"].ffill(),
            # Begin and end of player time in MLB in data sample
            player_first_row=data["prev_is_game"].isnull(),
            player_last_row=data["next_is_game"].isnull(),
        )

        return data

    def _calculate_non_mlb_time(self, data):

        # Calculate time spent not playing/outside of majors
        # Any >15 day span not injured within the same year
        return data.assign(
            time_since_last_game=np.where(
                data["game_date"].dt.year
                == data.groupby("player_id")["game_date"].shift(1).dt.year,
                (
                    data["game_date"]
                    - data.groupby("player_id")["game_date"].shift(1)
                ).dt.days,
                0,
            ),
            non_mlb_days=lambda x: x["time_since_last_game"]
            * ((x["time_since_last_game"] > 15) & x["prev_is_game"]),
        )

    def _compute_injury_spans(self, data):

        data = self._game_indicators(data)

        # Injury span
        data["injury_span_id"] = data.groupby("player_id")["game"].transform(
            "cumsum"
        )
        # Remove for non-injuries (games)
        data.loc[
            data["game"] == 1,
            ["prev_game_date", "next_game_date", "injury_span_id"],
        ] = np.nan

        # Create span ids during healthy playing days
        data["injury_span_id"] = data.groupby("player_id")[
            "injury_span_id"
        ].fillna(method="bfill")
        data.loc[data["game"] == 1, "injury_span_id"] = (
            data.loc[data["game"] == 1, "injury_span_id"] - 1
        )
        data["injury_span_id"] = data["injury_span_id"].fillna(
            data["injury_span_id"].max() + 1
        )

        data = self._calculate_non_mlb_time(data)

        return data

    def _summarize_injury_spans(self, data):

        data = data.assign(
            injury_type=pd.Categorical(
                data["injury_type"], np.flip(injury_priority)
            ).as_ordered(),
            injury_location=pd.Categorical(
                data["injury_location"], np.flip(location_priority)
            ).as_ordered(),
            non_mlb_days=data.groupby(["player_id", "injury_span_id"])[
                "non_mlb_days"
            ].transform(sum),
            il_days_max=data.groupby(["player_id", "injury_span_id"])[
                "il_days"
            ].transform(max),
            il_days_sum=data.groupby(["player_id", "injury_span_id"])[
                "il_days"
            ].transform(sum),
        )

        data[["injury_location", "injury_type"]] = data.groupby(
            ["player_id", "injury_span_id"]
        )[["injury_location", "injury_type"]].transform(max)

        # First row per player injury span
        data = (
            data.sort_values(["date"])
            .groupby(["player_id", "injury_span_id"], as_index=False)
            .nth(0)
        )

        # Last span days in the minors/out of mlb
        data["prev_non_mlb_days"] = (
            data.groupby(["player_id"])["non_mlb_days"].shift(1).fillna(0)
        )

        data[["injury_location", "injury_type"]] = (
            data[["injury_location", "injury_type"]]
            .astype(str)
            .replace("nan", np.nan)
        )

        return self._filter_to_events(data)

    def _filter_to_events(self, data):
        return (
            data[
                (data["game"] == 0)
                | data["player_first_row"]
                | data["player_last_row"]
            ]
            .reset_index(drop=True)
            .sort_values(["player_id", "date"])
        )

    def _create_event_dataframe(self, data, season_days):

        data.loc[
            data["player_first_row"], ["injury_type", "injury_location"]
        ] = data.loc[
            data["player_first_row"], ["injury_type", "injury_location"]
        ].fillna(
            "[START]"
        )

        data.loc[
            data["player_last_row"], ["injury_type", "injury_location"]
        ] = data.loc[
            data["player_last_row"], ["injury_type", "injury_location"]
        ].fillna(
            "[END]"
        )

        data = data.assign(
            start_date=data.groupby("player_id")["date"].transform("min"),
            prev_injury_end_date=data.groupby("player_id")[
                "next_game_date"
            ].shift(1)
            - pd.Timedelta(1, "days"),
        )

        data["prev_injury_end_date"] = data["prev_injury_end_date"].fillna(
            data["start_date"]
        )

        # Cumulative season days (current, prev injury, and next)
        cum_days = season_days.rename(
            columns={"season_days": "cum_season_days"}
        )
        prev_cum_days = season_days.rename(
            columns={
                "season_days": "prev_season_days",
                "date": "prev_injury_end_date",
            }
        )
        next_cum_days = season_days.rename(
            columns={
                "season_days": "next_game_season_days",
                "date": "next_game_date",
            }
        )

        data = (
            (
                data.merge(cum_days, how="left")
                .merge(prev_cum_days, how="left")
                .merge(next_cum_days, how="left")
            )
            .sort_values(["player_id", "date"])
            .fillna({"prev_season_days": 0, "cum_season_days": 0})
        )

        # Time measurements
        data = data.assign(
            dt=lambda x: np.maximum(
                x["cum_season_days"]
                - x["prev_season_days"]
                - x["prev_non_mlb_days"],
                0,
            ),
            in_season_duration=lambda x: x["next_game_season_days"]
            - x["cum_season_days"],
            date_duration=lambda x: (x["next_game_date"] - x["date"]).dt.days,
            duration=lambda x: x[["in_season_duration", "il_days_max"]].max(
                axis=1
            ),
        )

        data["t"] = data.groupby("player_id")["dt"].transform("cumsum")

        return data

    def _remove_internal_injuries(self, data):

        data = data[data["injury_type"] != "internal"]
        data["dt"] = data["t"] - data.groupby("player_id")["t"].shift(1)
        return data

    def process(self):

        player_data = self.mlb_players.pipe(add_cols_player_games)
        season_days = cum_season_days(player_data)
        return (
            self._combine_injuries_and_games(self.injury_data, player_data)
            .pipe(self._compute_injury_spans)
            .pipe(self._summarize_injury_spans)
            .pipe(self._create_event_dataframe, season_days=season_days)
            .pipe(self._remove_internal_injuries)
        )
