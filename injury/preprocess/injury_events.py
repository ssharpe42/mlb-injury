import numpy as np
import pandas as pd
from IPython import embed

from .injury_map import injury_priority
from .location_map import location_priority


def add_cols_player_games(player_data):
    """Add date, year columns to player game data

    Args:
        player_data: player game data
    Returns:
        _description_
    """

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
    # def __init__(self, injury_data, mlb_players):
    #     self.injury_data = injury_data
    #     self.mlb_players = mlb_players

    @staticmethod
    def _combine_injuries_and_games(injury_data, player_data):
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

        return data.sort_values(["date", "game"], ascending=[True, False])

    @staticmethod
    def _game_indicators(data):
        """Fill game and prev/next game indicators for calculating spans """

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
        data["prev_game_date_game"] = data.groupby("player_id")[
            "game_date"
        ].shift(1)
        data.loc[data["game"] == 1, "prev_game_date"] = data.loc[
            data["game"] == 1, "prev_game_date_game"
        ]
        data.drop(columns=["prev_game_date_game"], inplace=True)

        return data

    @staticmethod
    def _calculate_non_mlb_time(data, season_days):
        """Calculate time not spent actively playing in MLB
            - At least 15 in-season days between games

        Args:
            data: injury event data
            season_days: cumulative season days
        Returns:
            injury event data with mlb time
        """

        data = data.merge(
            season_days.rename(
                columns={
                    "date": "prev_game_date",
                    "season_days": "prev_season_days",
                }
            ),
            how="left",
        ).merge(
            season_days.rename(
                columns={"date": "game_date", "season_days": "cum_season_days"}
            ),
            how="left",
        )
        return data.assign(
            time_since_last_game=(
                data["cum_season_days"] - data["prev_season_days"]
            ),
            non_mlb_days=lambda x: x["time_since_last_game"]
            * ((x["time_since_last_game"] > 15) & x["prev_is_game"]),
        ).drop(columns=["prev_season_days", "cum_season_days"])

    @staticmethod
    def _compute_injury_spans(data):
        """Calculate injury spans (when injury time frame starts and ends)

        - set original span_id as cumulative games
        - set days where players played games to have NaN injury id
        - backfill ids for game days
        - set days with played games to next injury span - 1
        - set first/last row ids for start end tokens

        """

        # Injury span
        data["injury_span_id"] = data.groupby("player_id")["game"].transform(
            "cumsum"
        )
        # Remove for non-injuries (games)
        data.loc[data["game"] == 1, "injury_span_id"] = np.nan

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

        data.loc[data["game"] == 0, "player_last_row"] = data.groupby(
            ["player_id", "injury_span_id"]
        )["player_last_row"].transform("max")[data["game"] == 0]

        data.loc[data["game"] == 0, "player_first_row"] = data.groupby(
            ["player_id", "injury_span_id"]
        )["player_first_row"].transform("max")[data["game"] == 0]

        data["injury_span_id_orig"] = data["injury_span_id"]

        data["injury_span_id"] = (
            data["injury_span_id"].astype(int).astype(str)
            + "_"
            + data["player_first_row"].astype(int).astype(str)
            + data["player_last_row"].astype(int).astype(str)
        )
        # need to account for multiple injuries in last span

        return data

    @staticmethod
    def _summarize_injury_spans(data):
        """ Summarise injury data for each injury span by resolving injuries """
        data = data.assign(
            injury_type=pd.Categorical(
                data["injury_type"], np.flip(injury_priority)
            ).as_ordered(),
            injury_location=pd.Categorical(
                data["injury_location"], np.flip(location_priority)
            ).as_ordered(),
            non_mlb_days=data.groupby(["player_id", "injury_span_id_orig"])[
                "non_mlb_days"
            ].transform(sum),
            il_days_max=data.groupby(["player_id", "injury_span_id_orig"])[
                "il_days"
            ].transform(max),
            il_days_sum=data.groupby(["player_id", "injury_span_id_orig"])[
                "il_days"
            ].transform(sum),
            dtd=data.groupby(["player_id", "injury_span_id_orig"])["dtd"]
            .transform(min)
            .astype(bool),
        )

        def resolve_injury(df, col="injury_type"):
            """Resolve injuries in a span

            If all are null/misc return the highest priority location/type

            Otherwise find the longest il day stint that is not misc, and tie break all
            il stints of that length with the location priority
            """

            if df[
                df["injury_location"].notnull()
                & (df["injury_location"] != "misc/unk")
            ].empty:
                df[["injury_type", "injury_location"]] = df[
                    ["injury_type", "injury_location"]
                ].max(axis=0)
            else:

                df_ = df[df["injury_location"] != "misc/unk"].sort_values(
                    "il_days", ascending=False
                )
                df_ = df_[
                    df_.il_days == df_.iloc[0, df.columns.get_loc("il_days")]
                ].sort_values("injury_location")
                df[["injury_type", "injury_location"]] = df_.iloc[
                    0,
                    [
                        df.columns.get_loc("injury_type"),
                        df.columns.get_loc("injury_location"),
                    ],
                ]
            return df

        data = data.groupby(["player_id", "injury_span_id_orig"]).apply(
            resolve_injury
        )

        # First row per player injury span
        data = (
            data.sort_values(["date", "game"], ascending=[True, False])
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

        return data

    @staticmethod
    def _filter_to_events(data):
        """Filter to events (start, injuries, end)"""
        return (
            data[
                (data["game"] == 0)
                | data["player_first_row"]
                | data["player_last_row"]
            ]
            .reset_index(drop=True)
            .sort_values(["player_id", "date"])
        )

    @staticmethod
    def _create_event_dataframe(data, season_days):
        """Add start end tokens and compute time passed between injuries"""

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

    @staticmethod
    def _remove_internal_injuries(data):
        """Remove internal non-baseball injuries"""

        data = data[data["injury_type"] != "internal"]
        data["dt"] = (
            data["t"] - data.groupby("player_id")["t"].shift(1)
        ).fillna(0)
        return data

    def process(self, injury_data, mlb_players):
        """Run injury event processing

        Args:
            injury_data: injury data
            mlb_players: player game_data
        Returns:
            dataframe of injury events
        """

        player_data = mlb_players.pipe(add_cols_player_games)
        season_days = cum_season_days(player_data)
        return (
            self._combine_injuries_and_games(injury_data, player_data)
            .pipe(self._game_indicators)
            .pipe(self._compute_injury_spans)
            .pipe(self._calculate_non_mlb_time, season_days=season_days)
            .pipe(self._summarize_injury_spans)
            .pipe(self._filter_to_events)
            .pipe(self._create_event_dataframe, season_days=season_days)
            .pipe(self._remove_internal_injuries)
        )
