import numpy as np

from .injury_map import (
    alt_injury_regex,
    injury_cat,
    injury_large_cat_map,
    injury_priority,
)
from .location_map import (
    alt_location_regex,
    location_cat,
    location_large_cat_map,
    location_priority,
)

DEFAULT_INJURY = "misc/unk"


class InjuryCategorizer:
    """Process injury notes and categorize injury location/type"""

    @staticmethod
    def _preprocess_raw_injury_data(injury_data):
        """Preprocess injury data.

        - Keep transactions to the IL
        - Process notes
        - Extract notes about injury
        - Extract number of days on IL
        Args:
            injury_data: injury data (prosports/statsapi)
        Returns:
            preprocessed injury data
        """

        # Only keep transactions to the IL
        injury_data = injury_data[
            ~injury_data.activated & ~injury_data.transfer
        ].reset_index(drop=True)
        # Notes to lower
        injury_data["notes"] = (
            injury_data["notes"]
            .str.lower()
            .str.replace("\.$", "", regex=True)
            .fillna("")
            .str.strip()
        )

        # Extract info about injury
        injury_data["injury_notes"] = (
            injury_data["notes"]
            .str.extract("([^.]+)$")[0]
            .str.replace("\(dtd\)", "", regex=True)
            .str.replace(
                ".* placed .* on the .* (disabled|injured) list(\s+)?(retroactive to [a-z]+\s+[0-9]+\,\s+[0-9]+)?(\.)?(\s+)?",
                "",
                regex=True,
            )
            .str.strip()
            .fillna("")
        )

        # Number of IL days
        injury_data["il_days"] = injury_data["notes"].str.extract(
            r"(?<=on the )(\d+)(\s|-)day"
        )[0]
        injury_data["il_days"] = (
            injury_data["il_days"]
            .fillna(
                injury_data["notes"].str.extract(
                    r"(?<=to the )(\d+)(\s|-)day"
                )[0]
            )
            .fillna(
                injury_data["notes"].str.extract(
                    r"(?<=from the )(\d+)(\s|-)day"
                )[0]
            )
        ).astype(float)
        injury_data.loc[injury_data["dtd"], "il_days"] = 0

        return injury_data

    @staticmethod
    def _injury_priority_map(injuries, mapping, priority):
        """Mapping to resolve injury priorities if there are multiple types/locations

        Args:
            injuries: list of injury types/locations
            mapping: mapping from injury to larger category
            priority: priority order
        Returns:
            highest priority injury
        """
        mapped_categories = list(set([mapping.get(i, i) for i in injuries]))
        if len(mapped_categories) == 0:
            return DEFAULT_INJURY

        priorities = np.array([priority.index(i) for i in mapped_categories])
        return mapped_categories[np.argmin(priorities)]

    def _resolve_injury_priority(self, injury_data):
        """Identify injury categories and for multiple identified
            injuries, resolve to highest priority injury type/location
        Args:
            injury_data: injury data
        Returns:
            categorized injury data
        """
        # Injury alternative spelling normalization
        injury_data["injury_type_notes"] = injury_data["injury_notes"].replace(
            alt_injury_regex, regex=True
        )

        # Assign injury type categories
        injury_data["injury_type"] = (
            injury_data["injury_type_notes"]
            .str.findall(f"({'|'.join(injury_cat)})")
            .apply(
                lambda x: self._injury_priority_map(
                    x, injury_large_cat_map, injury_priority
                )
            )
        )

        # Replace expressions for body types
        injury_data["injury_location_notes"] = injury_data[
            "injury_notes"
        ].replace(alt_location_regex, regex=True)

        # Assign injury location categories
        injury_data["injury_location"] = (
            injury_data["injury_location_notes"]
            .str.findall(f"({'|'.join(location_cat)})")
            .apply(
                lambda x: self._injury_priority_map(
                    x, location_large_cat_map, location_priority
                )
            )
        )

        return injury_data

    def process(self, injury_data):

        return injury_data.pipe(self._preprocess_raw_injury_data).pipe(
            self._resolve_injury_priority
        )
