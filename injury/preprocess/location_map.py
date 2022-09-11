location_cat = [
    "upper leg",
    "lower leg",
    "upper arm",
    "lower arm",
    "hand",
    "(?<![a-z])back",  # ex: diamondbacks
    "knee",
    "internal",
    "torso",
    "shoulder",
    "foot",
    "elbow",
    "(?<![a-z])hip",  # ex: chip
    "unknown",
    "quad",
    "hamstring",
    "oblique",
    "wrist",
    "groin",
    "ankle",
    "calf",
    "head/neck",
]
general_location_cat = ["arm", "leg"]

location_cat = location_cat + general_location_cat


alt_location_spellings = {
    " hand ": ["finger", "thumb", "hamate", "carpal", "nail", "digit"],
    " lower arm ": ["forearm", "pronator", "ulna bone", "flexor mass"],
    " foot ": ["toe", "heel", "plantar", "heal"],
    " head/neck ": [
        "neck",
        "head",
        "concussion",
        "cervic",
        "facial",
        "nose",
        "mouth",
        "cheek",
        "jaw",
        "eye",
    ],
    " elbow ": ["tommy john", "ucl", "ulnar"],
    " upper arm ": ["bicep", "tricep", "radial nerve"],
    " shoulder ": [
        "rotator cuff",
        "ac joint",
        "sc joint",
        "teres",
        "deltoid",
        "clavicle",
        "collarbone",
        "a/c joint",
    ],
    " back ": [
        "spin",
        "lumbar",
        " lat | lat$|^lat ",
        "trapezius",
        "pars",
        "disc |disc$|disk",
        "rhomboid",
        "tailbone",
        "thoracic",
        "(s\.i\.|si) joint",
    ],
    " torso ": [
        "rib",
        "intercostal",
        "peck",
        "chest",
        "pector",
        "abdo",
        "core",
        " side injury|^(right |left )?side [a-z]+$",
    ],
    " ankle ": ["achill"],
    " upper leg ": ["adductor", "glute", "thigh"],
    " knee ": ["mcl | mcl", "acl | acl", "patella", "meniscus"],
    " lower leg ": ["fibula", "(?<![a-z])shin", "tibia"],
    " hip ": ["pelvic", "pelv"],
    " oblique ": ["obl"],
    " hamstring ": ["h[a-z]+string"],
}
alt_location_regex = {
    "|".join(v): k for k, v in alt_location_spellings.items()
}


location_large_cat_map = {
    "other arm": ["arm", "upper arm", "lower arm"],
    "other leg": ["leg", "upper leg", "lower leg", "quad", "calf"],
    "hip": ["groin"],
    "torso": ["oblique"],
}

location_large_cat_map = {
    x: k for k, v in location_large_cat_map.items() for x in v
}

location_priority = [
    "shoulder",
    "elbow",
    "knee",
    "back",
    "hamstring",
    "hand",
    "wrist",
    "foot",
    "ankle",
    "hip",
    "torso",
    "head/neck",
    "other arm",
    "other leg",
    "misc/unk",
]
