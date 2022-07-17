injury_cat = [
    "unkown",
    "misc",
    "strain",
    "internal",
    "soreness",
    "sprain",
    "contusion",
    "fracture",
    "break",
    "inflammation",
    "concussion",
    "surgery",
    "tightness",
    "dislocation",
    "laceration",
    "spasm",
    "hernia",
    "minor",
    "bruise",
    "stiff",
    "impingement",
    "loose bodies",
    "tear",
    "discomfort",
    "fatigue",
    "infection",
    "tommy john",
    "blister",
    "irritation",
    "spur",
    "inflammation",
    "nerve",
]


alt_injury_spellings = {
    "soreness": ["sore"],
    "tear": ["torn", "rupture"],
    "stiff": ["siff"],
    "internal": [
        "flu",
        "heart",
        "virus",
        "covid-19",
        "vertigo",
        "stomach ailment",
        "dehydration",
        "dizziness",
        "food poisoning",
        "strep throat",
        "allerg",
        "ulcerative colitis",
        "gastroenteritis",
        "medical condition",
        "bladder ailment",
        "cyst",
        "cancer",
        "disease",
        "stomach",
        "pneumonia",
        "illness",
        "blood clot",
        "irregular heart",
        "liver",
        "kidney",
        "lung",
        "tooth",
        "teeth",
        "mouth",
        " ear |^ear ",
        "cold",
        "eye",
        "gastro",
        "cramp",
        "vision",
        "root canal",
        "append",
        "esopha",
        "colitis",
        "fever",
        "pox",
        "gas",
        "infect",
        "viral",
        "abscess",
        "migraine",
        "throat",
        "nose",
        "bladder",
        "personal reasons",
        "testi",
        "dental",
        "jaw",
        "mono",
        "bronch",
        "thrombosis",
        "shingles",
        "brain",
        "abdominal issues",
        "lightheaded",
        "cellulitis",
        "gall bladder",
        "upset stomach",
    ],
    "fatigue": ["weakness"],
    "inflammation": [
        "plantar fasciitis",
        "fascitis",
        "tendinits",
        "inflammtion",
        "inflam",
        "tendonitis",
        "tendinopathy",
        "costochondritis",
        "bursitis",
        "tendinitis",
        "sesamoiditis",
        "epicondylitis",
    ],
    "break": ["crack", "brok"],
    "sprain": ["turf toe", "prained"],
    "nerve": ["neuritis", "thoracic outlet syndrome"],
    "fracture": ["stress reaction", "stress injury", "streaa"],
    "tightness": ["tigh"],
    "laceration": ["lacerated"],
    "surgery": ["repair", "debridement", "procedure", "construction"],
    "dislocation": ["subluxation", "hyperextend", "sublux", "disloca"],
    "spine surgery": ["microdiscectomy"],
    "spine": ["spinal", "pars defect"],
    "bruise": ["hematoma"],
}
alt_injury_regex = {"|".join(v): k for k, v in alt_injury_spellings.items()}


# injury_large_cat_map = {
#     "break/fracture": ["break", "fracture"],
#     "minor": [
#         "spasm",
#         "irritation",
#         "discomfort",
#         "tightness",
#         "fatigue",
#         "stiff",
#         "laceration",
#         "blister",
#         "soreness",
#     ],
#     "inflammation": ["impingement"],
#     "contusion/bruise": ["contusion", "bruise"],
#     "misc/unk": [
#         "misc",
#         "unknown",
#         "nerve",
#         "dislocation",
#         "hernia",
#         "loose bodies",
#         "spur",
#         "concussion",
#     ],
#     "internal": ["internal"],
#     "tommy john": ["tommy john"],
#     "surgery": ["surgery"],
#     "tear": ["tear"],
#     "strain": ["strain"],
#     "sprain": ["sprain"],
# }


injury_large_cat_map = {
    "break/fracture": ["break", "fracture"],
    "minor": [
        "spasm",
        "irritation",
        "discomfort",
        "tightness",
        "fatigue",
        "stiff",
        "laceration",
        "blister",
        "soreness",
    ],
    "inflammation": ["impingement"],
    "contusion/bruise": ["contusion", "bruise"],
    "misc/unk": [
        "misc",
        "unknown",
        "nerve",
        "dislocation",
        "hernia",
        "loose bodies",
        "spur",
        "concussion",
    ],
    "internal": ["internal"],
    "surgery": ["surgery", "tommy john"],
    "tear/strain/sprain": ["tear", "strain", "sprain"],
}


injury_large_cat_map = {
    x: k for k, v in injury_large_cat_map.items() for x in v
}

# severity_order = [
#     "internal",
#     "tommy john",
#     "surgery",
#     "tear",
#     "break/fracture",
#     "strain",
#     "sprain",
#     "contusion/bruise",
#     "inflammation",
#     "minor",
#     "misc/unk",
# ]


injury_priority = [
    "surgery",
    "tear/strain/sprain",
    "break/fracture",
    "contusion/bruise",
    "inflammation",
    "minor",
    "misc/unk",
    "internal",
]
