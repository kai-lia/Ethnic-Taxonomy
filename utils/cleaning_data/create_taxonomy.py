import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

output_path = Path("data/ethnicity_class/taxonomy_structure.pkl")


# hand creatied the taxonomy structure that my data will map to 
# based on general demographic info, cited in writeup
ethnicity_classification = {
    #greater terms
    "asian" : [],
    "asian american": [],
    "asian american pacific islander": [],
    "aapi": [],
    "pacific islander": [],

    # high level asia
    "south asian": ["asian"],
    "east asian": ["asian"],
    "central asian": ["asian"],
    "southeast asian": ["asian"],
    "western asian": ["asian"],

    #pacific islander
    "melanesian": ["pacific islander"],
    "micronesian": ["pacific islander"],
    "polynesian": ["pacific islander"],

    #east asian
    "chinese": ["asian", "east asian"],
    "korean": ["asian", "east asian"],
    "japanese": ["asian", "east asian"],
    "mongolian": ["asian", "east asian"],
    "taiwanese": ["asian", "east asian"],

    #south asian
    "indian": ["asian", "south asian"],
    "bangladeshi": ["asian", "south asian"],
    "nepalese": ["asian", "south asian"],
    "sri lankan": ["asian", "south asian"],
    "pakistani": ["asian", "south asian"],
    "bhutanese": ["asian", "south asian"],
    "afghan": ["asian", "south asian"],
    "sikh": ["asian", "south asian"],
    "asian indian": ["asian", "south asian"],

    # central asian
    "kazakh": ["asian", "central asian"],
    "kyrgyz": ["asian", "central asian"],
    "tajik": ["asian", "central asian"],
    "uzbek": ["asian", "central asian"],

    #southeast asian
    "indonesian": ["asian", "southeast asian"],
    "malaysian": ["asian", "southeast asian"],
    "filipino": ["asian", "southeast asian"],
    "laotian": ["asian", "southeast asian"],
    "thai": ["asian", "southeast asian"],
    "singaporean": ["asian", "southeast asian"],
    "vietnamese": ["asian", "southeast asian"],
    "hmong": ["asian", "southeast asian"],
    "mien": ["asian", "southeast asian"],
    "cambodian": ["asian", "southeast asian"],
    "burmese": ["asian", "southeast asian"],

    #polynesian
    "hawaiian": ["pacific islander", "polynesian"],
    "native hawaiian": ["pacific islander", "polynesian"],
    "maori": ["pacific islander", "polynesian"],
    "samoan": ["pacific islander", "polynesian"],
    "tahitian": ["pacific islander", "polynesian"],
    "tongan": ["pacific islander", "polynesian"],

    #micornesean
    "chamorro": ["pacific islander", "micronesian"],
    "palauan": ["pacific islander", "micronesian"],
    "guamanian": ["pacific islander", "micronesian"],
    "marshallese": ["pacific islander", "micronesian"],
    "chuukese": ["pacific islander", "micronesian"],

    #melinensian
    "fijian": ["pacific islander", "melanesian"]

}

# save file
with open(output_path, "wb") as f:
    pickle.dump(ethnicity_classification, f)

print(f"ethnicity_classification saved to {output_path}")