"""Static domain tables for normalization (hand-written; no external data sources).

Sources of the word lists: research/03_noise_audit.md (measured on train pairs) and
research/02_normalization_tools.md. State/region tables are general geographic knowledge,
not entity lookups.
"""

# ------------------------------------------------------------------ names
# variant (post-ascii, lowercase, dots removed) -> canonical legal form
LEGAL_TOKENS = {
    # US
    "inc": "inc", "incorporated": "inc", "lnc": "inc", "llc": "llc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "lp": "lp", "llp": "llp", "pllc": "pllc",
    "pc": "pc", "plc": "plc", "lllp": "lllp",
    # India
    "pvt": "pvt", "private": "pvt", "opc": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci", "sa": "sa", "snc": "snc",
    "scp": "scp", "selarl": "selarl", "scm": "scm", "sca": "sca", "cie": "cie", "eirl": "eirl", "gie": "gie",
    "scop": "scop", "fils": "fils", "freres": "freres",
    # other common
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}
# multi-word legal phrases, matched before single tokens (longest first)
LEGAL_PHRASES = {
    "limited liability company": "llc", "limited liability partnership": "llp", "one person company": "opc",
    "professional corporation": "pc", "private limited": "pvt ltd", "et fils": "fils", "and fils": "fils",
    "et freres": "freres", "and freres": "freres", "and sons": "fils", "and son": "fils", "and co": "co",
}

# words the noisy sources add to names (audit table 1j) + honorifics + French equivalents
FILLER = {
    "center", "centre", "services", "service", "partners", "enterprises", "enterprise", "trading", "the",
    "messrs", "ms", "m", "sri", "shri", "shree", "smt", "dr", "mr", "mrs", "kumari", "sh",
    "groupe", "group", "holding", "participations", "distribution", "associes",
}
STOP = {"and", "of", "a", "an", "for", "de", "du", "des", "la", "le", "les", "et", "d", "l", "au", "aux"}
# professional credentials: removed only when they are the trailing token(s) of a name
CREDENTIALS = {"dds", "dpm", "cpa", "md", "od", "dmd", "dvm", "phd", "esq", "np", "rn", "lcsw", "dc", "facs",
               "jd", "mba", "lpc", "do", "pa", "pt"}

# ------------------------------------------------------------------ addresses
NULL_SEGS = {"null", "<null>", "none", "nan", "n/a", "na", "nil", "-", "--", "unknown", "not available"}
ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
            "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th",
            "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th"}
# street type / direction canonicalisation. 'saint' -> 'st' on purpose: the noise mis-expands St -> Saint.
STREET = {
    "street": "st", "str": "st", "saint": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave",
    "av": "ave", "ave": "ave", "boulevard": "blvd", "bd": "blvd", "blvd": "blvd", "bvd": "blvd", "drive": "dr",
    "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "circle": "cir", "cir": "cir",
    "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq",
    "route": "rte", "rte": "rte", "expressway": "expy", "freeway": "fwy", "turnpike": "tpke", "way": "way",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    # France
    "rue": "rue", "r": "rue", "allee": "all", "all": "all", "impasse": "imp", "imp": "imp", "chemin": "ch",
    "che": "ch", "chem": "ch", "faubourg": "fbg", "fg": "fbg", "fbg": "fbg", "quai": "qu", "qu": "qu",
    "cours": "crs", "crs": "crs", "passage": "pas", "pass": "pas", "residence": "res", "res": "res",
    "lieu dit": "ld", "lieudit": "ld", "ld": "ld", "zone": "z", "za": "za", "zi": "zi", "zac": "zac",
    # India
    "marg": "marg", "mg": "marg", "nagar": "ngr", "ngr": "ngr", "colony": "col", "col": "col", "sector": "sec",
    "sec": "sec", "industrial": "ind", "indl": "ind", "ind": "ind", "estate": "est", "est": "est",
    "cross": "crs", "main": "main", "layout": "lyt", "extension": "extn", "extn": "extn", "ext": "extn",
    "mohalla": "mohalla", "gali": "gali", "chowk": "chowk", "bazar": "bazaar", "bazaar": "bazaar",
}
# unit / house-number prefix words: dropped from tokens, their numbers are kept
UNIT_WORDS = {
    "unit", "suite", "ste", "apt", "apartment", "fl", "floor", "flr", "room", "rm", "bldg", "building",
    "shop", "office", "off", "no", "nos", "number", "num", "h", "hno", "house", "plot", "sf", "door", "dno",
    "khasra", "kh", "flat", "survey", "sy", "ground", "gf", "ff", "sf", "tf", "basement", "lot", "pmb", "bp",
    "cedex", "n", "box", "po", "pob",
}
FR_REPEAT = {"bis": "b", "ter": "t", "quater": "q"}
CITY_AFFIX = ("city of ", "town of ", "village of ", "township of ")
CITY_SUFFIX = (" city", " twp", " township", " town")
IN_CITY = {"bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
           "gurgaon": "gurugram", "poona": "pune", "mysore": "mysuru", "trivandrum": "thiruvananthapuram",
           "baroda": "vadodara", "cochin": "kochi", "benares": "varanasi", "allahabad": "prayagraj",
           "mangalore": "mangaluru", "belgaum": "belagavi", "hubli": "hubballi", "pondicherry": "puducherry",
           "mumbai city": "mumbai", "bangalore urban": "bengaluru", "bengaluru urban": "bengaluru"}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "dc": "district of columbia", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "pr": "puerto rico", "gu": "guam",
    "vi": "virgin islands",
}
US_STATE_MAP = {**{k: k for k in US_STATES}, **{v: k for k, v in US_STATES.items()},
                "washington dc": "dc", "d c": "dc"}

# India: canonical code. Andhra Pradesh and Telangana share one bucket (the noise swaps them).
IN_STATE_MAP = {
    "andhra pradesh": "aptg", "ap": "aptg", "telangana": "aptg", "tg": "aptg", "ts": "aptg",
    "arunachal pradesh": "ar", "ar": "ar", "assam": "as", "as": "as", "bihar": "br", "br": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "cg": "cg", "ct": "cg", "goa": "ga", "ga": "ga",
    "gujarat": "gj", "gj": "gj", "haryana": "hr", "hr": "hr", "himachal pradesh": "hp", "hp": "hp",
    "jharkhand": "jh", "jh": "jh", "karnataka": "ka", "ka": "ka", "kerala": "kl", "keralam": "kl", "kl": "kl",
    "madhya pradesh": "mp", "mp": "mp", "maharashtra": "mh", "mh": "mh", "manipur": "mn", "mn": "mn",
    "meghalaya": "ml", "ml": "ml", "mizoram": "mz", "mz": "mz", "nagaland": "nl", "nl": "nl",
    "odisha": "od", "orissa": "od", "od": "od", "or": "od", "punjab": "pb", "pb": "pb",
    "rajasthan": "rj", "rj": "rj", "sikkim": "sk", "sk": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "tn": "tn", "tripura": "tr", "tr": "tr", "uttar pradesh": "up", "up": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "uk": "uk", "ut": "uk", "west bengal": "wb", "wb": "wb",
    "delhi": "dl", "nct of delhi": "dl", "dl": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "jk": "jk", "ladakh": "la", "chandigarh": "ch", "ch": "ch", "puducherry": "py", "pondicherry": "py",
    "py": "py", "andaman and nicobar islands": "an", "an": "an", "lakshadweep": "ld",
    "dadra and nagar haveli and daman and diu": "dn", "dadra and nagar haveli": "dn", "daman and diu": "dn",
}

# France: every department and region (current + pre-2016 names) -> current region code
_FR = {
    "ara": ["auvergne rhone alpes", "auvergne", "rhone alpes", "ain", "allier", "ardeche", "cantal", "drome",
            "isere", "loire", "haute loire", "puy de dome", "rhone", "savoie", "haute savoie", "metropole de lyon"],
    "bfc": ["bourgogne franche comte", "bourgogne", "franche comte", "cote d or", "doubs", "jura", "nievre",
            "haute saone", "saone et loire", "yonne", "territoire de belfort"],
    "bre": ["bretagne", "cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "cvl": ["centre val de loire", "centre", "cher", "eure et loir", "indre", "indre et loire", "loir et cher",
            "loiret"],
    "cor": ["corse", "corse du sud", "haute corse"],
    "ges": ["grand est", "alsace", "lorraine", "champagne ardenne", "ardennes", "aube", "marne", "haute marne",
            "meurthe et moselle", "meuse", "moselle", "bas rhin", "haut rhin", "vosges"],
    "hdf": ["hauts de france", "nord pas de calais", "picardie", "aisne", "nord", "oise", "pas de calais",
            "somme"],
    "idf": ["ile de france", "paris", "seine et marne", "yvelines", "essonne", "hauts de seine",
            "seine saint denis", "val de marne", "val d oise"],
    "nor": ["normandie", "basse normandie", "haute normandie", "calvados", "eure", "manche", "orne",
            "seine maritime"],
    "naq": ["nouvelle aquitaine", "aquitaine", "limousin", "poitou charentes", "charente", "charente maritime",
            "correze", "creuse", "dordogne", "gironde", "landes", "lot et garonne", "pyrenees atlantiques",
            "deux sevres", "vienne", "haute vienne"],
    "occ": ["occitanie", "midi pyrenees", "languedoc roussillon", "ariege", "aude", "aveyron", "gard",
            "haute garonne", "gers", "herault", "lot", "lozere", "hautes pyrenees", "pyrenees orientales", "tarn",
            "tarn et garonne"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "pac": ["provence alpes cote d azur", "paca", "alpes de haute provence", "hautes alpes", "alpes maritimes",
            "bouches du rhone", "var", "vaucluse"],
    "gp": ["guadeloupe"], "mq": ["martinique"], "gf": ["guyane", "guyane francaise"],
    "re": ["la reunion", "reunion"], "yt": ["mayotte"],
}
FR_STATE_MAP = {name: code for code, names in _FR.items() for name in names}

# country label (lowercased) -> state table. Unknown countries simply get no state parsing (open set).
STATE_TABLES = {
    "us": US_STATE_MAP, "usa": US_STATE_MAP, "united states": US_STATE_MAP,
    "india": IN_STATE_MAP, "in": IN_STATE_MAP,
    "france": FR_STATE_MAP, "fr": FR_STATE_MAP,
}
IN_LIKE = {"india", "in"}

# France only: descriptors the noise injects into S2/S3 names (11-31x more frequent than in clean S1; audit 08).
# Not global: 'international' is ordinary name vocabulary in India/US (applying it there costs -0.0002).
FR_INJECTED = {"international", "developpement", "france"}
