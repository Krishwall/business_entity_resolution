import re
import unicodedata
import pandas as pd

LEGAL_SUFFIXES = {
    r"\b(corporation|corp|incorporated|inc|limited|ltd|llc|llp|co|company)\b": "",
    r"\b(pvt\s*ltd|private\s*limited|pvt|proprietorship|prop)\b": "",
    r"\b(societe\s*anonyme|sa|sas|sasu|sarl|sci|snc|eurl)\b": "",
}

ADDRESS_MAPPINGS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bflr?\b": "floor",
    r"\bopp\b": "opposite",
    r"\bnear\b": "near",
    r"\bbvd\b": "boulevard",
    r"\br\b": "rue",
}


def normalize_ascii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ASCII", "ignore")
        .decode("utf-8")
        .lower()
        .strip()
    )


def clean_name(name: str) -> str:
    name = normalize_ascii(name)
    name = re.sub(r"&", " and ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    for pattern, repl in LEGAL_SUFFIXES.items():
        name = re.sub(pattern, repl, name)
    return re.sub(r"\s+", " ", name).strip()


def clean_address(addr: str) -> str:
    addr = normalize_ascii(addr)
    addr = re.sub(r"[^\w\s]", " ", addr)
    for pattern, repl in ADDRESS_MAPPINGS.items():
        addr = re.sub(pattern, repl, addr)
    return re.sub(r"\s+", " ", addr).strip()


def extract_numbers(addr: str) -> set[str]:
    if not addr:
        return set()
    return set(re.findall(r"\b\d{2,}\b", addr))


def load_source_table(
    filepath: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Loads TSV using low-memory types and returns fast lookup dicts:

    entity_id -> clean_name, clean_address, country, numbers
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
        keep_default_na=False,
    )

    names = {}
    addrs = {}
    countries = {}
    nums = {}

    for _, row in df.iterrows():
        eid = row["entity_id"]
        c_name = clean_name(row["business_name"])
        c_addr = clean_address(row["business_address"])
        names[eid] = c_name
        addrs[eid] = c_addr
        countries[eid] = row["country"].strip().upper() or "UNKNOWN"
        nums[eid] = extract_numbers(c_addr)

    return names, addrs, countries, nums