# =============================================================================
# FILE: backend/scripts/populate_gazetteer.py
#
# PURPOSE:
#   Builds the Chennai coordinate lookup table (JSON file) by querying
#   the Nominatim geocoding API once per location.
#
# WHY THIS EXISTS AS A SEPARATE ONE-TIME SCRIPT:
#   At runtime, the Geocoder must be fast and reliable. Calling a network
#   API during a live disaster demo introduces latency, rate limit risk,
#   and internet dependency. This script runs once, builds a complete local
#   lookup table, and the system never calls Nominatim again.
#
# CRASH-SAFE RESUME (Bug 15 fix):
#   If the script crashes at location 340 of 500, restarting it reads the
#   existing JSON and skips already-resolved locations. You never lose work.
#
# RATE LIMITING:
#   Nominatim's terms of service require maximum 1 request per second.
#   We enforce a 1.1-second delay between calls to stay safely within limits.
#   Violation can result in IP banning.
#
# OUTPUT:
#   backend/data/gazetteers/chennai_gazetteer.json
#
# RUN: Once only during project setup. Never again at runtime.
# =============================================================================

import os
import json
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# =============================================================================
# SECTION 1 — PATH CONFIGURATION
# =============================================================================

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAZETTEER_PATH= os.path.join(ROOT_DIR, "backend", "data", "gazetteers", "chennai_gazetteer.json")

# =============================================================================
# SECTION 2 — MASTER LOCATION LIST
#
# This is the complete list of Chennai locations we want coordinates for.
# Keys are the lookup strings (what BERT NER might return).
# We append "Chennai" to every query for unambiguous Nominatim results —
# "Velachery" alone might match a place in another state.
#
# ALIAS PATTERN:
#   We include both the full official name AND common abbreviations/nicknames.
#   Both will map to the same coordinates in the final JSON.
#   Example: "gh" and "government general hospital" → same lat/lng
#   This is the domain-specific knowledge advantage over generic geocoders.
# =============================================================================

# Format: "lookup_key": "nominatim search query"
# lookup_key   = what gets stored in the JSON (what BERT NER might return)
# search query = what we send to Nominatim (includes Chennai for precision)

CHENNAI_LOCATIONS = {

    # ── MAJOR LANDMARKS ──────────────────────────────────────────────
    "marina beach"                : "Marina Beach Chennai",
    "kapaleeshwarar temple"       : "Kapaleeshwarar Temple Mylapore Chennai",
    "fort st george"              : "Fort St George Chennai",
    "valluvar kottam"             : "Valluvar Kottam Chennai",
    "government museum"           : "Government Museum Egmore Chennai",
    "santhome cathedral"          : "Santhome Cathedral Chennai",
    "elliot's beach"              : "Elliot Beach Besant Nagar Chennai",
    "besant nagar beach"          : "Elliot Beach Besant Nagar Chennai",
    "anna square"                 : "Anna Square Marina Chennai",
    "ripon building"              : "Ripon Building Chennai",
    "central station"             : "Chennai Central Railway Station",
    "chennai central"             : "Chennai Central Railway Station",
    "chennai egmore"              : "Chennai Egmore Railway Station",
    "egmore station"              : "Chennai Egmore Railway Station",
    "cmbt"                        : "CMBT Koyambedu Bus Terminus Chennai",
    "koyambedu bus stand"         : "CMBT Koyambedu Bus Terminus Chennai",
    "broadway"                    : "Broadway Bus Terminus Chennai",
    "spencer plaza"               : "Spencer Plaza Chennai",
    "express avenue"              : "Express Avenue Mall Chennai",
    "chennai airport"             : "Chennai International Airport",
    "meenambakkam airport"        : "Chennai International Airport",

    # ── HOSPITALS (critical during disasters) ────────────────────────
    "government general hospital" : "Government General Hospital Chennai",
    "gg hospital"                 : "Government General Hospital Chennai",
    "ggh"                         : "Government General Hospital Chennai",
    "gh"                          : "Government General Hospital Chennai",
    "stanley hospital"            : "Stanley Medical College Hospital Chennai",
    "stanley"                     : "Stanley Medical College Hospital Chennai",
    "rajiv gandhi hospital"       : "Rajiv Gandhi Government General Hospital Chennai",
    "kilpauk medical college"     : "Kilpauk Medical College Hospital Chennai",
    "kmc hospital"                : "Kilpauk Medical College Hospital Chennai",
    "apollo hospital greams road" : "Apollo Hospital Greams Road Chennai",
    "apollo greams road"          : "Apollo Hospital Greams Road Chennai",
    "apollo hospital"             : "Apollo Hospital Greams Road Chennai",
    "apollo"                      : "Apollo Hospital Greams Road Chennai",
    "fortis malar"                : "Fortis Malar Hospital Adyar Chennai",
    "miot hospital"               : "MIOT International Chennai",
    "sri ramachandra hospital"    : "Sri Ramachandra Medical College Chennai",
    "kauvery hospital"            : "Kauvery Hospital Chennai",
    "vijaya hospital"             : "Vijaya Hospital Chennai",
    "billroth hospital"           : "Billroth Hospital Chennai",
    "mgm healthcare"              : "MGM Healthcare Chennai",
    "institute of mental health"  : "Institute of Mental Health Chennai",
    "imh"                         : "Institute of Mental Health Chennai",
    "omr apollo"                  : "Apollo Hospital OMR Chennai",

    # ── RAILWAY STATIONS ─────────────────────────────────────────────
    "tambaram station"            : "Tambaram Railway Station Chennai",
    "tambaram"                    : "Tambaram Chennai",
    "perambur station"            : "Perambur Railway Station Chennai",
    "perambur"                    : "Perambur Chennai",
    "mambalam station"            : "Mambalam Railway Station Chennai",
    "kodambakkam station"         : "Kodambakkam Railway Station Chennai",
    "saidapet station"            : "Saidapet Railway Station Chennai",
    "guindy station"              : "Guindy Railway Station Chennai",
    "st thomas mount station"     : "St Thomas Mount Railway Station Chennai",
    "pallavaram station"          : "Pallavaram Railway Station Chennai",
    "chromepet station"           : "Chromepet Railway Station Chennai",
    "avadi station"               : "Avadi Railway Station Chennai",
    "ambattur station"            : "Ambattur Railway Station Chennai",
    "basin bridge"                : "Basin Bridge Railway Station Chennai",
    "korukkupet"                  : "Korukkupet Railway Station Chennai",
    "tiruvallur"                  : "Tiruvallur Railway Station",
    "paranur"                     : "Paranur Railway Station Chennai",

    # ── MAJOR NEIGHBOURHOODS AND LOCALITIES ──────────────────────────
    "adyar"                       : "Adyar Chennai",
    "velachery"                   : "Velachery Chennai",
    "t nagar"                     : "T.Nagar Chennai",
    "t.nagar"                     : "T.Nagar Chennai",
    "anna nagar"                  : "Anna Nagar Chennai",
    "nungambakkam"                : "Nungambakkam Chennai",
    "mylapore"                    : "Mylapore Chennai",
    "kodambakkam"                 : "Kodambakkam Chennai",
    "guindy"                      : "Guindy Chennai",
    "thiruvanmiyur"               : "Thiruvanmiyur Chennai",
    "besant nagar"                : "Besant Nagar Chennai",
    "alwarpet"                    : "Alwarpet Chennai",
    "kilpauk"                     : "Kilpauk Chennai",
    "chetpet"                     : "Chetpet Chennai",
    "egmore"                      : "Egmore Chennai",
    "purasaiwalkam"               : "Purasaiwalkam Chennai",
    "perambur"                    : "Perambur Chennai",
    "royapuram"                   : "Royapuram Chennai",
    "tondiarpet"                  : "Tondiarpet Chennai",
    "madhavaram"                  : "Madhavaram Chennai",
    "kolathur"                    : "Kolathur Chennai",
    "villivakkam"                 : "Villivakkam Chennai",
    "ambattur"                    : "Ambattur Chennai",
    "avadi"                       : "Avadi Chennai",
    "porur"                       : "Porur Chennai",
    "valasaravakkam"              : "Valasaravakkam Chennai",
    "ramapuram"                   : "Ramapuram Chennai",
    "virugambakkam"               : "Virugambakkam Chennai",
    "vadapalani"                  : "Vadapalani Chennai",
    "ashok nagar"                 : "Ashok Nagar Chennai",
    "kk nagar"                    : "KK Nagar Chennai",
    "arumbakkam"                  : "Arumbakkam Chennai",
    "saligramam"                  : "Saligramam Chennai",
    "koyambedu"                   : "Koyambedu Chennai",
    "chromepet"                   : "Chromepet Chennai",
    "pallavaram"                  : "Pallavaram Chennai",
    "perungudi"                   : "Perungudi Chennai",
    "sholinganallur"              : "Sholinganallur Chennai",
    "siruseri"                    : "Siruseri Chennai",
    "thoraipakkam"                : "Thoraipakkam Chennai",
    "perungalathur"               : "Perungalathur Chennai",
    "selaiyur"                    : "Selaiyur Chennai",
    "medavakkam"                  : "Medavakkam Chennai",
    "madipakkam"                  : "Madipakkam Chennai",
    "nanganallur"                 : "Nanganallur Chennai",
    "ullagaram"                   : "Ullagaram Chennai",
    "pammal"                      : "Pammal Chennai",
    "alandur"                     : "Alandur Chennai",
    "st thomas mount"             : "St Thomas Mount Chennai",
    "nandanam"                    : "Nandanam Chennai",
    "saidapet"                    : "Saidapet Chennai",
    "kotturpuram"                 : "Kotturpuram Chennai",
    "mandaveli"                   : "Mandaveli Chennai",
    "r a puram"                   : "R.A.Puram Chennai",
    "poes garden"                 : "Poes Garden Chennai",
    "boat club"                   : "Boat Club Road Chennai",
    "abhiramapuram"               : "Abhiramapuram Chennai",
    "sowcarpet"                   : "Sowcarpet Chennai",
    "parrys"                      : "Parrys Corner Chennai",
    "parry's corner"              : "Parrys Corner Chennai",
    "washermanpet"                : "Washermanpet Chennai",
    "periamet"                    : "Periamet Chennai",
    "vepery"                      : "Vepery Chennai",
    "choolai"                     : "Choolai Chennai",
    "otteri"                      : "Otteri Chennai",
    "aminjikarai"                 : "Aminjikarai Chennai",
    "cit nagar"                   : "CIT Nagar Chennai",
    "west mambalam"               : "West Mambalam Chennai",
    "east tambaram"               : "East Tambaram Chennai",
    "chromepet"                   : "Chromepet Chennai",
    "mudichur"                    : "Mudichur Chennai",
    "perumbakkam"                 : "Perumbakkam Chennai",
    "kovilambakkam"               : "Kovilambakkam Chennai",
    "karapakkam"                  : "Karapakkam Chennai",
    "navalur"                     : "Navalur Chennai",
    "kelambakkam"                 : "Kelambakkam Chennai",

    # ── BRIDGES AND WATERWAYS (critical for flood reports) ───────────
    "adyar river"                 : "Adyar River Chennai",
    "cooum river"                 : "Cooum River Chennai",
    "cooum"                       : "Cooum River Chennai",
    "buckingham canal"            : "Buckingham Canal Chennai",
    "adyar bridge"                : "Adyar Bridge Chennai",
    "velachery bridge"            : "Velachery Main Road Bridge Chennai",
    "kotturpuram bridge"          : "Kotturpuram Bridge Chennai",
    "napier bridge"               : "Napier Bridge Chennai",
    "captain cotton bridge"       : "Captain Cotton Bridge Chennai",
    "ennore creek"                : "Ennore Creek Chennai",
    "adyar estuary"               : "Adyar Estuary Chennai",
    "red hills lake"              : "Red Hills Reservoir Chennai",
    "puzhal lake"                 : "Puzhal Lake Chennai",
    "chembarambakkam lake"        : "Chembarambakkam Lake Chennai",
    "chembarambakkam"             : "Chembarambakkam Lake Chennai",
    "porur lake"                  : "Porur Lake Chennai",
    "madambakkam lake"            : "Madambakkam Lake Chennai",
    "chetpet lake"                : "Chetpet Lake Chennai",
    "velachery lake"              : "Velachery Lake Chennai",
    "pallikaranai marsh"          : "Pallikaranai Wetland Chennai",
    "sholinganallur marsh"        : "Sholinganallur Lake Chennai",

    # ── MAJOR ROADS AND HIGHWAYS ──────────────────────────────────────
    "omr"                         : "Old Mahabalipuram Road Chennai",
    "old mahabalipuram road"      : "Old Mahabalipuram Road Chennai",
    "ecr"                         : "East Coast Road Chennai",
    "east coast road"             : "East Coast Road Chennai",
    "gst road"                    : "Grand Southern Trunk Road Chennai",
    "nh44"                        : "National Highway 44 Chennai",
    "anna salai"                  : "Anna Salai Chennai",
    "mount road"                  : "Mount Road Chennai",
    "poonamallee high road"       : "Poonamallee High Road Chennai",
    "inner ring road"             : "Inner Ring Road Chennai",
    "outer ring road"             : "Outer Ring Road Chennai",
    "rajiv gandhi salai"          : "Rajiv Gandhi Salai Chennai",
    "100 feet road"               : "100 Feet Road Vadapalani Chennai",
    "velachery main road"         : "Velachery Main Road Chennai",

    # ── BUS STANDS ────────────────────────────────────────────────────
    "koyambedu bus stand"         : "CMBT Koyambedu Chennai",
    "broadway bus stand"          : "Broadway Bus Terminus Chennai",
    "tambaram bus stand"          : "Tambaram Bus Stand Chennai",
    "guindy bus stand"            : "Guindy Bus Stand Chennai",
    "vadapalani bus stand"        : "Vadapalani Bus Stand Chennai",
    "madhavaram bus stand"        : "Madhavaram Bus Stand Chennai",
    "thiruvanmiyur bus stand"     : "Thiruvanmiyur Bus Stand Chennai",

    # ── SCHOOLS AND COLLEGES (used as shelters during disasters) ─────
    "iit madras"                  : "IIT Madras Chennai",
    "anna university"             : "Anna University Chennai",
    "loyola college"              : "Loyola College Chennai",
    "mcc"                         : "Madras Christian College Chennai",
    "madras christian college"    : "Madras Christian College Chennai",
    "presidency college"          : "Presidency College Chennai",
    "womens christian college"    : "Women's Christian College Chennai",
    "dg vaishnav"                 : "DG Vaishnav College Chennai",
    "ethiraj college"             : "Ethiraj College Chennai",
    "srm university"              : "SRM University Chennai",
    "vit chennai"                 : "VIT University Chennai",
    "madras university"           : "University of Madras Chennai",

    # ── ADMINISTRATIVE AREAS AND ZONES ───────────────────────────────
    "chennai north"               : "North Chennai",
    "chennai south"               : "South Chennai",
    "chennai central"             : "Central Chennai",
    "north chennai"               : "North Chennai",
    "south chennai"               : "South Chennai",
    "chennai"                     : "Chennai Tamil Nadu India",
    "kancheepuram"                : "Kancheepuram Tamil Nadu",
    "tiruvallur district"         : "Tiruvallur District Tamil Nadu",
}


# =============================================================================
# SECTION 3 — GAZETTEER POPULATION LOGIC
# =============================================================================

def load_existing_gazetteer() -> dict:
    """
    Loads the existing gazetteer JSON if it exists.
    Returns empty dict if file doesn't exist yet.

    This is the crash-safe resume mechanism (Bug 15 fix).
    If the script crashed partway through, we reload what we already have
    and skip those locations on restart.
    """
    if os.path.exists(GAZETTEER_PATH):
        with open(GAZETTEER_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"   📂 Loaded existing gazetteer: {len(existing)} locations already resolved")
        return existing
    return {}


def save_gazetteer(gazetteer: dict):
    """
    Saves the current gazetteer dict to JSON file.
    Called after EVERY successful lookup — this is what makes the script
    crash-safe. If it crashes at location 340, we already have 339 saved.
    """
    os.makedirs(os.path.dirname(GAZETTEER_PATH), exist_ok=True)
    with open(GAZETTEER_PATH, "w", encoding="utf-8") as f:
        json.dump(gazetteer, f, indent=2, ensure_ascii=False)


def populate():
    """
    Main function. Queries Nominatim for every location in CHENNAI_LOCATIONS
    and builds the gazetteer JSON.

    The JSON structure:
    {
      "velachery": {
        "lat": 12.9783,
        "lng": 80.2209,
        "full_name": "Velachery, Chennai, Tamil Nadu, India"
      },
      ...
    }

    Keys are always lowercase (the lookup_key from CHENNAI_LOCATIONS).
    This enables case-insensitive lookup in geocoder.py.
    """
    print("=" * 60)
    print("CrisisLens — Chennai Gazetteer Population Script")
    print("=" * 60)
    print(f"   Total locations to resolve: {len(CHENNAI_LOCATIONS)}")
    print(f"   Output: {GAZETTEER_PATH}")
    print(f"   Estimated time: ~{len(CHENNAI_LOCATIONS) * 1.1 / 60:.0f} minutes")
    print()

    # Load any existing progress (Bug 15 fix — crash safe resume)
    gazetteer = load_existing_gazetteer()

    # Initialise Nominatim geocoder
    # user_agent must be a unique string identifying your app
    # Nominatim requires this to track usage and enforce rate limits
    geolocator = Nominatim(user_agent="crisislens_gazetteer_builder_v1")

    resolved   = 0
    skipped    = 0
    failed     = 0
    failed_list= []

    total = len(CHENNAI_LOCATIONS)

    for i, (lookup_key, search_query) in enumerate(CHENNAI_LOCATIONS.items(), 1):

        # Skip if already resolved (crash-safe resume)
        if lookup_key in gazetteer:
            skipped += 1
            continue

        print(f"   [{i:>3}/{total}] Querying: '{search_query}'", end=" → ", flush=True)

        try:
            # Nominatim geocoding call
            # timeout=10: wait up to 10 seconds before giving up
            location = geolocator.geocode(search_query, timeout=10) # type: ignore

            if location:
                # Store with lowercase key (enables case-insensitive lookup later)
                gazetteer[lookup_key.lower()] = {
                    "lat"      : round(location.latitude,  6), # type: ignore
                    "lng"      : round(location.longitude, 6), # type: ignore
                    "full_name": location.address # type: ignore
                }
                print(f"{location.latitude:.4f}, {location.longitude:.4f} ✓") # type: ignore
                resolved += 1

                # Save after every success (Bug 15 fix — incremental persistence)
                save_gazetteer(gazetteer)

            else:
                print("NOT FOUND ⚠️")
                failed += 1
                failed_list.append((lookup_key, search_query))

        except GeocoderTimedOut:
            print("TIMEOUT ⚠️ — will retry next run")
            failed += 1
            failed_list.append((lookup_key, f"{search_query} [timeout]"))

        except GeocoderServiceError as e:
            print(f"SERVICE ERROR: {e} ⚠️")
            failed += 1
            failed_list.append((lookup_key, f"{search_query} [service error]"))

        # Rate limiting — Nominatim ToS requires max 1 request per second
        # We use 1.1s to be safely within limits
        time.sleep(1.1)

    # Final save
    save_gazetteer(gazetteer)

    # ── Summary Report ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("✅ Gazetteer Population Complete — Summary")
    print("=" * 60)
    print(f"   Resolved this run:  {resolved:>4}")
    print(f"   Already existed:    {skipped:>4}")
    print(f"   Failed to resolve:  {failed:>4}")
    print(f"   Total in gazetteer: {len(gazetteer):>4}")
    print(f"   Saved to: {GAZETTEER_PATH}")

    if failed_list:
        print(f"\n   ⚠️  Failed locations (add manually or retry):")
        for key, query in failed_list:
            print(f"      '{key}': '{query}'")
        print(f"\n   For failed locations, you can add coordinates manually:")
        print(f"   1. Search the location on Google Maps")
        print(f"   2. Right-click → copy coordinates")
        print(f"   3. Add to {GAZETTEER_PATH}:")
        print(f'      "location_name": {{"lat": 12.XXXX, "lng": 80.XXXX, "full_name": "..."}}')

    print("=" * 60)
    print("   Next step: run geocoder.py is now ready to use.")
    print("   geocoder.py loads this file at startup — no further setup needed.")
    print("=" * 60)


if __name__ == "__main__":
    populate()