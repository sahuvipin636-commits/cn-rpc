"""Stage 2a: turn free-text agent remarks (Hinglish, Kanglish, English, with typos) into tags.

Method: normalise, strip letters-only, snap to the nearest canonical phrase with fuzzy
matching, read tags from that phrase; extract dates, amounts and durations with regex.
Field remarks can have several parts separated by ';' and each part is matched separately.

Finding from Stage 2 analysis: every dial-remark phrase maps to exactly one disposition,
so remarks do NOT reveal mislogging. Their value is the detail a disposition drops:
new owner vs not known, long switched off, who answered, evening callback, where met.
"""
from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

# phrase -> "lang tag tag ..."   (lang: hi = Hinglish, kn = Kanglish, en = English)
DIAL_LEXICON = {
    # no answer
    "rnr": "en no_answer", "ringing no response": "en no_answer", "no answer": "en no_answer",
    "ring ho raha hai, koi pick nahi kar raha": "hi no_answer", "ring hua, uthaya nahi": "hi no_answer",
    "ring aaytu, yaaru pick maadilla": "kn no_answer",
    # rejected after ringing (avoidance)
    "busy bataya": "hi rejected", "# ring ke baad disconnect": "hi rejected", "call cut kar diya": "hi rejected",
    "call rejected": "en rejected", "disconnected after # rings": "en rejected",
    "call cut maadidru": "kn rejected", "busy antha barthide": "kn rejected",
    # network
    "not reachable": "en not_reachable", "out of coverage": "en not_reachable no_network",
    "network nahi hai": "hi not_reachable no_network", "network illa": "kn not_reachable no_network",
    "phone off": "en switched_off", "switch off": "en switched_off", "switched off": "en switched_off",
    "number switch off aa raha hai": "hi switched_off", "phone band hai": "hi switched_off",
    "switch off ide": "kn switched_off",
    "number band hai # mahine se": "hi switched_off long_off", "switched off for weeks": "en switched_off long_off",
    "number off ide eradu tingalinda": "kn switched_off long_off",
    "invalid number": "en invalid", "number does not exist": "en invalid", "does not exist message aaya": "hi invalid",
    "number exist nahi karta": "hi invalid", "number exist aagalla": "kn invalid",
    "language problem, local language speaker": "en language_barrier", "language barrier": "en language_barrier",
    "bhasha samajh nahi aayi": "hi language_barrier",
    # borrower spoke
    "hung up mid call": "en rpc hung_up", "beech mein phone kaat diya": "hi rpc hung_up",
    "baat karte karte call kaat diya": "hi rpc hung_up", "maatu madhyadalli call cut": "kn rpc hung_up",
    "shaam ko call back": "hi rpc callback evening", "call back in evening": "en rpc callback evening",
    "saayankala call back": "kn rpc callback evening", "baad mein call karne bola": "hi rpc callback",
    "asked to call back": "en rpc callback", "amele call maadi antharu": "kn rpc callback",
    "ptp #/#, amount #,#": "en rpc ptp", "ptp #/#": "en rpc ptp", "ptp #/# ka diya": "hi rpc ptp",
    "customer ne #/# ko pay karne ka bola": "hi rpc ptp", "customer promised #/#": "en rpc ptp",
    "#/# ge kodtini antharu": "kn rpc ptp", "ptp liya, customer cooperative": "hi rpc ptp",
    "customer ne ptp diya, follow up karna hai": "hi rpc ptp", "cust agreed to pay, ptp noted": "en rpc ptp",
    "rtp": "en rpc refused", "pay karne se mana kiya": "hi rpc refused", "refused to pay": "en rpc refused",
    "kodalla antharu": "kn rpc refused", "bola abhi paise nahi hai": "hi rpc refused no_funds",
    "says no funds": "en rpc refused no_funds", "duddu illa antharu": "kn rpc refused no_funds",
    "medical emergency": "en rpc hardship hardship_medical", "business band ho gaya": "hi rpc hardship hardship_business",
    "business shut": "en rpc hardship hardship_business", "job chali gayi bola": "hi rpc hardship hardship_job",
    "lost job": "en rpc hardship hardship_job", "kelsa hoytu antharu": "kn rpc hardship hardship_job",
    "bola payment kar diya, proof bhejega": "hi rpc claims_paid", "claims paid, will share proof": "en rpc claims_paid",
    "pay maadidini antharu, proof kalistaare": "kn rpc claims_paid",
    "charges pe dispute": "hi rpc dispute", "dispute on charges": "en rpc dispute",
    "dispute - modale pay maadidini antharu": "kn rpc dispute claims_paid",
    "dispute - bola pehle hi pay kar diya": "hi rpc dispute claims_paid",
    "dispute, claims already paid": "en rpc dispute claims_paid",
    # someone else answered
    "wife ne uthaya, bola bahar gaye hain": "hi third_party tp_spouse borrower_away",
    "father ne uthaya, message de denge": "hi third_party tp_parent message",
    "office se bola leave pe hai": "hi third_party tp_office borrower_away",
    "phone uthaya bhai ne": "hi third_party tp_sibling", "bhai ne uthaya": "hi third_party tp_sibling",
    "office colleague answered": "en third_party tp_office", "spouse answered": "en third_party tp_spouse",
    "brother answered, will pass message": "en third_party tp_sibling message",
    "office inda leave antharu": "kn third_party tp_office borrower_away",
    "avara thamma maataadidru": "kn third_party tp_sibling",
    "avara hendathi pick maadidru": "kn third_party tp_spouse",
    "spouse promised payment by #/#": "en third_party tp_spouse tp_ptp",
    "family member ne ptp diya #/#": "hi third_party tp_family tp_ptp",
    "wife ne bola #/# tak kar denge": "hi third_party tp_spouse tp_ptp",
    "hendathi #/# olage maadtaare antharu": "kn third_party tp_spouse tp_ptp",
    # stranger answered
    "galat number bola": "hi wrong_number", "wrong number": "en wrong_number",
    "wrong number antha helidru": "kn wrong_number",
    "customer ko nahi jaanta": "hi wrong_number not_known", "nobody by that name": "en wrong_number not_known",
    "ee hesarina yaaru illa antharu": "kn wrong_number not_known",
    "wrong number, is naam ka koi nahi": "hi wrong_number not_known",
    "bola mera naya number hai, kisi aur ka tha": "hi wrong_number new_owner",
    "hosa number antharu, customer gottilla": "kn wrong_number new_owner",
    "says new number, doesn't know customer": "en wrong_number new_owner",
}

FIELD_LEXICON = {
    "address sikkilla, tumba hudukide": "kn not_traceable", "address sikkilla": "kn not_traceable",
    "address nahi mila": "hi not_traceable", "is address pe ghar nahi mila": "hi not_traceable",
    "address trace nahi hua, kaafi dhunda": "hi not_traceable",
    "could not trace address after search": "en not_traceable", "address not traceable": "en not_traceable",
    "lock laga hai": "hi locked", "ghar band tha": "hi locked", "koi nahi mila, taala laga tha": "hi locked",
    "mane lock agide": "kn locked", "yaaru illa, beega haakide": "kn locked",
    "house locked": "en locked", "premises locked, nobody home": "en locked",
    "avara appa jothe maataadide": "kn met_family tp_parent",
    "father se baat hui, message de denge": "hi met_family tp_parent message",
    "met father, will pass message": "en met_family tp_parent message",
    "avara hendathi sikkidru, kelsakke hogidaare antharu": "kn met_family tp_spouse at_work",
    "wife mili, bola kaam pe gaye hain": "hi met_family tp_spouse at_work",
    "met spouse, customer at work": "en met_family tp_spouse at_work",
    "ghar pe beta tha, customer bahar": "hi met_family tp_child borrower_away",
    "customer ghar pe mila": "hi met_borrower at_home", "customer maneli sikkidru": "kn met_borrower at_home",
    "met customer at home": "en met_borrower at_home",
    "customer dukaan pe mila": "hi met_borrower at_shop", "customer angadi nalli sikkidru": "kn met_borrower at_shop",
    "met customer at shop": "en met_borrower at_shop",
    "customer se mila, baat hui": "hi met_borrower", "customer sikkidru, maataadide": "kn met_borrower",
    "met customer": "en met_borrower",
    "customer mila, bola #/# tak karega": "hi met_borrower ptp",
    "customer sikkidru, #/# olage kodtini andru": "kn met_borrower ptp",
    "met customer, promised by #/#": "en met_borrower ptp",
    "neighbour bola # mahine pehle ghar khali kar diya": "hi shifted", "padosi ne bola shift ho gaye": "hi shifted",
    "shift ho gaye, naya address pata nahi": "hi shifted", "pakkadavaru shift aagidaare antharu": "kn shifted",
    "mane khaali maadidaare": "kn shifted", "house vacated # months ago": "en shifted",
    "neighbour says shifted": "en shifted",
    "ee hesarina yaaru illa illi": "kn no_such_person", "is naam ka koi nahi rehta": "hi no_such_person",
    "ghar wale bole yeh insaan yahan nahi rehta": "hi no_such_person",
    "no such person at this address": "en no_such_person",
    "customer ne cash diya, receipt di": "hi met_borrower cash", "# cash collect kiya": "hi met_borrower cash",
    "# cash collect": "hi met_borrower cash", "cash collect maadide, receipt kottide": "kn met_borrower cash",
    "collected # cash": "en met_borrower cash", "collected cash, receipt issued": "en met_borrower cash",
}

# Landmark-based directions written by field agents ("real house is behind the temple")
CORRECTION_RE = re.compile(
    r"address galat|address tappu|asli ghar|nija mane|blue gate|saamne wali gali|eduru road|hattira ide|"
    r"hinde ide|peeche hai|ke paas hai|se right side|gali aage|cross munde|"
    r"actual house|address wrong|house near|road opposite|lanes? ahead|right side of")

DIAL_TAGS = sorted({t for v in DIAL_LEXICON.values() for t in v.split()[1:]})
FIELD_TAGS = sorted({t for v in FIELD_LEXICON.values() for t in v.split()[1:]} | {"address_correction"})
MATCH_CUTOFF = 80


def normalise(text: str) -> str:
    t = re.sub(r"^\s*bot:\s*", "", str(text).lower())
    return re.sub(r"\d[\d,]*", "#", t).strip()


def _squash(text: str) -> str:
    return re.sub(r"[^a-z#]", "", text)


def _matcher(lexicon: dict):
    keys = {_squash(k): k for k in lexicon}
    choices = list(keys)

    def match(part: str):
        if not part:
            return None, 0.0
        hit = process.extractOne(_squash(part), choices, scorer=fuzz.ratio, score_cutoff=MATCH_CUTOFF)
        return (keys[hit[0]], hit[1]) if hit else (None, 0.0)
    return match


def _extract(raw: str) -> dict:
    raw_l = str(raw).lower()
    out = {"ptp_day": None, "ptp_month": None, "amount": None, "months": None}
    d = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})", raw_l)
    if d:
        out["ptp_day"], out["ptp_month"] = int(d.group(1)), int(d.group(2))
    amt = re.search(r"(\d{1,3}(?:,\d{2,3})+|\d{3,})", re.sub(r"\d{1,2}\s*/\s*\d{1,2}", " ", raw_l))
    if amt:
        out["amount"] = float(amt.group(1).replace(",", ""))
    m = re.search(r"(\d+)\s*(?:mahine|maihne|mhaine|amhine|months?)", raw_l)
    if m:
        out["months"] = float(m.group(1))
    elif "eradu tingal" in raw_l or "eradu tinga" in raw_l:
        out["months"] = 2.0
    elif "weeks" in raw_l or "weesk" in raw_l:
        out["months"] = 0.5
    return out


def _tag_frame(ids: pd.Series, remarks: pd.Series, lexicon: dict, tags: list, multipart: bool,
               id_name: str) -> pd.DataFrame:
    match = _matcher(lexicon)
    cache = {}
    rows = []
    for rid, raw in zip(ids, remarks):
        key = str(raw)
        if key not in cache:
            parts = [p.strip() for p in normalise(key).split(";")] if multipart else [normalise(key)]
            found, scores, lang = set(), [], None
            canon_parts = []
            for p in parts:
                canon, score = match(p)
                if canon:
                    spec = lexicon[canon].split()
                    lang = lang or spec[0]
                    found.update(spec[1:])
                    scores.append(score)
                    canon_parts.append(canon)
                elif multipart and CORRECTION_RE.search(p):
                    found.add("address_correction")
                    scores.append(100.0)
            cache[key] = {"canon": " ; ".join(canon_parts) or None, "match_score": min(scores) if scores else 0.0,
                          "lang": lang, "is_bot": key.lower().startswith("bot:"),
                          **{f"tag_{t}": t in found for t in tags}, **_extract(key)}
        rows.append({id_name: rid, **cache[key]})
    return pd.DataFrame(rows)


def tag_dial_remarks(att: pd.DataFrame) -> pd.DataFrame:
    return _tag_frame(att.attempt_id, att.remark, DIAL_LEXICON, DIAL_TAGS, False, "attempt_id")


def tag_field_remarks(visits: pd.DataFrame) -> pd.DataFrame:
    return _tag_frame(visits.visit_id, visits.remark.fillna(""), FIELD_LEXICON, FIELD_TAGS, True, "visit_id")
