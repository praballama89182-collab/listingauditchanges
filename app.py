import io
import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

st.set_page_config(page_title="Catalog Refinery", page_icon="\U0001F4E6", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
:root{
    --navy:#131A22; --navy-2:#1E2733; --amber:#E8971A; --amber-dark:#8A5A0C;
    --paper:#F6F5F1; --ink:#1C1F26; --muted:#6B7280; --line:#E3E1D8;
    --success:#0F7B5D; --success-bg:#E4F3EC; --warn:#B8860B; --warn-bg:#FBF1DC;
    --danger:#C4432B; --danger-bg:#FBEAE5;
}
.stApp { background: var(--paper); font-family:'Inter',sans-serif; }
h1, h2, h3 { font-family:'Space Grotesk',sans-serif !important; color: var(--navy) !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1320px; }

.crHeader{
    background:var(--navy); color:#fff; padding:20px 28px; border-radius:12px;
    border-bottom:3px solid var(--amber); display:flex; align-items:center; gap:14px;
    margin-bottom:22px;
}
.crCrate{
    width:38px; height:38px; border:2px solid var(--amber); border-radius:6px;
    display:flex; align-items:center; justify-content:center; font-family:'Space Grotesk',sans-serif;
    font-weight:700; color:var(--amber); font-size:15px; flex-shrink:0;
}
.crHeader h1{ color:#fff !important; font-size:21px; margin:0; }
.crHeader p{ margin:2px 0 0; font-size:12.5px; color:#B8C0CC; }

.crSection{
    font-family:'Space Grotesk',sans-serif; font-size:12px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.8px; color:var(--navy);
    display:flex; align-items:center; gap:8px; margin:0 0 12px;
}
.crSection::before{ content:""; width:8px; height:8px; background:var(--amber); border-radius:2px; }

.crBadge { display:inline-block; padding:2px 10px; border-radius:10px; font-size:11px;
    font-family:'IBM Plex Mono',monospace; font-weight:600; }
.crGood { background:var(--success-bg); color:var(--success); }
.crWarn { background:var(--warn-bg); color:var(--warn); }
.crBad  { background:var(--danger-bg); color:var(--danger); }
.crGradeCircle { display:inline-flex; align-items:center; justify-content:center;
    width:44px; height:44px; border-radius:50%; background:var(--amber); color:#221400;
    font-weight:700; font-size:18px; font-family:'Space Grotesk',sans-serif; flex-shrink:0; }
.crGradeCard{
    background:var(--navy); border-radius:10px; padding:14px 18px; margin-top:10px;
    display:flex; align-items:center; gap:14px;
}
.crGradeCard .crMsg{ color:#B8C0CC; font-size:13px; }
.crGradeCard .crMsg b{ color:#fff; display:block; font-size:14px; font-family:'Space Grotesk',sans-serif; }

div[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:10px !important; }
.stButton > button[kind="primary"]{
    background:var(--amber); border-color:var(--amber); color:#221400; font-weight:700;
    font-family:'Space Grotesk',sans-serif;
}
.stButton > button[kind="primary"]:hover{ background:var(--amber-dark); border-color:var(--amber-dark); color:#fff; }
.stTabs [data-baseweb="tab"]{ font-family:'Space Grotesk',sans-serif; font-weight:600; }
</style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="crHeader">
        <div class="crCrate">CR</div>
        <div>
            <h1>Catalog Refinery</h1>
            <p>Amazon listing rewrite, compliance check, and a 5-image set generator — from raw product content and a raw photo.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CATEGORY_LEGACY_TITLE_MAX = {
    "general": 200, "apparel": 125, "electronics": 150,
    "babypet": 80, "supplement": 200, "media": 500,
}
CATEGORY_LABELS = {
    "general": "General / Standard",
    "apparel": "Apparel & Jewelry",
    "electronics": "Consumer Electronics",
    "babypet": "Baby / Pet Supplies",
    "supplement": "Supplements / Grocery / Regulated",
    "media": "Media (Books, Music, Video, Software)",
}
THEME_LABELS = {
    "auto": "Auto-detect from brand/category/copy",
    "baby_swim": "Baby + swim / water",
    "baby": "Baby & nursery",
    "swim": "Swim & water",
    "outdoor": "Outdoor & travel",
    "kitchen": "Kitchen & home",
    "fitness": "Fitness & sport",
    "tech": "Tech & electronics",
    "beauty": "Beauty & wellness",
    "general": "General / minimal",
}
PERSONA_LABELS = {
    "auto": "Auto-detect from brand/category/copy",
    "baby": "Baby",
    "man": "Man",
    "woman": "Woman",
    "adult": "Adult (neutral)",
    "none": "No figure",
}


def byte_len(s):
    return len(s.encode("utf-8"))


def limits_for(category, title_mode, bullet_mode):
    new_rule_eligible = title_mode == "new" and category != "media"
    title_max = 75 if new_rule_eligible else CATEGORY_LEGACY_TITLE_MAX[category]
    highlight_max = 125 if new_rule_eligible else 0
    return {
        "title_max": title_max,
        "highlight_max": highlight_max,
        "bullet_max": int(bullet_mode),
        "description_max": 2000,
        "backend_max_bytes": 249,
    }


def gauge_ratio(count, max_val):
    return count / max_val if max_val else 0


def gauge_badge(count, max_val, unit):
    ratio = gauge_ratio(count, max_val)
    cls = "crBad" if ratio > 1 else ("crWarn" if ratio > 0.9 else "crGood")
    return f'<span class="crBadge {cls}">{count}/{max_val} {unit}</span>'


def compute_grade(field_ratios):
    over = sum(1 for r in field_ratios if r > 1)
    warn = sum(1 for r in field_ratios if 0.9 < r <= 1)
    if over:
        grade = "D" if over > 1 else "C"
        msg = f"{over} field(s) are over the limit and will be truncated or de-indexed."
    elif warn:
        grade = "B"
        msg = f"{warn} field(s) are close to the limit — worth a tighter edit."
    else:
        grade = "A"
        msg = "All fields are within Amazon's field limits."
    return grade, msg


# ---------------------------------------------------------------------------
# Rule-based text generation (no external API — everything below is
# deterministic string processing over what the person typed in)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "for", "with", "and", "or", "of", "in", "on", "to", "is",
    "are", "this", "that", "these", "those", "it", "its", "your", "our", "you",
    "we", "at", "by", "from", "as", "be", "was", "were", "will", "can", "has",
    "have", "had", "not", "no", "but", "so", "than", "then", "into", "up",
    "out", "if", "all", "each", "per",
}

# Amazon explicitly disallows these categories in backend search terms and
# discourages them platform-wide: subjective/superlative claims, award or
# ranking claims ("Amazon's Choice", "Best Seller"), promotional/time-sensitive
# language, and unverifiable guarantees. Source: Amazon Seller Central search
# terms guidance and 2026 backend-keyword policy writeups.
MARKETING_WORDS = {
    "best", "bestseller", "bestsellers", "cheap", "cheapest", "affordable",
    "amazing", "awesome", "good", "great", "top", "toprated", "rated",
    "guarantee", "guaranteed", "warranty", "premium", "new", "sale", "sales",
    "discount", "limited", "now", "perfect", "ideal", "ultimate", "award",
    "awards", "winning", "winner", "choice", "certified", "approved", "fda",
    "doctor", "clinically", "proven", "miracle", "revolutionary", "exclusive",
    "luxury", "superior", "number1", "no1",
}

RISKY_TERMS = re.compile(
    r"\b(cure|cures|cured|treat|treats|treating|treatment|disease|diagnos\w*|"
    r"prevent\w*|cancer|diabetes|arthritis|tumou?r|heart disease|blood pressure|"
    r"inflammation|infection)\b",
    re.IGNORECASE,
)

MARKETING_TERMS = re.compile(
    r"\b(" + "|".join(sorted(MARKETING_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

MATERIAL_KEYWORDS = [
    "cotton", "nylon", "silicone", "stainless steel", "polyester", "leather",
    "bamboo", "plastic", "wood", "glass", "ceramic", "rubber", "wool", "linen",
    "ABS plastic", "aluminum", "ripstop", "neoprene", "spandex", "fleece",
]
CARE_KEYWORDS = [
    "machine washable", "hand wash", "wipe clean", "dishwasher safe",
    "air dry", "spot clean", "reusable", "disposable",
]

THEME_TAGLINES = {
    "baby_swim": "Made for splashing and everyday play",
    "baby": "Designed for cozy, everyday comfort",
    "swim": "Built for pool days and beyond",
    "outdoor": "Ready for the trail and beyond",
    "kitchen": "A staple for everyday cooking",
    "fitness": "Built to keep up with you",
    "tech": "Designed to fit your everyday routine",
    "beauty": "A simple step in your daily routine",
    "general": "Made for real, everyday use",
}


def strip_separators(text):
    text = re.sub(r"[\-\u2013\u2014|]+", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def split_lines(text):
    return [l.strip(" \t\u2022-\u2013\u2014*") for l in (text or "").splitlines() if l.strip()]


def dedupe_words_preserve_order(words):
    seen, out = set(), []
    for w in words:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def truncate_at_word(text, max_len):
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut.rstrip(",.;:- ") or text[:max_len]


def truncate_bytes_at_word(text, max_bytes):
    text = (text or "").strip()
    if byte_len(text) <= max_bytes:
        return text
    words, out, total = text.split(" "), [], 0
    for w in words:
        wb = byte_len(w) + (1 if out else 0)
        if total + wb > max_bytes:
            break
        out.append(w)
        total += wb
    return " ".join(out)


def apply_regulated_filter(sentences, regulated):
    if not regulated:
        return sentences, False
    kept, flagged = [], False
    for s in sentences:
        if RISKY_TERMS.search(s):
            flagged = True
        else:
            kept.append(s)
    return kept, flagged


def apply_marketing_filter(sentences):
    kept, flagged = [], False
    for s in sentences:
        if MARKETING_TERMS.search(s):
            flagged = True
        else:
            kept.append(s)
    return kept, flagged


def clean_word(w):
    return re.sub(r"^[^\w]+|[^\w]+$", "", w)


def strip_marketing_words(words):
    return [w for w in words if w.lower() not in MARKETING_WORDS]


def collapse_plurals(words):
    seen_singular, out = set(), []
    for w in words:
        wl = w.lower()
        singular = wl[:-1] if wl.endswith("s") and len(wl) > 3 else wl
        if singular in seen_singular:
            continue
        seen_singular.add(singular)
        out.append(w)
    return out


def build_title(brand, packsize, category, old_title, old_bullets, extra_keywords, title_max):
    def remove_brand(text):
        if brand and brand.strip():
            text = re.sub(r"\b" + re.escape(brand.strip()) + r"\b", "", text or "", flags=re.IGNORECASE)
        return text

    base_words = [clean_word(w) for w in strip_separators(remove_brand(old_title)).split() if clean_word(w)]
    extra_words = [clean_word(w) for w in strip_separators(remove_brand(extra_keywords)).split() if clean_word(w)]
    bullet_words = [clean_word(w) for w in strip_separators(remove_brand(" ".join(old_bullets))).split() if clean_word(w)]

    base_words = strip_marketing_words(base_words)
    extra_words = strip_marketing_words(extra_words)
    bullet_words = strip_marketing_words(bullet_words)

    body_words = dedupe_words_preserve_order(base_words)

    if len(" ".join(body_words)) < title_max * 0.6:
        existing_lower = {x.lower() for x in body_words}
        for w in extra_words + bullet_words:
            if w.lower() in existing_lower:
                continue
            body_words.append(w)
            existing_lower.add(w.lower())
            if len(" ".join(body_words)) >= title_max * 0.85:
                break

    parts = ([brand] if brand else []) + body_words + ([packsize] if packsize else [])
    title = " ".join(dedupe_words_preserve_order(parts))
    title = re.sub(r"\s+", " ", title).strip()
    return truncate_at_word(title, title_max)


def build_bullets(old_bullets, description, extra_keywords, bullet_max, regulated):
    pool = list(old_bullets)
    pool += split_sentences(description)
    if extra_keywords.strip():
        pool += [p.strip() for p in re.split(r"[,\n]", extra_keywords) if p.strip()]

    pool, _ = apply_regulated_filter(pool, regulated)
    pool, marketing_flagged = apply_marketing_filter(pool)
    pool = dedupe_words_preserve_order([p for p in pool if p])

    bullets = []
    for p in pool[:5]:
        text = p[0].upper() + p[1:] if p else p
        bullets.append(truncate_at_word(text, bullet_max))
    return bullets, marketing_flagged


def build_description(old_description, old_bullets, description_max, regulated):
    if old_description.strip():
        sentences = split_sentences(old_description)
    else:
        sentences = old_bullets
    sentences, reg_flagged = apply_regulated_filter(sentences, regulated)
    sentences, marketing_flagged = apply_marketing_filter(sentences)
    text = " ".join(sentences)
    text = re.sub(r"\s+", " ", text).strip()
    return truncate_at_word(text, description_max), (reg_flagged or marketing_flagged)


def build_backend_terms(old_title, new_title, old_bullets, new_bullets, description, extra_keywords, backend_max_bytes):
    used = set(w.lower() for w in strip_separators(new_title).split())
    for b in new_bullets:
        used.update(clean_word(w).lower() for w in strip_separators(b).split())

    candidates = []
    for source in [old_title] + old_bullets + [description, extra_keywords]:
        for w in strip_separators(source).split():
            wl = clean_word(w).lower()
            if len(wl) < 3 or wl in STOPWORDS or wl in used or wl in MARKETING_WORDS:
                continue
            if wl.isdigit():
                continue
            candidates.append(wl)

    candidates = dedupe_words_preserve_order(candidates)
    candidates = collapse_plurals(candidates)
    terms = " ".join(candidates)
    return truncate_bytes_at_word(terms, backend_max_bytes)


def build_listing_rulebased(brand, packsize, category, old_title, old_bullets_raw, old_description, extra_keywords, regulated, lim):
    old_bullets_all = split_lines(old_bullets_raw)
    old_bullets, bullets_reg_flagged = apply_regulated_filter(old_bullets_all, regulated)
    old_bullets, bullets_mkt_flagged = apply_marketing_filter(old_bullets)

    safe_title = old_title
    safe_extra = extra_keywords
    other_flagged = False
    if regulated:
        if RISKY_TERMS.search(old_title or ""):
            safe_title = re.sub(r"\s+", " ", RISKY_TERMS.sub("", old_title)).strip()
            other_flagged = True
        if RISKY_TERMS.search(extra_keywords or ""):
            safe_extra = re.sub(r"\s+", " ", RISKY_TERMS.sub("", extra_keywords)).strip()
            other_flagged = True
    if MARKETING_TERMS.search(extra_keywords or ""):
        safe_extra = re.sub(r"\s+", " ", MARKETING_TERMS.sub("", safe_extra)).strip()
        other_flagged = True

    title = build_title(brand, packsize, category, safe_title, old_bullets, safe_extra, lim["title_max"])

    item_highlights = ""
    if lim["highlight_max"]:
        leftover = strip_separators(" ".join(old_bullets) + " " + safe_extra)
        item_highlights = truncate_at_word(leftover, lim["highlight_max"])

    bullets, marketing_flagged = build_bullets(old_bullets, old_description, safe_extra, lim["bullet_max"], regulated)
    description, desc_flagged = build_description(old_description, old_bullets, lim["description_max"], regulated)
    backend_terms = build_backend_terms(safe_title, title, old_bullets, bullets, description, safe_extra, lim["backend_max_bytes"])

    flagged = desc_flagged or bullets_reg_flagged or bullets_mkt_flagged or other_flagged or marketing_flagged
    return {
        "title": title,
        "item_highlights": item_highlights,
        "bullets": bullets,
        "description": description,
        "backend_terms": backend_terms,
    }, flagged


def build_image_content_rulebased(brand, category, packsize, bullets_text, description_text, title_text, theme_key):
    bullets = [b for b in re.split(r"\s*\|\s*|\n", bullets_text) if b.strip()]
    sentences = split_sentences(description_text)
    blob = " ".join([bullets_text, description_text, title_text]).lower()

    callouts = []
    for b in bullets[:6]:
        words = strip_separators(b).split()[:4]
        if words:
            callouts.append(" ".join(w.capitalize() if i == 0 else w for i, w in enumerate(words)))
    while len(callouts) < 6:
        callouts.append("")

    material = next((m for m in MATERIAL_KEYWORDS if m in blob), "\u2014")
    care = next((c for c in CARE_KEYWORDS if c in blob), "\u2014")
    spec_items = [
        {"label": "Size", "value": packsize or "\u2014"},
        {"label": "Material", "value": material.capitalize()},
        {"label": "Use", "value": category.replace("_", " ").title() if category else "\u2014"},
        {"label": "Care", "value": care.capitalize()},
        {"label": "Pack", "value": packsize or "\u2014"},
        {"label": "Brand", "value": brand or "\u2014"},
    ]

    supporting_line = sentences[0] if sentences else (bullets[0] if bullets else "")
    supporting_line = truncate_at_word(supporting_line, 90)

    lifestyle_tagline = THEME_TAGLINES.get(theme_key, THEME_TAGLINES["general"])
    size_guide_label = packsize if packsize else "True-to-size fit"

    return {
        "feature_callouts": callouts,
        "spec_items": spec_items,
        "supporting_line": supporting_line,
        "lifestyle_tagline": lifestyle_tagline,
        "size_guide_label": size_guide_label,
    }


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "listing" not in st.session_state:
    st.session_state.listing = None
if "history" not in st.session_state:
    st.session_state.history = []
if "imagery" not in st.session_state:
    st.session_state.imagery = None

tab_listing, tab_imagery = st.tabs(["Listing rewrite", "Product imagery"])

# ===========================================================================
# TAB 1 — LISTING REWRITE
# ===========================================================================

with tab_listing:
    left, right = st.columns([2, 3], gap="large")

    with left:
        card = st.container(border=True)
        with card:
            st.markdown('<p class="crSection">Product identity</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            brand = c1.text_input("Brand name", key="brand", placeholder="e.g. Biomed")
            packsize = c2.text_input("Pack / size / count", key="packsize", placeholder="e.g. 2 fl oz, Pack of 3")

            category = st.selectbox(
                "Category", options=list(CATEGORY_LABELS.keys()),
                format_func=lambda k: CATEGORY_LABELS[k], key="category",
            )

            title_mode = st.radio(
                "Title format", options=["new", "legacy"],
                format_func=lambda v: "2026 rule (75 char + highlights)" if v == "new" else "Legacy (category limit)",
                horizontal=True, key="title_mode",
                disabled=(category == "media"),
                index=1 if category == "media" else 0,
            )
            if category == "media":
                title_mode = "legacy"
                st.caption("Media titles are exempt from the 2026 75-character rule — using the legacy limit.")

            bullet_mode = st.radio(
                "Bullet length target", options=["200", "255", "500"],
                format_func=lambda v: {"200": "Concise (200)", "255": "Standard (255)", "500": "Extended (500, Brand Reg.)"}[v],
                horizontal=True, key="bullet_mode",
            )

            lim = limits_for(category, title_mode, bullet_mode)
            readout = f"Applying: title \u2264 {lim['title_max']} chars"
            if lim["highlight_max"]:
                readout += f", item highlights \u2264 {lim['highlight_max']} chars"
            if category == "media":
                readout += " (media is exempt from the 2026 75-char rule)"
            st.caption(readout)

            regulated = st.checkbox(
                "Regulated / supplement listing — enforce structure-function language only, "
                "no disease, cure, or organ-function claims.",
                key="regulated",
            )

        card2 = st.container(border=True)
        with card2:
            st.markdown('<p class="crSection">Current listing content</p>', unsafe_allow_html=True)
            old_title = st.text_area("Old title", key="old_title", height=70)
            old_bullets = st.text_area("Old bullet points (one per line)", key="old_bullets", height=120)
            old_description = st.text_area("Old description (optional)", key="old_description", height=100)
            extra_keywords = st.text_area("Known keywords / features (optional)", key="extra_keywords", height=80)

            generate_clicked = st.button("Refine listing", type="primary", use_container_width=True)

    if generate_clicked:
        with st.spinner("Refining listing..."):
            try:
                parsed, desc_flagged = build_listing_rulebased(
                    brand, packsize, category, old_title, old_bullets, old_description,
                    extra_keywords, regulated, lim,
                )
                if st.session_state.listing:
                    st.session_state.history.insert(0, {
                        "stamp": datetime.now().strftime("%H:%M:%S"),
                        "data": st.session_state.listing,
                    })
                st.session_state.listing = parsed
                if desc_flagged:
                    st.warning(
                        "One or more sentences were dropped \u2014 either they contained language "
                        "Amazon prohibits (subjective claims, award/ranking claims, promotional or "
                        "time-sensitive wording) or, in regulated mode, a disease/treatment claim. "
                        "Review the listing manually before publishing."
                    )
            except Exception as e:
                st.error(f"Could not generate the listing. {e}")

    with right:
        out_card = st.container(border=True)
        with out_card:
            st.markdown('<p class="crSection">Refined listing</p>', unsafe_allow_html=True)
            listing = st.session_state.listing
            if not listing:
                st.info(
                    "Fill in the manifest on the left and click Refine listing — your optimized title, "
                    "bullets, description, and backend search terms will appear here, each with a live "
                    "compliance gauge."
                )
            else:
                title_val = st.text_area("Title", value=listing.get("title", ""), key="out_title", height=70)
                title_count = len(title_val)
                st.markdown(gauge_badge(title_count, lim["title_max"], "chars"), unsafe_allow_html=True)
                ratios = [gauge_ratio(title_count, lim["title_max"])]

                if lim["highlight_max"]:
                    hl_val = st.text_area("Item highlights", value=listing.get("item_highlights", ""), key="out_highlights", height=70)
                    hl_count = len(hl_val)
                    st.markdown(gauge_badge(hl_count, lim["highlight_max"], "chars"), unsafe_allow_html=True)
                    ratios.append(gauge_ratio(hl_count, lim["highlight_max"]))

                st.markdown("**Bullet points**")
                bullets = listing.get("bullets", []) or [""] * 5
                for i in range(5):
                    b_val = st.text_area(f"Bullet {i+1}", value=bullets[i] if i < len(bullets) else "", key=f"out_bullet_{i}", height=70)
                    b_count = len(b_val)
                    st.markdown(gauge_badge(b_count, lim["bullet_max"], "chars"), unsafe_allow_html=True)
                    ratios.append(gauge_ratio(b_count, lim["bullet_max"]))

                desc_val = st.text_area("Description", value=listing.get("description", ""), key="out_description", height=200)
                desc_count = len(desc_val)
                st.markdown(gauge_badge(desc_count, lim["description_max"], "chars"), unsafe_allow_html=True)
                ratios.append(gauge_ratio(desc_count, lim["description_max"]))

                backend_val = st.text_area("Backend search terms", value=listing.get("backend_terms", ""), key="out_backend", height=70)
                backend_count = byte_len(backend_val)
                st.markdown(gauge_badge(backend_count, lim["backend_max_bytes"], "bytes"), unsafe_allow_html=True)
                st.caption("Backend terms are indexed by byte length, not character count.")
                with st.expander("What's excluded from backend terms, and why"):
                    st.markdown(
                        "- **Words already in the title or bullets** \u2014 Amazon already indexes those fields; repeating them wastes bytes.\n"
                        "- **Stopwords** (a, an, the, for, of, with...) \u2014 no indexing value.\n"
                        "- **Subjective, award, and promotional claims** (best, amazing, guaranteed, award-winning, "
                        "Amazon's Choice, on sale, limited time...) \u2014 Amazon's search-terms policy disallows these.\n"
                        "- **Redundant plurals** \u2014 Amazon's algorithm matches singular and plural automatically.\n"
                        "- **Brand name and ASIN-like tokens** \u2014 already associated with the listing; not useful here."
                    )
                ratios.append(gauge_ratio(backend_count, lim["backend_max_bytes"]))

                grade, grade_msg = compute_grade(ratios)
                st.markdown(
                    f'<div class="crGradeCard">'
                    f'<div class="crGradeCircle">{grade}</div>'
                    f'<div class="crMsg"><b>Listing health</b>{grade_msg}</div></div>',
                    unsafe_allow_html=True,
                )

        if st.session_state.history:
            hist_card = st.container(border=True)
            with hist_card:
                st.markdown('<p class="crSection">Previous versions</p>', unsafe_allow_html=True)
                for i, entry in enumerate(st.session_state.history):
                    with st.expander(f"Version saved {entry['stamp']}"):
                        d = entry["data"]
                        st.text_area("Title", value=d.get("title", ""), key=f"hist_title_{i}", height=60, disabled=True)
                        for j, b in enumerate(d.get("bullets", [])):
                            st.text_area(f"Bullet {j+1}", value=b, key=f"hist_bullet_{i}_{j}", height=60, disabled=True)
                        st.text_area("Description", value=d.get("description", ""), key=f"hist_desc_{i}", height=120, disabled=True)
                        st.text_area("Backend terms", value=d.get("backend_terms", ""), key=f"hist_backend_{i}", height=60, disabled=True)
                if st.button("Clear history"):
                    st.session_state.history = []
                    st.rerun()


# ===========================================================================
# IMAGE HELPERS
# ===========================================================================

def get_font(size, bold=False):
    names = (
        ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"]
        if bold else
        ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]
    )
    paths = [f"/usr/share/fonts/truetype/dejavu/{n}" for n in names] + names
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    words = (text or "").split()
    lines, line = [], ""
    for w in words:
        test = f"{line} {w}".strip()
        if draw.textlength(test, font=font) > max_width and line:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    return lines


def draw_wrapped(draw, text, xy, font, max_width, line_height, fill, anchor_h="l"):
    x, y = xy
    lines = wrap_text(draw, text, font, max_width)
    for i, l in enumerate(lines):
        if anchor_h == "m":
            w = draw.textlength(l, font=font)
            draw.text((x - w / 2, y + i * line_height), l, font=font, fill=fill)
        elif anchor_h == "r":
            w = draw.textlength(l, font=font)
            draw.text((x - w, y + i * line_height), l, font=font, fill=fill)
        else:
            draw.text((x, y + i * line_height), l, font=font, fill=fill)
    return len(lines)


def mix(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def luminance(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def smoothstep(v, lo, hi):
    t = max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi != lo else 0.0
    return t * t * (3 - 2 * t)


def build_cutout(img):
    img = img.convert("RGBA")
    orig_w, orig_h = img.width, img.height

    # Cap the OUTPUT resolution generously above the 2000px canvas so nothing
    # ever gets upscaled (which reads as blurry/"small" detail once composited).
    max_output_dim = 2600
    out_scale = min(1.0, max_output_dim / max(orig_w, orig_h))
    full_w, full_h = max(1, int(orig_w * out_scale)), max(1, int(orig_h * out_scale))
    full = img.resize((full_w, full_h), Image.LANCZOS) if out_scale < 1.0 else img

    # Compute the background-removal mask on a small, fast copy.
    mask_dim = 600
    mask_scale = min(1.0, mask_dim / max(full_w, full_h))
    mw, mh = max(1, int(full_w * mask_scale)), max(1, int(full_h * mask_scale))
    work = full.resize((mw, mh), Image.LANCZOS)
    arr = np.array(work).astype(np.float64)

    patch = max(4, int(min(mw, mh) * 0.04))

    def patch_avg(px, py):
        sub = arr[py:py + patch, px:px + patch, :3]
        if sub.size == 0:
            return np.array([255.0, 255.0, 255.0])
        return sub.reshape(-1, 3).mean(axis=0)

    corners = [patch_avg(0, 0), patch_avg(max(0, mw - patch), 0),
               patch_avg(0, max(0, mh - patch)), patch_avg(max(0, mw - patch), max(0, mh - patch))]
    bg = np.mean(corners, axis=0)

    diff = arr[:, :, :3] - bg
    dist = np.sqrt((diff ** 2).sum(axis=2))
    thresh_lo, thresh_hi = 42, 95
    t = np.clip((dist - thresh_lo) / (thresh_hi - thresh_lo), 0, 1)
    small_alpha = (t * t * (3 - 2 * t) * 255).astype(np.uint8)

    # Upscale the mask back to full output resolution (smooth resample keeps
    # edges soft rather than blocky) and apply it to the full-res pixels.
    alpha_img = Image.fromarray(small_alpha, "L").resize((full_w, full_h), Image.LANCZOS)
    full_rgba = np.array(full)
    full_rgba[:, :, 3] = np.array(alpha_img)
    out = Image.fromarray(full_rgba, "RGBA")

    alpha_full = full_rgba[:, :, 3]
    mask = alpha_full > 25
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        bbox = (0, 0, full_w, full_h)
    else:
        pad = int(min(full_w, full_h) * 0.01)
        bbox = (
            max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(full_w, int(xs.max()) + 1 + pad), min(full_h, int(ys.max()) + 1 + pad),
        )
    cropped = out.crop(bbox)

    carr = np.array(cropped)
    ca = carr[:, :, 3]
    crgb = carr[:, :, :3]
    opaque = ca > 25
    bright = (crgb[:, :, 0] > 235) & (crgb[:, :, 1] > 235) & (crgb[:, :, 2] > 235)
    dark = (crgb[:, :, 0] < 18) & (crgb[:, :, 1] < 18) & (crgb[:, :, 2] < 18)
    valid = opaque & ~bright & ~dark
    valid_px = crgb[valid]

    primary = (74, 108, 130)
    secondary = primary
    if len(valid_px) > 0:
        binned = (np.round(valid_px / 28) * 28).astype(int)
        step = max(1, len(binned) // 20000)
        sample = binned[::step]
        counts = Counter(map(tuple, sample.tolist()))
        common = counts.most_common(10)
        if common:
            primary = common[0][0]
            secondary = primary
            for color, _ in common[1:]:
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(color, primary)))
                if d > 60:
                    secondary = color
                    break

    white = (255, 255, 255)
    theme = {
        "primary": primary, "secondary": secondary,
        "light": mix(white, primary, 0.14),
        "tint": mix(white, primary, 0.07),
        "text_on_primary": (28, 31, 38) if luminance(primary) > 150 else (255, 255, 255),
    }
    return {"image": cropped, "width": cropped.width, "height": cropped.height, "theme": theme}


def add_drop_shadow(canvas, box, opacity=70):
    S = canvas.width
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    cx = (box["left"] + box["right"]) / 2
    ry = max(14, box["h"] * 0.045)
    rx = box["w"] * 0.36
    sd.ellipse([cx - rx, box["bottom"] - ry * 0.6, cx + rx, box["bottom"] + ry], fill=(0, 0, 0, opacity))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=S * 0.012))
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), shadow).convert(canvas.mode))


def paste_product(canvas, cutout, cx, cy, max_w, max_h):
    prod = cutout["image"]
    scale = min(max_w / prod.width, max_h / prod.height)
    w, h = max(1, int(prod.width * scale)), max(1, int(prod.height * scale))
    resized = prod.resize((w, h), Image.LANCZOS)
    left, top = int(cx - w / 2), int(cy - h / 2)
    canvas.paste(resized, (left, top), resized)
    return {"w": w, "h": h, "left": left, "top": top, "right": left + w, "bottom": top + h}


def vertical_gradient(S, top_c, bottom_c):
    top = np.array(top_c, dtype=float).reshape(1, 1, 3)
    bottom = np.array(bottom_c, dtype=float).reshape(1, 1, 3)
    t = np.linspace(0, 1, S).reshape(S, 1, 1)
    grad = top + (bottom - top) * t
    grad = np.repeat(grad, S, axis=1).astype(np.uint8)
    return Image.fromarray(grad, "RGB")


def diagonal_gradient(S, a_c, b_c):
    a = np.array(a_c, dtype=float)
    b = np.array(b_c, dtype=float)
    xx, yy = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
    t = ((xx + yy) / 2)[:, :, None]
    grad = a.reshape(1, 1, 3) + (b - a.reshape(1, 1, 3)) * t
    return Image.fromarray(grad.astype(np.uint8), "RGB")


def detect_theme(selected, brand, category, bullets_text, description_text, title_text, extra_text):
    if selected != "auto":
        return selected
    blob = " ".join([brand, category, bullets_text, description_text, title_text, extra_text]).lower()
    has_baby = bool(re.search(r"\b(baby|babies|infant|toddler|nursery|newborn|diaper|nappy)\b", blob))
    has_swim = bool(re.search(r"\b(swim|pool|water|beach|float|bath)\b", blob))
    if has_baby and has_swim:
        return "baby_swim"
    if has_baby:
        return "baby"
    if has_swim:
        return "swim"
    if re.search(r"\b(camp|hike|hiking|trail|outdoor|travel|backpack)\b", blob):
        return "outdoor"
    if re.search(r"\b(kitchen|cook|cooking|coffee|utensil|bake|baking|mug|pan)\b", blob):
        return "kitchen"
    if re.search(r"\b(fitness|gym|workout|yoga|sport|athletic|training)\b", blob):
        return "fitness"
    if re.search(r"\b(electronic|charger|cable|bluetooth|device|gadget|usb|wireless)\b", blob):
        return "tech"
    if re.search(r"\b(beauty|skincare|cosmetic|serum|makeup|spa|wellness)\b", blob):
        return "beauty"
    if category == "babypet":
        return "baby"
    if category == "supplement":
        return "beauty"
    if category == "electronics":
        return "tech"
    return "general"


def detect_persona(selected, theme_key, brand, category, bullets_text, description_text, title_text, extra_text):
    if selected != "auto":
        return selected
    if theme_key in ("baby", "baby_swim"):
        return "baby"
    blob = " ".join([brand, category, bullets_text, description_text, title_text, extra_text]).lower()
    if re.search(r"\b(men|mens|men's|man|male|for him|gentlemen)\b", blob):
        return "man"
    if re.search(r"\b(women|womens|women's|woman|female|for her|ladies)\b", blob):
        return "woman"
    if category == "babypet" and theme_key in ("baby", "baby_swim"):
        return "baby"
    return "adult"


def draw_motif(base, theme_key, opacity):
    S = base.width
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * opacity)

    if theme_key in ("swim", "baby_swim"):
        for band in range(3):
            base_y = S * (0.62 + band * 0.13)
            pts = [(0, base_y)]
            for x in range(0, S + 1, 40):
                y = base_y + math.sin((x / S) * math.pi * 4 + band) * 22
                pts.append((x, y))
            pts += [(S, S), (0, S)]
            band_a = int(a * (1 - band * 0.25))
            d.polygon(pts, fill=(255, 255, 255, max(0, band_a)))
        for bx, by, r in [(0.1, 0.14, 26), (0.2, 0.3, 14), (0.85, 0.12, 30), (0.78, 0.26, 16), (0.92, 0.4, 12), (0.06, 0.42, 18)]:
            d.ellipse([S * bx - r, S * by - r, S * bx + r, S * by + r], fill=(255, 255, 255, int(a * 0.8)))

    if theme_key in ("baby", "baby_swim"):
        ba = int(a * (0.8 if theme_key == "baby_swim" else 1.0))
        for bx, by, r in [(0.14, 0.16, 90), (0.86, 0.2, 70), (0.1, 0.86, 80), (0.9, 0.84, 100)]:
            d.ellipse([S * bx - r, S * by - r, S * bx + r, S * by + r], fill=(255, 255, 255, int(ba * 0.6)))

        def star(cx, cy, r, alpha):
            pts = []
            for i in range(5):
                a1 = math.pi * 2 * i / 5 - math.pi / 2
                a2 = a1 + math.pi / 5
                pts.append((cx + math.cos(a1) * r, cy + math.sin(a1) * r))
                pts.append((cx + math.cos(a2) * r * 0.42, cy + math.sin(a2) * r * 0.42))
            d.polygon(pts, fill=(255, 255, 255, alpha))

        star(S * 0.24, S * 0.32, 22, int(ba * 0.9))
        star(S * 0.78, S * 0.4, 16, int(ba * 0.9))
        star(S * 0.15, S * 0.7, 18, int(ba * 0.9))
        for i in range(14):
            x, y = (i * 137) % S, (i * 311) % S
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255, int(ba * 0.5)))

    elif theme_key == "outdoor":
        d.ellipse([S * 0.84 - 90, S * 0.16 - 90, S * 0.84 + 90, S * 0.16 + 90], fill=(255, 255, 255, int(a * 0.7)))
        pts = [(0, S * 0.9), (S * 0.22, S * 0.68), (S * 0.4, S * 0.84), (S * 0.6, S * 0.6),
               (S * 0.82, S * 0.84), (S, S * 0.72), (S, S), (0, S)]
        d.polygon(pts, fill=(255, 255, 255, a))

    elif theme_key == "kitchen":
        for sx, sy in [(0.14, 0.7), (0.86, 0.68), (0.1, 0.2)]:
            pts = []
            t = 0.0
            while t <= 1:
                x = S * sx + math.sin(t * math.pi * 3) * 18
                y = S * sy - t * S * 0.24
                pts.append((x, y))
                t += 0.05
            d.line(pts, fill=(255, 255, 255, int(a * 1.0)), width=10, joint="curve")
        d.ellipse([S * 0.5 - S * 0.34, S * 0.94 - S * 0.05, S * 0.5 + S * 0.34, S * 0.94 + S * 0.05],
                  fill=(255, 255, 255, int(a * 0.5)))

    elif theme_key == "fitness":
        for i in range(-2, 8):
            x0, y0 = i * S * 0.16, S * 1.05
            x1, y1 = i * S * 0.16 + S * 0.3, -S * 0.05
            d.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, int(a * 0.5)), width=26)

    elif theme_key == "tech":
        step = S / 12
        for i in range(1, 12):
            d.line([(i * step, 0), (i * step, S)], fill=(255, 255, 255, int(a * 0.55)), width=3)
            d.line([(0, i * step), (S, i * step)], fill=(255, 255, 255, int(a * 0.55)), width=3)
        for gx, gy in [(3, 4), (8, 2), (10, 9), (2, 10), (6, 7)]:
            x, y = gx * step, gy * step
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(255, 255, 255, int(a * 0.8)))

    elif theme_key == "beauty":
        def bloom(cx, cy, r, petals, alpha):
            for i in range(petals):
                ang = math.pi * 2 * i / petals
                ex, ey = cx + math.cos(ang) * r * 0.6, cy + math.sin(ang) * r * 0.6
                bbox = [ex - r * 0.5, ey - r * 0.24, ex + r * 0.5, ey + r * 0.24]
                petal_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
                pd = ImageDraw.Draw(petal_layer)
                pd.ellipse(bbox, fill=(255, 255, 255, alpha))
                rotated = petal_layer.rotate(math.degrees(ang), center=(ex, ey))
                layer.alpha_composite(rotated)

        bloom(S * 0.16, S * 0.2, 90, 7, int(a * 0.5))
        bloom(S * 0.88, S * 0.82, 110, 8, int(a * 0.5))

    elif theme_key == "general" and theme_key not in ("baby", "baby_swim", "swim"):
        for x, y, r in [(0.15, 0.18, 260), (0.86, 0.22, 180), (0.12, 0.82, 220), (0.9, 0.85, 300)]:
            d.ellipse([S * x - r, S * y - r, S * x + r, S * y + r], fill=(255, 255, 255, a))

    base.paste(layer, (0, 0), layer)


# ===========================================================================
# IMAGE TEMPLATES
# ===========================================================================

def draw_main_image(cutout):
    S = 2000
    canvas = Image.new("RGB", (S, S), (255, 255, 255))
    box = paste_product(canvas, cutout, S / 2, S / 2, S * 0.85, S * 0.85)
    add_drop_shadow(canvas, box, opacity=45)
    return canvas


def draw_feature_infographic(cutout, callouts, supporting_line, theme, theme_key):
    S = 2000
    canvas = vertical_gradient(S, theme["tint"], (255, 255, 255)).convert("RGBA")
    draw_motif(canvas, theme_key, 0.16)
    canvas = canvas.convert("RGB")
    d = ImageDraw.Draw(canvas)
    font_sub = get_font(30)
    font_body = get_font(26, bold=True)

    if supporting_line:
        draw_wrapped(d, supporting_line, (S / 2, S * 0.045), font_sub, S * 0.7, 38, theme["primary"], anchor_h="m")

    box = paste_product(canvas, cutout, S / 2, S * 0.55, S * 0.46, S * 0.56)
    add_drop_shadow(canvas, box, opacity=40)
    d = ImageDraw.Draw(canvas)

    items = (callouts if callouts else ["Key feature"] * 6)[:6]
    positions = [
        {"x": S * 0.16, "y": S * 0.24, "side": "left", "anchor": (box["left"], box["top"] + box["h"] * 0.15)},
        {"x": S * 0.16, "y": S * 0.55, "side": "left", "anchor": (box["left"], box["top"] + box["h"] * 0.5)},
        {"x": S * 0.16, "y": S * 0.86, "side": "left", "anchor": (box["left"], box["bottom"] - box["h"] * 0.15)},
        {"x": S * 0.84, "y": S * 0.24, "side": "right", "anchor": (box["right"], box["top"] + box["h"] * 0.15)},
        {"x": S * 0.84, "y": S * 0.55, "side": "right", "anchor": (box["right"], box["top"] + box["h"] * 0.5)},
        {"x": S * 0.84, "y": S * 0.86, "side": "right", "anchor": (box["right"], box["bottom"] - box["h"] * 0.15)},
    ]

    for text, pos in zip(items, positions):
        bw, bh = S * 0.28, S * 0.12
        bx = pos["x"] - 8 if pos["side"] == "left" else pos["x"] - bw + 8
        by = pos["y"] - bh / 2
        dot_x = bx + bw if pos["side"] == "left" else bx

        d.line([(dot_x, pos["y"]), pos["anchor"]], fill=theme["primary"], width=4)
        d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=20, fill=(255, 255, 255), outline=theme["primary"], width=4)

        icon_x = bx + 38 if pos["side"] == "left" else bx + bw - 38
        text_x = bx + 68 if pos["side"] == "left" else bx + bw - 22
        d.ellipse([icon_x - 18, by + bh / 2 - 18, icon_x + 18, by + bh / 2 + 18], fill=theme["primary"])
        d.line([(icon_x - 9, by + bh / 2), (icon_x - 2, by + bh / 2 + 8), (icon_x + 10, by + bh / 2 - 9)],
               fill=(255, 255, 255), width=4)

        anchor_h = "l" if pos["side"] == "left" else "r"
        draw_wrapped(d, text, (text_x, by + bh / 2 - 18), font_body, bw - 90, 32, (28, 31, 38), anchor_h=anchor_h)

    return canvas


def draw_specs_infographic(cutout, spec_items, theme, brand, packsize, theme_key):
    S = 2000
    canvas = Image.new("RGB", (S, S), (255, 255, 255))
    strip = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    strip_bg = ImageDraw.Draw(strip)
    strip_bg.rectangle([0, S * 0.86, S, S], fill=theme["tint"] + (255,))
    draw_motif(strip, theme_key, 0.5)
    canvas.paste(strip.convert("RGB"), (0, int(S * 0.86)), None)
    canvas.paste(strip.convert("RGB").crop((0, int(S * 0.86), S, S)), (0, int(S * 0.86)))

    d = ImageDraw.Draw(canvas)
    header_h = S * 0.14
    d.rectangle([0, 0, S, header_h], fill=theme["primary"])
    font_header = get_font(50, bold=True)
    font_small = get_font(28)
    d.text((60, header_h / 2 - 25), (brand or "Product").upper(), font=font_header, fill=theme["text_on_primary"])
    if packsize:
        w = d.textlength(packsize, font=font_small)
        d.text((S - 60 - w, header_h / 2 - 14), packsize, font=font_small, fill=theme["text_on_primary"])

    box = paste_product(canvas, cutout, S * 0.27, header_h + (S * 0.86 - header_h) * 0.52, S * 0.42, (S * 0.86 - header_h) * 0.86)
    add_drop_shadow(canvas, box, opacity=40)
    d = ImageDraw.Draw(canvas)

    items = (spec_items if spec_items else [
        {"label": "Size", "value": packsize or "\u2014"},
        {"label": "Material", "value": "\u2014"},
        {"label": "Use", "value": "\u2014"},
        {"label": "Care", "value": "\u2014"},
        {"label": "Pack", "value": "\u2014"},
        {"label": "Fit", "value": "\u2014"},
    ])[:6]

    list_x, list_w = S * 0.54, S * 0.40
    row_h = (S - header_h) / len(items)
    font_label = get_font(30, bold=True)
    font_value = get_font(26)
    font_icon = get_font(26, bold=True)
    for i, item in enumerate(items):
        row_y = header_h + row_h * i
        if i > 0:
            d.line([(list_x, row_y), (list_x + list_w, row_y)], fill=(227, 225, 216), width=2)
        d.ellipse([list_x, row_y + row_h / 2 - 26, list_x + 60, row_y + row_h / 2 + 26], fill=theme["primary"])
        letter = (item.get("label", "?")[:1] or "?").upper()
        lw = d.textlength(letter, font=font_icon)
        d.text((list_x + 30 - lw / 2, row_y + row_h / 2 - 15), letter, font=font_icon, fill=theme["text_on_primary"])
        d.text((list_x + 76, row_y + row_h / 2 - 34), item.get("label", ""), font=font_label, fill=(28, 31, 38))
        draw_wrapped(d, item.get("value", ""), (list_x + 76, row_y + row_h / 2 + 4), font_value, list_w - 96, 32, (107, 114, 128))

    return canvas


def draw_baby_silhouette(d, cx, cy, scale, color):
    head_r = 40 * scale
    d.ellipse([cx - head_r, cy - 2.6 * head_r, cx + head_r, cy - 0.6 * head_r], fill=color)
    body_w, body_h = 92 * scale, 74 * scale
    body_top = cy - 0.6 * head_r
    d.rounded_rectangle([cx - body_w / 2, body_top, cx + body_w / 2, body_top + body_h], radius=30 * scale, fill=color)
    leg_r = 22 * scale
    d.ellipse([cx - body_w / 2 - 4 * scale, body_top + body_h - leg_r, cx - body_w / 2 + leg_r * 1.6, body_top + body_h + leg_r], fill=color)
    d.ellipse([cx + body_w / 2 - leg_r * 1.6, body_top + body_h - leg_r, cx + body_w / 2 + 4 * scale, body_top + body_h + leg_r], fill=color)
    arm_r = 16 * scale
    d.ellipse([cx - body_w / 2 - arm_r * 1.3, body_top + 10 * scale, cx - body_w / 2 + arm_r * 0.4, body_top + 10 * scale + arm_r * 2], fill=color)
    d.ellipse([cx + body_w / 2 - arm_r * 0.4, body_top + 10 * scale, cx + body_w / 2 + arm_r * 1.3, body_top + 10 * scale + arm_r * 2], fill=color)


def draw_adult_silhouette(d, cx, top_y, scale, color, gender="neutral"):
    head_r = 46 * scale
    head_cy = top_y + head_r
    if gender == "woman":
        hair_w = head_r * 1.35
        d.ellipse([cx - hair_w, head_cy - head_r * 0.5, cx + hair_w, head_cy + head_r * 2.0], fill=color)
    d.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=color)

    torso_top = head_cy + head_r * 0.85
    shoulder_w = (100 if gender == "woman" else 118) * scale
    waist_w = 66 * scale
    torso_h = 195 * scale
    d.polygon([
        (cx - shoulder_w / 2, torso_top),
        (cx + shoulder_w / 2, torso_top),
        (cx + waist_w / 2, torso_top + torso_h),
        (cx - waist_w / 2, torso_top + torso_h),
    ], fill=color)

    arm_w = 26 * scale
    arm_h = 150 * scale
    d.rounded_rectangle([cx - shoulder_w / 2 - arm_w * 0.6, torso_top + 10 * scale,
                          cx - shoulder_w / 2 + arm_w * 0.5, torso_top + 10 * scale + arm_h],
                         radius=arm_w / 2, fill=color)
    d.rounded_rectangle([cx + shoulder_w / 2 - arm_w * 0.5, torso_top + 10 * scale,
                          cx + shoulder_w / 2 + arm_w * 0.6, torso_top + 10 * scale + arm_h],
                         radius=arm_w / 2, fill=color)

    leg_top = torso_top + torso_h
    leg_w = waist_w / 2 - 6 * scale
    leg_h = 175 * scale
    d.rounded_rectangle([cx - waist_w / 2, leg_top, cx - waist_w / 2 + leg_w, leg_top + leg_h], radius=leg_w / 2, fill=color)
    d.rounded_rectangle([cx + waist_w / 2 - leg_w, leg_top, cx + waist_w / 2, leg_top + leg_h], radius=leg_w / 2, fill=color)
    return leg_top + leg_h


def draw_person(canvas_rgba, S, persona_key):
    if persona_key == "none":
        return
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    color = (255, 255, 255, 200)
    scale = S / 1400

    if persona_key == "baby":
        cx = S * 0.20
        cy = S * 0.78
        draw_baby_silhouette(d, cx, cy, scale * 1.4, color)
    else:
        gender = persona_key if persona_key in ("man", "woman") else "neutral"
        cx = S * 0.16
        top_y = S * 0.34
        draw_adult_silhouette(d, cx, top_y, scale, color, gender)

    canvas_rgba.alpha_composite(layer)


def draw_ground_band(canvas_rgba, S, theme):
    band = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    dark = mix(theme["primary"], (0, 0, 0), 0.35)
    bd.rectangle([0, S * 0.82, S, S], fill=dark + (60,))
    canvas_rgba.alpha_composite(band)



def draw_lifestyle_graphic(cutout, tagline, theme, theme_key, persona_key="adult"):
    S = 2000
    canvas = diagonal_gradient(S, theme["primary"], theme["secondary"]).convert("RGBA")
    draw_motif(canvas, theme_key, 0.22)
    if theme_key in ("tech", "beauty", "fitness", "general"):
        draw_ground_band(canvas, S, theme)
    draw_person(canvas, S, persona_key)
    canvas = canvas.convert("RGB")

    box = paste_product(canvas, cutout, S / 2, S * 0.5, S * 0.6, S * 0.6)
    add_drop_shadow(canvas, box, opacity=55)
    d = ImageDraw.Draw(canvas)

    banner_h = S * 0.14
    banner_y = S - banner_h - S * 0.06
    d.rounded_rectangle([S * 0.1, banner_y, S * 0.9, banner_y + banner_h], radius=28, fill=(255, 255, 255))

    font_tag = get_font(42, bold=True)
    draw_wrapped(d, tagline or "Made for real, everyday use", (S / 2, banner_y + banner_h / 2 - 30),
                 font_tag, S * 0.72, 52, theme["primary"], anchor_h="m")
    return canvas


def draw_size_guide(cutout, size_label, packsize, theme):
    S = 2000
    canvas = Image.new("RGB", (S, S), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, S * 0.86, S, S], fill=theme["tint"])

    box = paste_product(canvas, cutout, S / 2, S * 0.44, S * 0.58, S * 0.62)
    add_drop_shadow(canvas, box, opacity=40)
    d = ImageDraw.Draw(canvas)

    arrow_y = box["bottom"] + 60
    dash_len, gap = 14, 10
    x = box["left"]
    while x < box["right"]:
        d.line([(x, arrow_y), (min(x + dash_len, box["right"]), arrow_y)], fill=theme["primary"], width=4)
        x += dash_len + gap
    for xline in [box["left"], box["right"]]:
        d.line([(xline, arrow_y - 14), (xline, arrow_y + 14)], fill=theme["primary"], width=4)

    font_label = get_font(32, bold=True)
    label = size_label or "True-to-size fit"
    lw = d.textlength(label, font=font_label)
    d.text((S / 2 - lw / 2, arrow_y + 26), label, font=font_label, fill=theme["primary"])

    font_pack = get_font(38, bold=True)
    pack_text = packsize or "See size chart for full details"
    pw = d.textlength(pack_text, font=font_pack)
    d.text((S / 2 - pw / 2, S * 0.93 - 20), pack_text, font=font_pack, fill=(28, 31, 38))

    return canvas


# ===========================================================================
# TAB 2 — PRODUCT IMAGERY
# ===========================================================================

with tab_imagery:
    img_card = st.container(border=True)
    with img_card:
        st.markdown('<p class="crSection">Product imagery</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([3, 2])
        with c1:
            raw_file = st.file_uploader("Raw product photo", type=["png", "jpg", "jpeg", "webp"])
        with c2:
            sku_input = st.text_input("SKU / ASIN (optional, used for filenames)", key="sku_input")
            theme_choice = st.selectbox(
                "Imagery theme", options=list(THEME_LABELS.keys()),
                format_func=lambda k: THEME_LABELS[k], key="theme_choice",
            )
            persona_choice = st.selectbox(
                "Lifestyle figure", options=list(PERSONA_LABELS.keys()),
                format_func=lambda k: PERSONA_LABELS[k], key="persona_choice",
                help="Adds a simple stylized human silhouette to the lifestyle-style graphic — not a photoreal person.",
            )
        st.caption(
            "Works best when the raw photo already has a fairly plain, light background — the tool "
            "auto-detects and clears it. Busy scenes may need manual cleanup first."
        )
        generate_images_clicked = st.button(
            "Generate image set", type="primary", use_container_width=True,
            disabled=not raw_file,
        )

    if generate_images_clicked and raw_file:
        with st.spinner("Generating image set..."):
            try:
                img = Image.open(raw_file)
                cutout = build_cutout(img)

                listing = st.session_state.listing
                bullets_text = " | ".join(listing.get("bullets", [])) if listing else st.session_state.get("old_bullets", "")
                description_text = listing.get("description", "") if listing else st.session_state.get("old_description", "")
                title_text = listing.get("title", "") if listing else st.session_state.get("old_title", "")

                brand_val = st.session_state.get("brand", "")
                category_val = st.session_state.get("category", "general")
                packsize_val = st.session_state.get("packsize", "")
                extra_val = st.session_state.get("extra_keywords", "")

                theme_key = detect_theme(theme_choice, brand_val, category_val, bullets_text, description_text, title_text, extra_val)
                persona_key = detect_persona(persona_choice, theme_key, brand_val, category_val, bullets_text, description_text, title_text, extra_val)

                content = build_image_content_rulebased(
                    brand_val, category_val, packsize_val, bullets_text, description_text, title_text, theme_key,
                )

                sku = re.sub(r"[^a-zA-Z0-9]", "", sku_input) or "product"
                theme = cutout["theme"]

                images = [
                    {"label": "Main image", "code": "MAIN",
                     "note": "Pure white background, product \u226585% of frame \u2014 meets Amazon's main-image spec.",
                     "canvas": draw_main_image(cutout)},
                    {"label": "Feature infographic", "code": "PT01",
                     "note": "Secondary slot \u2014 callouts and color backgrounds are allowed here.",
                     "canvas": draw_feature_infographic(cutout, content.get("feature_callouts"), content.get("supporting_line"), theme, theme_key)},
                    {"label": "Spec infographic", "code": "PT02", "note": "Secondary slot.",
                     "canvas": draw_specs_infographic(cutout, content.get("spec_items"), theme, brand_val, packsize_val, theme_key)},
                    {"label": "Lifestyle-style graphic", "code": "PT03",
                     "note": "Stylized backdrop with a simple human silhouette, not a photoreal scene \u2014 secondary slot.",
                     "canvas": draw_lifestyle_graphic(cutout, content.get("lifestyle_tagline"), theme, theme_key, persona_key)},
                    {"label": "Size guide", "code": "PT04", "note": "Secondary slot.",
                     "canvas": draw_size_guide(cutout, content.get("size_guide_label"), packsize_val, theme)},
                ]
                for item in images:
                    item["filename"] = f"{sku}.{item['code']}.jpg"

                st.session_state.imagery = images
            except Exception as e:
                st.error(f"Could not generate the image set. {e}")


    out_img_card = st.container(border=True)
    with out_img_card:
        st.markdown('<p class="crSection">Generated set</p>', unsafe_allow_html=True)
        images = st.session_state.imagery
        if not images:
            st.info(
                "Upload a raw product photo and click Generate image set. You'll get a compliant "
                "white-background main image plus four secondary images (feature infographic, spec "
                "infographic, lifestyle-style graphic, size guide), themed to your product's own colors "
                "and ready to download."
            )
        else:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for item in images:
                    b = io.BytesIO()
                    item["canvas"].save(b, format="JPEG", quality=92)
                    zf.writestr(item["filename"], b.getvalue())
            st.download_button(
                "Download all (.zip)", data=zip_buf.getvalue(),
                file_name="amazon_image_set.zip", mime="application/zip",
                use_container_width=True,
            )

            hero = images[0]
            hero_col, hero_meta_col = st.columns([3, 1])
            with hero_col:
                st.image(hero["canvas"], use_container_width=True)
            with hero_meta_col:
                st.markdown(f"**{hero['label']}**")
                st.caption(hero["note"])
                buf = io.BytesIO()
                hero["canvas"].save(buf, format="JPEG", quality=92)
                st.download_button(
                    "Download", data=buf.getvalue(), file_name=hero["filename"],
                    mime="image/jpeg", key="dl_0", use_container_width=True,
                )

            st.markdown("---")
            cols = st.columns(2)
            for i, item in enumerate(images[1:], start=1):
                with cols[(i - 1) % 2]:
                    cell = st.container(border=True)
                    with cell:
                        st.image(item["canvas"], use_container_width=True)
                        st.markdown(f"**{item['label']}**")
                        st.caption(item["note"])
                        buf = io.BytesIO()
                        item["canvas"].save(buf, format="JPEG", quality=92)
                        st.download_button(
                            "Download", data=buf.getvalue(), file_name=item["filename"],
                            mime="image/jpeg", key=f"dl_{i}", use_container_width=True,
                        )

