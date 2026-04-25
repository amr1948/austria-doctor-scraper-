#!/usr/bin/env python3
"""
Austria Doctor Scraper
======================
Scrapes doctor Name and Address from docfinder.at — Austria's largest
medical directory — across all ~45 medical specialties, then deduplicates
and exports to CSV.

Usage
-----
  python austria_doctors_scraper.py                         # full run
  python austria_doctors_scraper.py --test                  # 2 pages / specialty
  python austria_doctors_scraper.py --output doctors.csv    # custom output file
  python austria_doctors_scraper.py --specialty zahnarzt    # one specialty only
  python austria_doctors_scraper.py --list-specialties      # print all slugs
"""

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_OUTPUT   = "austria_doctors.csv"
MAX_CONCURRENT   = 3          # polite: simultaneous requests
REQUEST_DELAY    = (1.0, 2.5) # random pause per request (seconds)
MAX_RETRIES      = 3
MAX_PAGES        = 300        # safety cap per specialty
CHECKPOINT_EVERY = 10         # write partial CSV every N specialties

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-AT,de;q=0.9,en;q=0.5",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Referer": "https://www.docfinder.at/",
}

# ── Austrian medical specialties on docfinder.at ─────────────────────────────
# Each slug maps to: https://www.docfinder.at/suche/<slug>

SPECIALTIES: List[str] = [
    # General practice
    "praktischer-arzt",
    # Surgery
    "allgemeinchirurg",
    "unfallchirurg",
    "plastischer-chirurg",
    "neurochirurg",
    "herzchirurg",
    "gefaesschirurg",
    "thoraxchirurg",
    "kieferchirurg",
    "oralchirurg",
    # Internal medicine & subspecialties
    "internist",
    "kardiologe",
    "gastroenterologe",
    "pneumologe",
    "rheumatologe",
    "nephrologe",
    "onkologe",
    "endokrinologe",
    "infektiologe",
    "haematologe",
    "intensivmedizin",
    # Neurology / Psychiatry
    "neurologe",
    "psychiater",
    "kinder-und-jugendpsychiater",
    "psychosomatik",
    # Pediatrics
    "kinderarzt",
    "kinderfacharzt",
    "neonatologe",
    # Women's health
    "frauenarzt",
    # Skin
    "hautarzt",
    # Eyes
    "augenfacharzt",
    # ENT
    "hno-arzt",
    # Orthopedics / Urology
    "orthopaede",
    "urologe",
    # Imaging / Radiation
    "radiologe",
    "nuklearmediziner",
    "strahlentherapeut",
    # Dental
    "zahnarzt",
    "kieferorthopaedie",
    # Mental health / Therapy
    "psychologe",
    "psychotherapeut",
    # Physical medicine / Occupational
    "physikalische-medizin",
    "sportarzt",
    "arbeitsmediziner",
    # Anesthesia / Lab / Pathology
    "anaesthesiologie",
    "labormedizin",
    "pathologie",
    # Geriatrics / Emergency
    "geriatrie",
    "notarzt",
    # Other
    "humangenetiker",
]

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Doctor:
    name: str
    address: str


# ── Regex helpers ─────────────────────────────────────────────────────────────

# 4-digit Austrian postal code followed by a capital letter (city start)
AT_ZIP_RE = re.compile(r"\b\d{4}\s+[A-ZÄÖÜ]")

# Doctor name prefix patterns
DR_TITLE_RE = re.compile(
    r"^(Dr\.?\s|Univ\.?[-–\s]|OA\s|OÄ\s|Prof\.?\s|Prim\.?\s|DDr\.?\s|"
    r"Doz\.?\s|Priv\.?[-\s]?Doz\.?|Ass\.?\s?Prof|FH\s|Mag\.?\s|"
    r"DI\s|PhD\s|MBA\s|MSc\s)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _dedup(docs: List[Doctor]) -> List[Doctor]:
    seen: Set[Tuple[str, str]] = set()
    result = []
    for d in docs:
        key = (d.name.lower(), d.address.lower())
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


# ── HTTP fetch ────────────────────────────────────────────────────────────────

async def fetch(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
    page: int = 1,
) -> Optional[str]:
    """GET with polite delay, semaphore, and exponential-backoff retries."""
    params = {"page": page} if page > 1 else None
    for attempt in range(MAX_RETRIES):
        try:
            await asyncio.sleep(random.uniform(*REQUEST_DELAY))
            async with sem:
                resp = await client.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                log.warning(f"Rate-limited – sleeping {wait}s")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                log.debug(f"  failed {url} page={page}: {exc}")
                return None
            await asyncio.sleep(3 * 2 ** attempt)
    return None


# ── HTML parsing ──────────────────────────────────────────────────────────────

def parse_page(html: str) -> List[Doctor]:
    """Extract doctors from one result page using two complementary strategies."""
    soup = BeautifulSoup(html, "lxml")
    doctors: List[Doctor] = []

    # ── Strategy 1: Schema.org JSON-LD structured data ────────────────────────
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            blob = json.loads(tag.string or "")
            items = blob if isinstance(blob, list) else [blob]
            for item in items:
                # unwrap @graph
                if "@graph" in item:
                    items = list(items) + item["@graph"]
                t = item.get("@type", "")
                if t not in ("Physician", "LocalBusiness", "MedicalBusiness",
                             "MedicalOrganization", "Person"):
                    continue
                name = item.get("name", "")
                addr = item.get("address", {})
                if isinstance(addr, str):
                    addr_str = addr
                elif isinstance(addr, dict):
                    addr_str = ", ".join(filter(None, [
                        addr.get("streetAddress", ""),
                        addr.get("postalCode", ""),
                        addr.get("addressLocality", ""),
                    ]))
                else:
                    addr_str = ""
                if name and addr_str and AT_ZIP_RE.search(
                    addr_str.replace(",", " ")
                ):
                    doctors.append(Doctor(_clean(name), _clean(addr_str)))
        except Exception:
            pass

    if doctors:
        return _dedup(doctors)

    # ── Strategy 2: Line-based text scan ─────────────────────────────────────
    # Flatten all visible text to lines; look for (name line, address line) pairs.
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        if not (5 < len(line) < 160):
            continue
        if not DR_TITLE_RE.match(line):
            continue
        # found a probable name – scan the next few lines for an Austrian address
        for j in range(i + 1, min(i + 6, len(lines))):
            candidate = lines[j]
            if AT_ZIP_RE.search(candidate) and len(candidate) < 200:
                doctors.append(Doctor(_clean(line), _clean(candidate)))
                break

    # ── Strategy 3: DOM block scan (fallback) ─────────────────────────────────
    if not doctors:
        for block in soup.find_all(["article", "li", "div"]):
            block_text = _clean(block.get_text(" "))
            if not (10 < len(block_text) < 600):
                continue
            if not AT_ZIP_RE.search(block_text):
                continue
            if not DR_TITLE_RE.search(block_text):
                continue
            name = _extract_name_from_block(block)
            addr = _extract_addr_from_block(block)
            if name and addr:
                doctors.append(Doctor(name, addr))

    return _dedup(doctors)


def _extract_name_from_block(block) -> Optional[str]:
    for tag in ("h2", "h3", "h4", "h5", "strong", "b"):
        el = block.find(tag)
        if el:
            t = _clean(el.get_text())
            if DR_TITLE_RE.match(t) and 5 < len(t) < 160:
                return t
    for s in block.strings:
        t = _clean(s)
        if DR_TITLE_RE.match(t) and 5 < len(t) < 160:
            return t
    return None


def _extract_addr_from_block(block) -> Optional[str]:
    addr_el = block.find("address")
    if addr_el:
        return _clean(addr_el.get_text(", "))
    for tag in ("p", "span", "div", "small"):
        for el in block.find_all(tag):
            t = _clean(el.get_text(" "))
            if AT_ZIP_RE.search(t) and 5 < len(t) < 200:
                return t
    for s in block.strings:
        t = _clean(s)
        if AT_ZIP_RE.search(t) and 5 < len(t) < 200:
            return t
    return None


def _has_next_page(html: str, current_page: int) -> bool:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if re.search(r"weiter|next|›|»|▶", text, re.IGNORECASE):
            return True
        if re.search(rf"[?&]page={current_page + 1}\b", href):
            return True
    return False


# ── Scraper ───────────────────────────────────────────────────────────────────

class DocFinderScraper:
    BASE = "https://www.docfinder.at/suche"

    def __init__(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        max_pages: int,
    ) -> None:
        self.client    = client
        self.sem       = sem
        self.max_pages = max_pages

    async def scrape_specialty(self, slug: str) -> List[Doctor]:
        url = f"{self.BASE}/{slug}"
        all_docs: List[Doctor] = []

        for page in range(1, self.max_pages + 1):
            html = await fetch(self.client, url, self.sem, page)
            if html is None:
                break

            page_docs = parse_page(html)
            if not page_docs:
                break

            all_docs.extend(page_docs)

            if not _has_next_page(html, page):
                break

        return _dedup(all_docs)

    async def run(
        self,
        specialties: List[str],
        output: str,
    ) -> List[Doctor]:
        all_docs: List[Doctor] = []

        for i, slug in enumerate(
            tqdm(specialties, desc="Specialties", unit="spec", ncols=70)
        ):
            try:
                docs = await self.scrape_specialty(slug)
                all_docs.extend(docs)
                log.info(
                    f"  ✓ {slug:<40s}  {len(docs):>4d} found"
                    f"  │  {len(_dedup(all_docs)):,} total unique"
                )
            except Exception as exc:
                log.error(f"  ✗ {slug}: {exc}")

            # Periodic checkpoint so partial results are never lost
            if (i + 1) % CHECKPOINT_EVERY == 0:
                _save_csv(_dedup(all_docs), output)
                log.info(f"  [checkpoint] saved {len(_dedup(all_docs)):,} doctors → {output}")

        return _dedup(all_docs)


# ── CSV export ────────────────────────────────────────────────────────────────

def _save_csv(doctors: List[Doctor], path: str) -> None:
    unique = {(d.name, d.address): d for d in doctors if d.name and d.address}
    rows = sorted(unique.values(), key=lambda d: d.name.lower())
    df = pd.DataFrame([{"Name": d.name, "Address": d.address} for d in rows])
    # utf-8-sig BOM makes the file open correctly in Excel on Windows/macOS
    df.to_csv(path, index=False, encoding="utf-8-sig")


def export_csv(doctors: List[Doctor], path: str) -> None:
    _save_csv(doctors, path)
    unique_count = len({(d.name, d.address) for d in doctors if d.name and d.address})
    print()
    print("━" * 60)
    print(f"  Unique doctors : {unique_count:,}")
    print(f"  Output file    : {path}")
    print("━" * 60)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scrape Austrian doctor Name + Address from docfinder.at → CSV.\n"
            "Iterates through all medical specialties and paginates each one."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test mode: only fetch 2 pages per specialty",
    )
    p.add_argument(
        "--specialty", "-s",
        metavar="SLUG",
        help="Scrape a single specialty slug (e.g. zahnarzt)",
    )
    p.add_argument(
        "--list-specialties",
        action="store_true",
        help="Print all available specialty slugs and exit",
    )
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    args = _build_args()

    if args.list_specialties:
        print(f"Available specialties ({len(SPECIALTIES)} total):")
        for slug in SPECIALTIES:
            print(f"  {slug}")
        return

    max_pages   = 2 if args.test else MAX_PAGES
    specialties = [args.specialty] if args.specialty else SPECIALTIES

    print()
    print("━━  Austria Doctor Scraper  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Source      : docfinder.at")
    print(f"  Specialties : {len(specialties)}")
    print(f"  Max pages   : {max_pages} per specialty")
    print(f"  Concurrency : {MAX_CONCURRENT} requests")
    print(f"  Output      : {args.output}")
    if args.test:
        print("  Mode        : TEST (2 pages per specialty)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient(
        headers=HTTP_HEADERS,
        follow_redirects=True,
        timeout=httpx.Timeout(30.0),
        http2=True,
    ) as client:
        scraper   = DocFinderScraper(client, sem, max_pages)
        all_docs  = await scraper.run(specialties, args.output)

    export_csv(all_docs, args.output)


if __name__ == "__main__":
    asyncio.run(main())
