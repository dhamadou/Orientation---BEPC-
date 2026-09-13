import json
import datetime
import os

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = f'{_TOOLS_DIR}/raw'
REPO_DATA = os.path.normpath(f'{_TOOLS_DIR}/../data')

def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

import re

BRACKET_RE = re.compile(r'^[\[\]\|]+|[\[\]\|]+$')

def clean_records(records):
    out = []
    dropped = 0
    for r in records:
        r = dict(r)
        r.pop('doc', None)
        if not r.get('nom_prenom'):
            dropped += 1
            continue
        # A row with neither sexe nor date of birth read is either an OCR
        # failure on that row, or a name fragment left over from a table row
        # that wrapped across a page break (the rest of that row's data
        # belongs to a page this excerpt doesn't include) -- in both cases
        # there is no complete, trustworthy record to show, so drop it
        # rather than surface a misleading near-empty entry.
        if not r.get('sexe') and not r.get('date_naissance'):
            dropped += 1
            continue
        for field in ('lieu_naissance', 'etablissement_origine', 'region_origine', 'nom_prenom'):
            if r.get(field):
                r[field] = BRACKET_RE.sub('', r[field]).strip()
        if not r.get('ecole_orientation'):
            r['ecole_orientation'] = "École non identifiée (page source manquante)"
        out.append(r)
    return out, dropped

def build_sections(records):
    order = []
    counts = {}
    pages = {}
    for r in records:
        key = r.get('ecole_orientation') or 'INCONNU'
        if key not in counts:
            order.append(key)
            counts[key] = 0
            pages[key] = set()
        counts[key] += 1
        pages[key].add(r['page'])
    lines = []
    for key in order:
        pg = sorted(pages[key])
        pgtxt = f"p.{pg[0]}-{pg[-1]}" if len(pg) > 1 else f"p.{pg[0]}"
        lines.append(f"{key} — {counts[key]} élève(s) ({pgtxt})")
    return lines, sum(counts.values())

def main():
    doc1, dropped1 = clean_records(load(f'{BASE}/doc1_records.json'))
    doc2, dropped2 = clean_records(load(f'{BASE}/doc2_records.json'))

    os.makedirs(REPO_DATA, exist_ok=True)
    with open(f'{REPO_DATA}/2026-2027.json', 'w', encoding='utf-8') as f:
        json.dump(doc1, f, ensure_ascii=False, indent=0)
    with open(f'{REPO_DATA}/2025-2026.json', 'w', encoding='utf-8') as f:
        json.dump(doc2, f, ensure_ascii=False, indent=0)

    sections1, total1 = build_sections(doc1)
    sections2, total2 = build_sections(doc2)

    meta = {
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "datasets": [
            {
                "annee": "2026-2027",
                "description": (
                    f"Décision N°0131/MEN/A/PL/SG/DL/DGPQ/DECOS du 07/09/2026. "
                    f"Document officiel de 50 pages ; seules les pages 1 à 24 étaient disponibles pour cet outil "
                    f"({total1} élèves, {len(sections1)} établissement(s) d'accueil couverts sur ces pages). "
                    f"Les pages 25 à 50 (sections suivantes de la liste alphabétique / autres écoles) ne sont "
                    f"pas encore intégrées. {dropped1} ligne(s) illisible(s) par la reconnaissance automatique "
                    f"ont été exclues plutôt que d'afficher une donnée erronée."
                ),
                "sections": sections1,
            },
            {
                "annee": "2025-2026",
                "description": (
                    f"Décision d'orientation post-BEPC 2025-2026 (« Suite 3 et fin du document »). "
                    f"Document officiel de 54 pages ; seules les pages 37 à 54 (fin du document) étaient "
                    f"disponibles pour cet outil ({total2} élèves, {len(sections2)} établissement(s) d'accueil "
                    f"couverts sur ces pages). Les pages 1 à 36 (début de la liste) ne sont pas encore intégrées : "
                    f"les premiers élèves de cette liste (avant la première école identifiée ci-dessous) sont "
                    f"affichés avec une école d'accueil « non identifiée » car leur section apparaît sur une page "
                    f"manquante. {dropped2} ligne(s) illisible(s) par la reconnaissance automatique ont été "
                    f"exclues plutôt que d'afficher une donnée erronée."
                ),
                "sections": sections2,
            },
        ],
    }
    with open(f'{REPO_DATA}/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print("doc1 records:", len(doc1), "sections:", len(sections1))
    print("doc2 records:", len(doc2), "sections:", len(sections2))

if __name__ == '__main__':
    main()
