import sys, os, re, json, glob, time
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import process_page
import cv2
import pytesseract

def clean_text(t):
    t = t.strip()
    t = re.sub(r'^[|!;:.\-\s]+', '', t)
    t = re.sub(r'[|!;:\-\s]+$', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def ocr_generic(img, x0, y0, x1, y1, psm=6, pad=5):
    h, w = img.shape[:2]
    x0i = max(0, int(x0)+pad); x1i = min(w, int(x1)-pad)
    y0i = max(0, int(y0)+pad); y1i = min(h, int(y1)-pad)
    if x1i <= x0i or y1i <= y0i:
        return ""
    crop = img[y0i:y1i, x0i:x1i]
    crop = cv2.resize(crop, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    cfg = f'--psm {psm} -c preserve_interword_spaces=1'
    txt = pytesseract.image_to_string(crop, lang='fra', config=cfg)
    return clean_text(txt)

def ocr_sexe(img, x0, y0, x1, y1, pad=5):
    h, w = img.shape[:2]
    x0i = max(0, int(x0)+pad); x1i = min(w, int(x1)-pad)
    y0i = max(0, int(y0)+pad); y1i = min(h, int(y1)-pad)
    if x1i <= x0i or y1i <= y0i:
        return ""
    crop = img[y0i:y1i, x0i:x1i]
    crop = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    for psm in (8, 7, 10, 6):
        txt = pytesseract.image_to_string(th, lang='fra', config=f'--psm {psm} -c tessedit_char_whitelist=MF')
        txt = txt.strip()
        if txt in ('M', 'F'):
            return txt
    return clean_text(txt)

DATE_RE = re.compile(r'(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})\s*(?:[àaA][\s]|,)?\s*(.*)')

def split_date_lieu(txt):
    txt = clean_text(txt)
    m = DATE_RE.match(txt)
    if m:
        date = m.group(1).replace('-', '/').replace('.', '/')
        lieu = m.group(2).strip(' àaA,')
        return date, lieu
    return "", txt

def is_header_row(nom_txt, n_txt):
    t = (nom_txt or "").lower()
    return 'nom' in t and 'pr' in t

def clean_header(text):
    m = re.match(r'^\s*(\d{1,2})[\.\)]\s+(.*)$', text)
    name = clean_text(m.group(2)) if m else clean_text(text)
    if name.count('(') > name.count(')'):
        name = re.sub(r'\(([A-ZÉÈÀÂÎÔÛÇ]{2,8})(?=\s|$)', r'(\1)', name, count=1)
    return (int(m.group(1)), name) if m else (None, name)

def process_document(doc_id, png_files, out_path, log):
    records = []
    current_header = None
    last_header_num = None
    section_counters = {}
    # a section header can appear at the very bottom of a page, with that
    # section's table only starting on the *next* page (the current page
    # ends with the tail of the previous section instead) -- carry such
    # headers forward so they aren't silently dropped, which would also
    # break the sequential-number validation for every section after it
    carried_headers = []
    for pi, png in enumerate(png_files):
        page_num = pi + 1
        t0 = time.time()
        gray_r, blocks, headers, angle = process_page(png)
        prev_y1 = -1
        for bi, b in enumerate(blocks):
            cols = b['cols']; rows = b['rows']
            if len(cols) != 7:
                log(f"[{doc_id}] WARNING page {page_num}: expected 7 column boundaries, got {len(cols)} ({cols}) -- skipping block")
                continue
            block_top = rows[0]
            # find nearest header above this block and below prev block's bottom
            candidates = [h for h in headers if prev_y1 - 5 <= h['y'] <= block_top + 5]
            if bi == 0:
                candidates = carried_headers + candidates
            # Only accept a header whose section number is the very first one
            # seen, or exactly one more than the last accepted section number
            # (sections are numbered sequentially) -- this rejects OCR noise
            # from data rows that happen to match "<digits>. Capital..." by
            # coincidence. A candidate with num=None (its leading "N." was
            # dropped by OCR entirely) is accepted only as a fallback, and
            # only once we have a baseline to count up from -- its section
            # number is then inferred positionally rather than read.
            numbered_valid = [
                h for h in candidates
                if h['num'] is not None and (last_header_num is None or h['num'] == last_header_num + 1)
            ]
            if numbered_valid:
                chosen = max(numbered_valid, key=lambda h: h['y'])
                current_header = chosen['text']
                last_header_num = chosen['num']
            elif last_header_num is not None:
                unnumbered = [h for h in candidates if h['num'] is None]
                if unnumbered:
                    chosen = max(unnumbered, key=lambda h: h['y'])
                    current_header = chosen['text']
                    last_header_num = last_header_num + 1
            hdr_num, hdr_name = clean_header(current_header) if current_header else (None, None)
            start_ri = 0
            first_nom = ocr_generic(gray_r, cols[1], rows[0], cols[2], rows[1])
            first_n = ocr_generic(gray_r, cols[0], rows[0], cols[1], rows[1], psm=7)
            if is_header_row(first_nom, first_n):
                start_ri = 1
            for ri in range(start_ri, len(rows)-1):
                y0, y1 = rows[ri], rows[ri+1]
                nom = ocr_generic(gray_r, cols[1], y0, cols[2], y1)
                if not nom:
                    continue
                sexe = ocr_sexe(gray_r, cols[2], y0, cols[3], y1)
                date_lieu = ocr_generic(gray_r, cols[3], y0, cols[4], y1)
                date, lieu = split_date_lieu(date_lieu)
                etab = ocr_generic(gray_r, cols[4], y0, cols[5], y1)
                region = ocr_generic(gray_r, cols[5], y0, cols[6], y1)
                key = hdr_name or "INCONNU"
                section_counters[key] = section_counters.get(key, 0) + 1
                records.append({
                    'doc': doc_id,
                    'page': page_num,
                    'section_num': hdr_num,
                    'ecole_orientation': hdr_name,
                    'num': section_counters[key],
                    'nom_prenom': nom,
                    'sexe': sexe,
                    'date_naissance': date,
                    'lieu_naissance': lieu,
                    'etablissement_origine': etab,
                    'region_origine': region,
                })
            prev_y1 = rows[-1]
        # any header on this page that sits below the last block (or, on a
        # page with no usable block at all, any header at all) belongs to a
        # table that hasn't started yet -- hand it to the next page
        carried_headers = [h for h in headers if h['y'] > prev_y1 + 5]
        log(f"[{doc_id}] page {page_num}/{len(png_files)} blocks={len(blocks)} headers={[h['text'] for h in headers]} records_so_far={len(records)} time={time.time()-t0:.1f}s")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=1)
    return records

if __name__ == "__main__":
    # Expects page images rendered at 300dpi, e.g.:
    #   pdftoppm -png -r 300 decision-2026-2027.pdf tools/pages/doc1
    #   pdftoppm -png -r 300 decision-2025-2026.pdf tools/pages/doc2
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(f'{base}/raw', exist_ok=True)
    doc1_files = sorted(glob.glob(f'{base}/pages/doc1-*.png'))
    doc2_files = sorted(glob.glob(f'{base}/pages/doc2-*.png'))
    def log(msg):
        print(msg, flush=True)
    if doc1_files:
        process_document('2026-2027', doc1_files, f'{base}/raw/doc1_records.json', log)
    if doc2_files:
        process_document('2025-2026', doc2_files, f'{base}/raw/doc2_records.json', log)
    if not doc1_files and not doc2_files:
        print(f"No page images found under {base}/pages/ -- see comment above for how to render them.")
