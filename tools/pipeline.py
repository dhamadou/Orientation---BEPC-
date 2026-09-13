import cv2
import numpy as np
import pytesseract
import re
import json
import sys
import os

def get_bw(gray):
    return cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -2)

def line_mask(bw, horiz=True, size_div=30):
    h, w = bw.shape
    if horiz:
        size = max(10, w // size_div)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
    else:
        size = max(10, h // size_div)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    m = cv2.erode(bw, kernel)
    m = cv2.dilate(m, kernel)
    return m

def estimate_skew(horiz):
    lines = cv2.HoughLinesP(horiz, 1, np.pi/1800, threshold=300, minLineLength=horiz.shape[1]*0.4, maxLineGap=20)
    if lines is None:
        return 0.0
    lines = lines.reshape(-1, 4)
    angles = []
    for x1,y1,x2,y2 in lines:
        if x2 == x1:
            continue
        ang = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(ang) < 5:
            angles.append(ang)
    if not angles:
        return 0.0
    return float(np.median(angles))

def components(mask, min_len, axis):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if axis == 'h' and ww > min_len:
            out.append((y + hh/2.0, x, x+ww))
        elif axis == 'v' and hh > min_len:
            out.append((x + ww/2.0, y, y+hh))
    out.sort(key=lambda t: t[0])
    merged = []
    for c, a, b in out:
        if merged and abs(c - merged[-1][0]) < 8:
            merged[-1] = ((merged[-1][0]+c)/2.0, min(merged[-1][1], a), max(merged[-1][2], b))
        else:
            merged.append((c, a, b))
    return merged

def raw_v_components(mask, min_len):
    # unlike components(), does NOT merge nearby x-centers -- two separate
    # tables stacked on the same page often share near-identical column x
    # positions, and merging by x alone would wrongly union their y-ranges
    # into one fake continuous line spanning both tables
    h, w = mask.shape
    # scan artifacts (binder shadow, torn/curled corner, scanner-bed edge)
    # show up as a spurious vertical blob hugging the very edge of the page,
    # at heights ranging from short fragments to tall near-full-page blobs --
    # observed within ~60px of the true edge across many pages. A genuine
    # table's outermost ruled column, on every page checked, sits at least
    # ~85px in from the edge (printed layouts always keep a margin), so a
    # pure position cutoff cleanly separates the two without depending on
    # height at all.
    edge_margin = 80
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        cx = x + ww / 2.0
        if hh <= min_len:
            continue
        if cx < edge_margin or cx > w - edge_margin:
            continue
        out.append((cx, y, y + hh))
    return out

def find_table_blocks(gray):
    bw = get_bw(gray)
    horiz = line_mask(bw, True, 30)
    angle = estimate_skew(horiz)
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    gray_r = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    bw_r = get_bw(gray_r)
    horiz_r = line_mask(bw_r, True, 30)
    vert_r = line_mask(bw_r, False, 60)

    vcomps = raw_v_components(vert_r, h*0.05)  # (x, y0, y1), un-merged
    # cluster vcomps into table blocks by overlapping y-range
    vcomps_sorted = sorted(vcomps, key=lambda t: t[1])
    blocks = []
    for x, y0, y1 in vcomps_sorted:
        placed = False
        for blk in blocks:
            by0, by1 = blk['yr']
            # overlap check
            if not (y1 < by0 - 30 or y0 > by1 + 30):
                blk['cols'].append((x, y0, y1))
                blk['yr'] = (min(by0, y0), max(by1, y1))
                placed = True
                break
        if not placed:
            blocks.append({'yr': (y0, y1), 'cols': [(x, y0, y1)]})

    result_blocks = []
    for blk in blocks:
        cols = sorted(set(round(x) for x, _, _ in blk['cols']))
        # merge close/spurious col lines: a real table has 6 columns, so any
        # boundary closer than ~40px to its neighbour is noise (double-detected
        # ruling line), not a genuine extra column
        merged_cols = []
        for x in cols:
            if merged_cols and abs(x - merged_cols[-1]) < 40:
                continue
            merged_cols.append(x)
        if len(merged_cols) < 5:
            continue  # not a real 6-col table
        y0, y1 = blk['yr']
        rowcomps = components(horiz_r, w*0.3, 'h')
        # Recover a missing outer column boundary (a genuinely faded ruling,
        # or one this function's own edge-artifact filter had to sacrifice)
        # from the table's own top/bottom border, which always spans its
        # true full width regardless of which vertical dividers survived.
        # Gated strictly on the column *count* (not on whether row-matching
        # already succeeded): a table missing its outer ruling still has row
        # lines wide enough to satisfy the row-match below using the fewer
        # columns alone, so that check can't be trusted to detect the gap.
        if len(merged_cols) < 7:
            nearby_rows_wide = [(c, a, b) for c, a, b in rowcomps
                                 if y0 - 15 <= c <= y1 + 15
                                 and a <= merged_cols[0] + 60 and b >= merged_cols[-1] - 60]
            if nearby_rows_wide:
                true_left = min(a for _, a, _ in nearby_rows_wide)
                true_right = max(b for _, _, b in nearby_rows_wide)
                # bound the recovered gap to one plausible missing column's
                # width so an unrelated, much wider stray line (e.g. a page
                # separator rule) can't get mistaken for the table's edge
                if 40 < merged_cols[0] - true_left < 300:
                    merged_cols.insert(0, true_left)
                if 40 < true_right - merged_cols[-1] < 300:
                    merged_cols.append(true_right)
        rows = [c for c, a, b in rowcomps if y0-15 <= c <= y1+15 and a <= merged_cols[0]+20 and b >= merged_cols[-1]-20]
        rows = sorted(set(rows))
        if len(rows) < 2:
            continue
        result_blocks.append({'cols': merged_cols, 'rows': rows, 'y0': y0, 'y1': y1})

    result_blocks.sort(key=lambda b: b['y0'])
    return gray_r, result_blocks, angle

def ocr_cell(img, x0, y0, x1, y1, psm=6, pad=4):
    h, w = img.shape[:2]
    x0 = max(0, int(x0)+pad); x1 = min(w, int(x1)-pad)
    y0 = max(0, int(y0)+pad); y1 = min(h, int(y1)-pad)
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cfg = f'--psm {psm} -c preserve_interword_spaces=1'
    txt = pytesseract.image_to_string(crop, lang='fra', config=cfg)
    txt = ' '.join(txt.split())
    return txt.strip()

HEADER_RE = re.compile(r"^\s*(\d{1,2})[\.\)]\s+([A-ZÀÂÉÈÊÎÔÛÇÏÜ0-9][A-Za-zÀ-ÿ0-9'’\-\(\)\./ ]{4,})$")

# a school-name title occasionally scans with its leading "N." dropped
# entirely by OCR (the numeral is small and isolated) -- this catches the
# title text alone so extract_all.py can still recognize the section
# started, inferring the number positionally (previous section's number + 1)
# instead of relying on OCR having read the digit. Deliberately stricter
# than HEADER_RE's name part (long, multi-word, no lowercase run) since,
# without a number to anchor on, it must not fire on a short wrapped
# continuation fragment (e.g. "D'AGADEZ") or ordinary paragraph text.
TITLE_LINE_RE = re.compile(r"^[A-ZÀÂÉÈÊÎÔÛÇÏÜ0-9][A-Z0-9À-Ü'’\-\(\)\./ ]{14,90}$")

TABLE_KEYWORDS_RE = re.compile(r'\b(noms?|pr[ée]noms?|sexe|naissance|etablissement|[ée]tablissement|r[ée]gion)\b', re.I)

def find_headers(gray_r):
    data = pytesseract.image_to_data(gray_r, lang='fra', config='--psm 4', output_type=pytesseract.Output.DICT)
    lines = {}
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        if key not in lines:
            lines[key] = {'words': [], 'top': data['top'][i], 'bottom': data['top'][i]+data['height'][i]}
        lines[key]['words'].append(txt)
        lines[key]['top'] = min(lines[key]['top'], data['top'][i])
        lines[key]['bottom'] = max(lines[key]['bottom'], data['top'][i]+data['height'][i])

    all_lines = sorted(
        [{'text': ' '.join(v['words']), 'top': v['top'], 'bottom': v['bottom']} for v in lines.values()],
        key=lambda l: l['top']
    )

    def merge_continuation(idx, line):
        text = line['text']
        line_height = line['bottom'] - line['top']
        cur_bottom = line['bottom']
        for nxt in all_lines[idx+1:idx+3]:
            gap = nxt['top'] - cur_bottom
            if gap < 0 or gap > line_height * 1.4:
                break
            if HEADER_RE.match(nxt['text']):
                break
            if TABLE_KEYWORDS_RE.search(nxt['text']):
                break
            if len(nxt['text']) > 45 or len(nxt['text'].split()) > 6:
                break
            text = text + ' ' + nxt['text']
            cur_bottom = nxt['bottom']
        return text, cur_bottom

    headers = []
    claimed = set()
    for idx, line in enumerate(all_lines):
        m = HEADER_RE.match(line['text'])
        if not m:
            continue
        # merge a short continuation line immediately below (wrapped school name,
        # e.g. "3. LYCEE ... (LPHT)" then "D'AGADEZ" on the next line)
        text, cur_bottom = merge_continuation(idx, line)
        headers.append({'num': int(m.group(1)), 'text': text, 'y': (line['top']+cur_bottom)/2.0})
        claimed.add(idx)

    # fallback: a title's leading "N." can be dropped entirely by OCR (the
    # numeral is small and isolated on the line) -- catch the bare title text
    # too, with no number, so the caller can still recognize a new section
    # started and infer its number positionally (previous + 1) instead of
    # depending on the digit having been read at all
    for idx, line in enumerate(all_lines):
        if idx in claimed:
            continue
        if not TITLE_LINE_RE.match(line['text']):
            continue
        if TABLE_KEYWORDS_RE.search(line['text']):
            continue
        text, cur_bottom = merge_continuation(idx, line)
        headers.append({'num': None, 'text': text, 'y': (line['top']+cur_bottom)/2.0})

    headers.sort(key=lambda h: h['y'])
    return headers

def process_page(png_path):
    img = cv2.imread(png_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_r, blocks, angle = find_table_blocks(gray)
    headers = find_headers(gray_r)
    return gray_r, blocks, headers, angle

if __name__ == "__main__":
    png = sys.argv[1]
    gray_r, blocks, headers, angle = process_page(png)
    print("angle", angle)
    print("headers", headers)
    for b in blocks:
        print("block rows", len(b['rows']), "cols", b['cols'])
