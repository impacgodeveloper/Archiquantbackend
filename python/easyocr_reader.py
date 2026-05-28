import cv2, easyocr, re, json, numpy as np, math, sys
from pdf2image import convert_from_path
 
# ─────────────────────────────────────────────
MIN_CONF   = 0.28
reader     = easyocr.Reader(['en'], gpu=False)
input_path = sys.argv[1]
 
# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
if input_path.lower().endswith(".pdf"):
    pages = convert_from_path(input_path, dpi=300)
else:
    raw = cv2.imread(input_path)
    pages = [raw] if raw is not None else []
 
raw_images = []
ocr_tokens = []   # {"text", "x", "y"}
 
def preprocess(img):
    if not isinstance(img, np.ndarray):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th
 
for page in pages:
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
    raw_images.append(image)
    raw = reader.readtext(preprocess(image))
    # Sort row-first (top→bottom, left→right)
    for (bbox, text, prob) in sorted(raw, key=lambda r: (round(r[0][0][1] / 25), r[0][0][0])):
        if prob < MIN_CONF:
            continue
        x = int(sum(p[0] for p in bbox) / 4)
        y = int(sum(p[1] for p in bbox) / 4)
        ocr_tokens.append({"text": text.upper().strip(), "x": x, "y": y})
 
img_h = raw_images[0].shape[0] if raw_images else 1000
img_w = raw_images[0].shape[1] if raw_images else 1000
 
# ─────────────────────────────────────────────
#  STEP 1 — CLEAN & NORMALIZE
#  Fix every known OCR misread of the standard format
# ─────────────────────────────────────────────
def normalize(t):
    t = t.upper().strip()
 
    # ── Dimension separator → X ──────────────────────────────────────────
    for c in ['×', '\xd7', '✕', '*', '·', 'X']:
        t = t.replace(c, 'X')
 
    # ── Equals/dash normalization ─────────────────────────────────────────
    # T-9 → T=9,  H-10 → H=10,  D-3X7 → D=3X7,  W-4X4 → W=4X4
    t = re.sub(r'\b([THDW])-(\d)', r'\1=\2', t)
 
    # ── Digit/letter confusion in decimal specs ───────────────────────────
    # T=O.10 → T=0.10,  H-2.1O → H=2.10
    t = re.sub(r'([=\-])O\.(\d)', r'\g<1>0.\2', t)
    t = re.sub(r'(\d\.\d*)O\b',   r'\g<1>0',   t)
 
    # ── Fix "0" read as "D" in door labels ───────────────────────────────
    # 01 → D1,  07 → D7,  0(MAIN) → D(MAIN)
    t = re.sub(r'^0\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
    t = re.sub(r'^0\s*\(',             'D(',       t)
    t = re.sub(r'^0(\d+)$',            r'D\1',     t)
 
    # ── Fix W label digit confusion ───────────────────────────────────────
    t = re.sub(r'^W([AQ])$',  'W4', t)   # WA → W4
    t = re.sub(r'^W([S5])$',  'W5', t)   # WS → W5
    t = re.sub(r'^W([B8])$',  'W8', t)   # WB → W8
    t = re.sub(r'^W([G6])$',  'W6', t)   # WG → W6
    t = re.sub(r'^W([Il1])$', 'W1', t)   # WI/Wl → W1
 
    # ── Fix ZONE digit confusion ──────────────────────────────────────────
    t = re.sub(r'^ZONE\s+S([A-Z]?)$', r'ZONE 5\1', t)  # ZONE SA → ZONE 5A
    t = re.sub(r'^ZONE\s+I([A-Z]?)$', r'ZONE 1\1', t)  # ZONE IA → ZONE 1A
    t = re.sub(r'^ZONE\s+l([A-Z]?)$', r'ZONE 1\1', t)  # ZONE lA → ZONE 1A
    t = re.sub(r'^ZONE\s+G([A-Z]?)$', r'ZONE 6\1', t)  # ZONE GA → ZONE 6A
 
    # ── Fix garbled IW/EW ─────────────────────────────────────────────────
    t = re.sub(r'\bEW,\b', 'EW', t)
    t = t.replace('1W', 'IW').replace('lW', 'IW').replace('|W', 'IW')
 
    # ── Fix garbled T/H specs ─────────────────────────────────────────────
    t = re.sub(r'\bTZ(\d)',  r'T=\1', t)   # TZ4 → T=4
    t = re.sub(r'H-L(\d)',   r'H=1\1', t)  # H-L0 → H=10
    t = t.replace('T-G', 'T=9').replace('T=G', 'T=9')
 
    # ── Fix dimension letter confusion: W=AX4 → W=4X4 ────────────────────
    t = re.sub(r'\b([DW]=\d*)([A-OQ-Z])(X)', lambda m: m.group(1) + '4' + m.group(3), t)
    t = re.sub(r'(X)([A-OQ-Z])\b',           r'\g<1>4', t)
 
    # ── Whitespace ────────────────────────────────────────────────────────
    t = re.sub(r'\s+', ' ', t).strip()
    return t
 
for tok in ocr_tokens:
    tok["text"] = normalize(tok["text"])
 
# ─────────────────────────────────────────────
#  STEP 2 — MERGE SPLIT TOKENS
#  OCR sometimes splits "ZONE" + "1" or "W=4" + "X4"
# ─────────────────────────────────────────────
def merge_tokens(tokens):
    out  = []
    used = set()
 
    for i, tok in enumerate(tokens):
        if i in used:
            continue
        cur = tok["text"]
 
        if i + 1 < len(tokens) and i + 1 not in used:
            nxt    = tokens[i + 1]
            dy     = abs(nxt["y"] - tok["y"])
            dx     = abs(nxt["x"] - tok["x"])
            same_row = dy < 35
            close    = dx < 220
 
            if same_row and close:
                n = nxt["text"]
                mx = (tok["x"] + nxt["x"]) // 2
                my = (tok["y"] + nxt["y"]) // 2
 
                # "ZONE" + single digit/label → "ZONE 1", "ZONE 1A"
                if cur == "ZONE" and re.match(r'^\d[A-Z]?$', n):
                    out.append({"text": "ZONE " + n, "x": mx, "y": my})
                    used.add(i); used.add(i + 1); continue
 
                # "ZONE N" + single letter → "ZONE 1A"
                if re.match(r'^ZONE \d+$', cur) and re.match(r'^[A-Z]$', n):
                    out.append({"text": cur + n, "x": mx, "y": my})
                    used.add(i); used.add(i + 1); continue
 
                # Dimension merge: "W=4" + "X4" or "3.50" + "X4.25"
                combined = cur + n
                if re.search(r'[WD]=?\d.*X\d', combined):
                    out.append({"text": combined, "x": mx, "y": my})
                    used.add(i); used.add(i + 1); continue
 
                # "IW" + "T=4..." or "EW" + "T=9..."
                if re.match(r'^(IW|EW)$', cur) and re.search(r'T[=\-]\d', n):
                    out.append({"text": cur + " " + n, "x": mx, "y": my})
                    used.add(i); used.add(i + 1); continue
 
        out.append(tok)
        used.add(i)
    return out
 
ocr_tokens = merge_tokens(ocr_tokens)
 
# Second pass: rescue orphan "ZONE" tokens (number on different row)
orphans    = [t for t in ocr_tokens if t["text"] == "ZONE"]
digits     = [t for t in ocr_tokens if re.match(r'^\d[A-Z]?$', t["text"])]
used_digit = set()
 
for orphan in orphans:
    best, bd = None, 999
    for i, d in enumerate(digits):
        if i in used_digit:
            continue
        dist_val = math.hypot(orphan["x"] - d["x"], orphan["y"] - d["y"])
        if dist_val < 300 and dist_val < bd:
            # Ensure this digit isn't already claimed by another ZONE label
            claimed = any(t["text"] == "ZONE " + d["text"] for t in ocr_tokens)
            if not claimed:
                bd, best = dist_val, (i, d)
    if best:
        idx, d = best
        ocr_tokens.append({
            "text": "ZONE " + d["text"],
            "x": (orphan["x"] + d["x"]) // 2,
            "y": (orphan["y"] + d["y"]) // 2,
        })
        used_digit.add(idx)
 
# Remove legend text (bottom 20% of image)
LEGEND_KW = [
    "= EXTERNAL WALL", "= INTERNAL WALL", "EW =", "IW =", "EW:",
    "W = WINDOW", "D = DOOR", "T = THICKNESS", "H = HEIGHT",
    "ROOM TINTS", "BEDROOMS / STUDY", "BATHROOMS / LIVING",
]
ocr_tokens = [
    t for t in ocr_tokens
    if not (t["y"] > img_h * 0.80 and any(kw in t["text"] for kw in LEGEND_KW))
]
 
# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])
 
def is_dup(x, y, seen, th=50):
    return any(abs(x - px) < th and abs(y - py) < th for px, py in seen)
 
def nearby(base, r=260):
    return [t for t in ocr_tokens if dist(base, t) < r]
 
def find_num(tokens, pat):
    for t in tokens:
        m = re.search(pat, t["text"])
        if m:
            try: return float(m.group(1))
            except: pass
    return None
 
def find_dir(tokens):
    for t in tokens:
        for d in ["NORTH", "SOUTH", "EAST", "WEST", "MIDDLE", "CENTER"]:
            if d in t["text"]: return d.lower()
    return None
 
def pos_from_xy(x, y):
    m = 0.15
    if   y < img_h * m:       return "north"
    elif y > img_h * (1 - m): return "south"
    elif x < img_w * m:       return "west"
    elif x > img_w * (1 - m): return "east"
    return "inner"
 
def parse_ft_in(text):
    """12'0" → 12,  14'6" → 15,  6' → 6"""
    if not text: return None
    text = str(text).replace(" ", "")
    m = re.search(r"(\d+)'", text)
    if not m: return None
    feet = int(m.group(1))
    im   = re.search(r"'(\d+)", text)
    frac = re.search(r"(\d+)/(\d+)", text)
    inches = int(im.group(1)) if im else 0
    frac_v = int(frac.group(1)) / int(frac.group(2)) if frac else 0
    if inches > 12: inches = int(str(inches)[0])
    return math.ceil(feet + (inches + frac_v) / 12)
 
# ─────────────────────────────────────────────
#  STEP 3 — EXTERNAL WALLS  (EW T=9 H=10)
# ─────────────────────────────────────────────
EW_RE = re.compile(
    r'^EW[\s,]'            # starts with EW
    r'|^EW T[=\-]\d'       # EW T=9...
    r'|^EW$'               # standalone EW
    r'|^EW\d+[A-Z]?$'      # EW1, EW2
)
 
ew_anchors, ew_seen = [], []
for t in ocr_tokens:
    if EW_RE.match(t["text"]):
        if not is_dup(t["x"], t["y"], ew_seen):
            ew_anchors.append(dict(t))
            ew_seen.append((t["x"], t["y"]))
 
external_walls = {}
for i, anchor in enumerate(ew_anchors):
    eid = f"ew{i+1}"
    nb  = nearby(anchor, 300)
    tv  = find_num(nb, r'T[=\s](\d+\.?\d*)')
    hv  = find_num(nb, r'H[=\s](\d+\.?\d*)')
    pos = find_dir(nb) or pos_from_xy(anchor["x"], anchor["y"])
    external_walls[eid] = {
        "id": eid, "position": pos,
        "length_ft": None,
        "thickness_in": int(tv) if tv else 9,
        "height_ft":    int(hv) if hv else 10,
        "connected": {"windows": [], "doors": [], "internal_walls": []},
    }
    anchor["id"] = eid
 
# ─────────────────────────────────────────────
#  STEP 4 — INTERNAL WALLS  (IW T=4 H=10)
# ─────────────────────────────────────────────
IW_RE = re.compile(
    r'^IW[\s,\(]'
    r'|^IW T[=\-]\d'
    r'|^IW$'
    r'|^IW\d+[A-Z]?$'
)
 
iw_anchors, iw_seen = [], []
for t in ocr_tokens:
    if IW_RE.match(t["text"]):
        if not is_dup(t["x"], t["y"], iw_seen):
            iw_anchors.append(dict(t))
            iw_seen.append((t["x"], t["y"]))
 
internal_walls = {}
for i, anchor in enumerate(iw_anchors):
    iwid = f"iw{i+1}"
    nb   = nearby(anchor, 300)
    tv   = find_num(nb, r'T[=\s](\d+\.?\d*)')
    hv   = find_num(nb, r'H[=\s](\d+\.?\d*)')
    pos  = find_dir(nb)
    # Connect to nearest 2 EWs
    conn = []
    for ea in sorted(ew_anchors, key=lambda a: dist(anchor, a))[:2]:
        conn.append(ea["id"])
        if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
            external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
    internal_walls[iwid] = {
        "id": iwid, "position": pos,
        "thickness_in": int(tv) if tv else 4,
        "height_ft":    int(hv) if hv else 10,
        "length_ft": None,
        "connects_external": conn,
        "connected": {"doors": [], "windows": []},
    }
    anchor["id"] = iwid
 
# Fallback: if no EW/IW found use OpenCV
if not ew_anchors and raw_images:
    img   = raw_images[0]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=img_w//10, maxLineGap=20)
    ec = ic = 0
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            ang = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            ln  = math.hypot(x2-x1, y2-y1)
            bm  = 0.12
            if ang < 15 or ang > 165:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if cy < img_h*bm or cy > img_h*(1-bm):
                    ec += 1; eid = f"ew{ec}"
                    pos = "north" if cy < img_h/2 else "south"
                    external_walls[eid] = {"id":eid,"position":pos,"length_ft":round(ln/(img_w/40)),"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
                    ew_anchors.append({"x":cx,"y":cy,"id":eid})
            elif 75 < ang < 105:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if cx < img_w*0.12 or cx > img_w*0.88:
                    ec += 1; eid = f"ew{ec}"
                    pos = "west" if cx < img_w/2 else "east"
                    external_walls[eid] = {"id":eid,"position":pos,"length_ft":round(ln/(img_h/40)),"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
                    ew_anchors.append({"x":cx,"y":cy,"id":eid})
 
# ─────────────────────────────────────────────
#  STEP 5 — DOORS  (D=3×7, D=4×7)
# ─────────────────────────────────────────────
DOOR_RE = re.compile(r'^D[=\-\s]?(\d+\.?\d*)X(\d+\.?\d*)$')
DOOR_NAMED = re.compile(r'^D\(MAIN\)$|^D\(EXT\)$|^D\d+$|^DOOR\d*$')
 
def nearest_wall(x, y):
    best, bd = None, 999999
    for a in ew_anchors + iw_anchors:
        d = math.hypot(x - a["x"], y - a["y"])
        if d < bd: bd, best = d, a["id"]
    return best
 
def find_nearby_dim(base, r=300):
    for t in sorted([t for t in ocr_tokens if dist(base, t) < r], key=lambda t: dist(base, t)):
        m = re.search(r'(\d+\.?\d*)\s*X\s*(\d+\.?\d*)', t["text"])
        if m:
            w, h = float(m.group(1)), float(m.group(2))
            if 0.5 <= w <= 20 and 0.5 <= h <= 20:
                return w, h
        m = re.search(r"(\d+'\d*\"?)\s*X\s*(\d+'\d*\"?)", t["text"])
        if m:
            w = parse_ft_in(m.group(1))
            h = parse_ft_in(m.group(2))
            if w and h: return float(w), float(h)
    return None, None
 
door_list, door_seen = [], []
for t in ocr_tokens:
    tx = t["text"]
    m  = DOOR_RE.match(tx)
    if m and not is_dup(t["x"], t["y"], door_seen):
        door_list.append({"id": f"d{len(door_list)+1}", "x": t["x"], "y": t["y"],
                           "width_ft": float(m.group(1)), "height_ft": float(m.group(2))})
        door_seen.append((t["x"], t["y"]))
    elif DOOR_NAMED.match(tx) and not is_dup(t["x"], t["y"], door_seen):
        dw, dh = find_nearby_dim(t)
        door_list.append({"id": f"d{len(door_list)+1}", "x": t["x"], "y": t["y"],
                           "width_ft": dw or (4.0 if "MAIN" in tx or "EXT" in tx else 3.0),
                           "height_ft": dh or 7.0})
        door_seen.append((t["x"], t["y"]))
 
doors_out = {}
for d in door_list:
    ow = nearest_wall(d["x"], d["y"])
    doors_out[d["id"]] = {"id": d["id"], "width_ft": d["width_ft"],
                           "height_ft": d["height_ft"], "on_wall": ow}
    if ow:
        wl = external_walls.get(ow) or internal_walls.get(ow)
        if wl and d["id"] not in wl["connected"]["doors"]:
            wl["connected"]["doors"].append(d["id"])
 
# ─────────────────────────────────────────────
#  STEP 6 — WINDOWS  (W=4×4, W=4'6"×4')
# ─────────────────────────────────────────────
WIN_DIM_RE  = re.compile(r"^W[=\-]?(\d+\.?\d*|'\d*\"?)X(\d+\.?\d*|'\d*\"?)$")
WIN_FT_RE   = re.compile(r"^W[=\-]?(\d+['']\d*[\"']?)\s*X\s*(\d+['']\d*[\"']?)$")
WIN_LABEL_RE= re.compile(r'^W\d+$|^WIN\d*$')
 
window_list, win_seen = [], []
for t in ocr_tokens:
    tx = t["text"]
    # W=4X4 or W=4'6"X4'
    m  = WIN_DIM_RE.match(tx) or WIN_FT_RE.match(tx)
    if m and not is_dup(t["x"], t["y"], win_seen):
        try:
            w = parse_ft_in(m.group(1)) or float(re.sub(r"[^0-9.]","",m.group(1)) or 4)
            h = parse_ft_in(m.group(2)) or float(re.sub(r"[^0-9.]","",m.group(2)) or 4)
        except: w, h = 4.0, 4.0
        window_list.append({"id": f"w{len(window_list)+1}", "x": t["x"], "y": t["y"],
                             "width_ft": float(w), "height_ft": float(h)})
        win_seen.append((t["x"], t["y"]))
    elif WIN_LABEL_RE.match(tx) and not is_dup(t["x"], t["y"], win_seen):
        wv, hv = find_nearby_dim(t)
        window_list.append({"id": f"w{len(window_list)+1}", "x": t["x"], "y": t["y"],
                             "width_ft": wv or 4.0, "height_ft": hv or 4.0})
        win_seen.append((t["x"], t["y"]))
 
windows_out = {}
for w in window_list:
    ow = nearest_wall(w["x"], w["y"])
    windows_out[w["id"]] = {"id": w["id"], "width_ft": w["width_ft"],
                             "height_ft": w["height_ft"], "on_wall": ow}
    if ow:
        wl = external_walls.get(ow) or internal_walls.get(ow)
        if wl and w["id"] not in wl["connected"]["windows"]:
            wl["connected"]["windows"].append(w["id"])
 
# ─────────────────────────────────────────────
#  STEP 7 — ZONES  (ZONE 1 / ZONE 1A)
# ─────────────────────────────────────────────
ZONE_RE = re.compile(r'^ZONE\s*\d+[A-Z]?$')
 
ROOM_KW = [
    "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL",
    "BALCONY","STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM",
    "C.BEDROOM","MASTER","DRAWING","UTILITY","GARAGE","PASSAGE",
    "WASH","BATH","FAMILY","PANTRY","LAUNDRY","TERRACE","VERANDAH",
    "ENTRANCE","FOYER","MBEDROOM","CBEDROOM","BEDROOM 2","BEDROOM 3",
    "LIVING ROOM","DINING ROOM","PRAYER","POWDER",
    # French
    "CHAMBRE","SALON","CUISINE","SALLE","BAIN","TOILETTE","WC",
    "COULOIR","ENTREE","BUREAU","SEJOUR","TERRASSE","DOUCHE","MAGASIN",
]
 
def nearest_room_label(base, r=400):
    for t in sorted(ocr_tokens, key=lambda t: dist(base, t)):
        if dist(base, t) > r: break
        for kw in ROOM_KW:
            if kw in t["text"]: return t["text"]
    return "UNKNOWN"
 
def zone_dimensions(base, r=500):
    """Read zone dimensions from nearby text."""
    for t in ocr_tokens:
        if dist(base, t) > r: continue
        tx = t["text"]
        # 14'0"X17'0"
        m = re.search(r"(\d+['']\d+[\"']?)\s*X\s*(\d+['']\d+[\"']?)", tx)
        if m:
            ww = parse_ft_in(m.group(1))
            ll = parse_ft_in(m.group(2))
            if ww and ll and 4 <= ww <= 150 and 4 <= ll <= 150:
                return min(ww, ll), max(ww, ll)
        # 14'0"17'0" (no separator)
        m = re.search(r"(\d+['']\d+)[\"']\s*(\d+['']\d+)", tx)
        if m:
            ww = parse_ft_in(m.group(1))
            ll = parse_ft_in(m.group(2))
            if ww and ll and 4 <= ww <= 150 and 4 <= ll <= 150:
                return min(ww, ll), max(ww, ll)
        # Simple NxM (feet, no fractions) — skip if looks like door/window
        m = re.search(r'(\d{1,3})\s*X\s*(\d{1,3})', tx)
        if m:
            ww, ll = int(m.group(1)), int(m.group(2))
            if 4 <= ww <= 100 and 4 <= ll <= 100:
                if not re.match(r'^[DW]=', tx):
                    return min(ww, ll), max(ww, ll)
    return None, None
 
# Auto pixel-per-foot from overall building label
def px_per_ft():
    for t in ocr_tokens:
        # "50'0"X36'0"" total building dims
        m = re.search(r"(\d{2,3})'0\"?\s*X\s*(\d{2,3})'0\"?", t["text"])
        if m:
            w_ft = float(m.group(1))
            if 20 <= w_ft <= 500:
                return img_w / w_ft
    # Fallback: use two zones with known dims to compute scale
    pts = []
    for t in ocr_tokens:
        if ZONE_RE.match(t["text"]):
            wft, lft = zone_dimensions(t)
            if wft and lft:
                pts.append((t, max(wft, lft)))
    if len(pts) >= 2:
        best_d, best_s = 0, None
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                ea, fa = pts[i]; eb, fb = pts[j]
                pd = math.hypot(ea["x"]-eb["x"], ea["y"]-eb["y"])
                ef = (fa + fb) / 1.5
                if pd > best_d and ef > 0:
                    s = pd / ef
                    if 5 < s < 600:
                        best_d, best_s = pd, s
        if best_s:
            return best_s
    return img_w / 40.0   # final fallback
 
SCALE = px_per_ft()
 
zones_out = {}
for t in ocr_tokens:
    if not ZONE_RE.match(t["text"]): continue
    zid  = re.sub(r'\s+', '', t["text"])
    name = nearest_room_label(t)
    wft, lft = zone_dimensions(t)
    area = round(wft * lft, 2) if wft and lft else None
 
    # Zone proximity radius in pixels
    if area:
        zone_side_px = math.sqrt(area) * SCALE
        zp = zone_side_px * 0.75
    else:
        zp = img_w * 0.28
 
    conn_ew = list(dict.fromkeys(
        a["id"] for a in ew_anchors if math.hypot(t["x"]-a["x"], t["y"]-a["y"]) < zp * 1.4
    ))
    conn_iw = list(dict.fromkeys(
        a["id"] for a in iw_anchors if math.hypot(t["x"]-a["x"], t["y"]-a["y"]) < zp * 1.2
    ))
    conn_d  = list(dict.fromkeys(
        d["id"] for d in door_list
        if math.hypot(t["x"]-d["x"], t["y"]-d["y"]) < zp
    ))
    conn_w  = list(dict.fromkeys(
        w["id"] for w in window_list
        if math.hypot(t["x"]-w["x"], t["y"]-w["y"]) < zp
    ))
 
    zones_out[zid] = {
        "id": zid, "label": name,
        "width_ft": wft, "length_ft": lft, "area_sqft": area,
        "connected_external_walls": conn_ew,
        "connected_internal_walls": conn_iw,
        "total_walls_connected": len(conn_ew) + len(conn_iw),
        "doors": conn_d, "windows": conn_w,
    }
 
# ─────────────────────────────────────────────
#  STEP 8 — DEDUP + SUMMARY
# ─────────────────────────────────────────────
all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])
 
for z in zones_out.values():
    z["connected_external_walls"] = list(dict.fromkeys(z["connected_external_walls"]))
    z["connected_internal_walls"] = list(dict.fromkeys(z["connected_internal_walls"]))
    z["doors"]   = list(dict.fromkeys(z["doors"]))
    z["windows"] = list(dict.fromkeys(z["windows"]))
    z["total_walls_connected"] = (
        len(z["connected_external_walls"]) + len(z["connected_internal_walls"])
    )
 
total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]), 2)
 
# ─────────────────────────────────────────────
#  OUTPUT
# ─────────────────────────────────────────────
print(json.dumps({
    "external_walls": external_walls,
    "internal_walls": internal_walls,
    "doors":    doors_out,
    "windows":  windows_out,
    "zones":    zones_out,
    "summary": {
        "total_external_walls":    len(external_walls),
        "total_internal_walls":    len(internal_walls),
        "total_doors":             len(doors_out),
        "total_windows":           len(windows_out),
        "total_zones":             len(zones_out),
        "total_area_sqft":         total_area,
        "unique_doors_in_zones":   len(all_zone_doors),
        "unique_windows_in_zones": len(all_zone_windows),
        "detection_strategy":      "archiquant_standard",
        "px_per_ft":               round(SCALE, 2),
    }
}))
