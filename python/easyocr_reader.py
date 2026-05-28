"""
ArchiQuant Universal Floor Plan OCR
Works with ANY architectural drawing:
  - Any scale (metric/imperial)
  - Any label format (EW1, EW T=9, E.W., EXTERNAL WALL, etc.)
  - Any separator (x, X, ×, *, ·)
  - Any OCR noise (0→D, A→4, S→5, *→X, -→=)
  - Split tokens merged automatically
  - French + English room names
  - Fallback: OpenCV lines → auto boundary
"""
import cv2, easyocr, re, json, numpy as np, math, sys
from pdf2image import convert_from_path

MIN_CONF   = 0.28
reader     = easyocr.Reader(['en'], gpu=False)
input_path = sys.argv[1]

# ══════════════════════════════════════════════
#  LABEL PATTERNS — every known architect format
# ══════════════════════════════════════════════
EW_PATTERNS = [
    r"^EW[\s,=T\d\.]*$", r"^EW\d+[A-Z]?$",
    r"^E\.W\.?\d*$", r"^E\d+$",
    r"^EXT[\s\-]?WALL\d*$", r"^EXTERNAL[\s\-]?WALL\d*$",
    r"^OUTER[\s\-]?WALL\d*$", r"^BOUNDARY[\s\-]?WALL\d*$",
    r"^MUR[\s\-]?EXT\d*$",
]
IW_PATTERNS = [
    r"^IW[\s,=T\d\.]*$", r"^IW\d+[A-Z]?$",
    r"^IW[\s\(].*$", r"^I\.W\.?\d*$", r"^I\d+$",
    r"^INT[\s\-]?WALL\d*$", r"^INTERNAL[\s\-]?WALL\d*$",
    r"^INNER[\s\-]?WALL\d*$", r"^PARTITION[\s\-]?\d*$",
    r"^DIVIDER[\s\-]?\d*$", r"^PW\d*$", r"^CLOISON\d*$",
]
DOOR_PATTERNS = [
    r"^D[=\-\s]?(\d+\.?\d*)[xX*](\d+\.?\d*)$",
    r"^DOOR[=\-\s]?(\d+\.?\d*)[xX*](\d+\.?\d*)$",
    r"^D[\s\(].*?(\d+\.?\d*)[xX*](\d+\.?\d*).*$",
    r"^D\(MAIN\)$", r"^D\(EXT\)$",
    r"^D\d+$", r"^DOOR\d*$", r"^PORTE\d*$",
]
WINDOW_PATTERNS = [
    r"^W[=\-\s]?(\d+\.?\d*)[xX*](\d+\.?\d*)$",
    r"^WIN[=\-\s]?(\d+\.?\d*)[xX*](\d+\.?\d*)$",
    r"^WINDOW[=\-\s]?(\d+\.?\d*)[xX*](\d+\.?\d*)$",
    r"^W\d+$", r"^WIN\d*$", r"^WINDOW\d*$", r"^FENETRE\d*$",
]
ROOM_KEYWORDS = [
    # English
    "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL",
    "BALCONY","STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM",
    "C.BEDROOM","MASTER","DRAWING","UTILITY","GARAGE","PASSAGE",
    "WASH","BATH","FAMILY","PANTRY","LAUNDRY","TERRACE","VERANDAH",
    "ENTRANCE","FOYER","MBEDROOM","CBEDROOM","BEDROOM 2","BEDROOM 3",
    "LIVING ROOM","DINING ROOM","STORE ROOM","PRAYER","POWDER",
    # French
    "CHAMBRE","SALON","CUISINE","SALLE","BAIN","TOILETTE","WC",
    "COULOIR","ENTREE","BUREAU","DRESSING","BUANDERIE","SEJOUR",
    "TERRASSE","CELLIER","PLACARD","DEGAGEMENT","DOUCHE","MAGASIN",
]

# ══════════════════════════════════════════════
#  LOAD IMAGE / PDF
# ══════════════════════════════════════════════
if input_path.lower().endswith(".pdf"):
    pages = convert_from_path(input_path, dpi=300)
else:
    img   = cv2.imread(input_path)
    pages = [img] if img is not None else []

elements   = []
raw_images = []

def preprocess(img):
    if not isinstance(img, np.ndarray):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

for page in pages:
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
    raw_images.append(image)
    raw_results = reader.readtext(preprocess(image))

    # Sort by row then column for sequential merging
    raw_sorted = sorted(raw_results, key=lambda r: (round(r[0][0][1]/25), r[0][0][0]))
    used = set()

    for i, (bbox, text, prob) in enumerate(raw_sorted):
        if i in used or prob < MIN_CONF: continue
        x = int(sum([p[0] for p in bbox]) / 4)
        y = int(sum([p[1] for p in bbox]) / 4)
        cur = text.upper().strip()

        # Try merging with next token
        if i+1 < len(raw_sorted) and i+1 not in used:
            bbox2, text2, prob2 = raw_sorted[i+1]
            x2 = int(sum([p[0] for p in bbox2]) / 4)
            y2 = int(sum([p[1] for p in bbox2]) / 4)
            nxt = text2.upper().strip()

            same_line = abs(y2-y) < 35
            close     = abs(x2-x) < 220

            if same_line and close and prob2 >= MIN_CONF:
                with_sp  = cur + " " + nxt
                no_sp    = cur + nxt

                # ZONE + number → "ZONE 1", "ZONE 4", "ZONE 1A"
                if cur == "ZONE" and re.match(r"^\d+[A-Z]?$", nxt):
                    elements.append({"text": "ZONE " + nxt, "x": (x+x2)//2, "y": (y+y2)//2})
                    used.add(i); used.add(i+1); continue

                # "ZONE N" + letter → "ZONE 1A"
                if re.match(r"^ZONE \d+$", cur) and re.match(r"^[A-Z]$", nxt):
                    elements.append({"text": cur + nxt, "x": (x+x2)//2, "y": (y+y2)//2})
                    used.add(i); used.add(i+1); continue

                # Dimension merge: "W4" + "X4" or "3.50" + "X4.25"
                if re.search(r"[WD].*\d+.*[Xx*].*\d+", no_sp):
                    elements.append({"text": no_sp, "x": (x+x2)//2, "y": (y+y2)//2})
                    used.add(i); used.add(i+1); continue

                # "IW" + "T=4 H=10" → keep as one element
                if re.match(r"^(IW|EW)$", cur) and re.search(r"T[=\-]\d", nxt):
                    elements.append({"text": with_sp, "x": (x+x2)//2, "y": (y+y2)//2})
                    used.add(i); used.add(i+1); continue

        elements.append({"text": cur, "x": x, "y": y})
        used.add(i)

# ══════════════════════════════════════════════
#  UNIVERSAL CLEAN — fixes ALL OCR misreads
# ══════════════════════════════════════════════
def clean(t):
    t = t.upper().strip()

    # Normalize dimension separators → X
    for ch in ['×', '\xd7', '✕', '*', '·']:
        t = t.replace(ch, 'X')

    # Fix letter↔digit confusion in decimal values
    # O→0 in numbers: T=O.10→T=0.10, H=2.1O→H=2.10
    t = re.sub(r'([=\-])O\.(\d)', r'\g<1>0.\2', t)
    t = re.sub(r'(\d\.)(\d*)O\b', r'\g<1>\g<2>0', t)
    t = re.sub(r'(\d+\.\d+)O', r'\g<1>0', t)

    # Fix D read as 0: 01→D1, 07→D7, 0(MAIN)→D(MAIN)
    t = re.sub(r'^0\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
    t = re.sub(r'^0\s*\(', 'D(', t)
    t = re.sub(r'^0(\d+)$', r'D\1', t)

    # Fix W label confusion: WA→W4, WS→W5, WB→W8, WG→W6, WI/Wl→W1
    t = re.sub(r'^W([AQ])$',    'W4', t)
    t = re.sub(r'^W([S])$',     'W5', t)
    t = re.sub(r'^W([B8])$',    'W8', t)
    t = re.sub(r'^W([G6])$',    'W6', t)
    t = re.sub(r'^W([Il1])$',   'W1', t)

    # Fix ZONE label confusion: ZONE SA→ZONE 5A, ZONE IA→ZONE 1A
    t = re.sub(r'^ZONE\s+S([A-Z]?)$',  r'ZONE 5\1', t)
    t = re.sub(r'^ZONE\s+I([A-Z]?)$',  r'ZONE 1\1', t)
    t = re.sub(r'^ZONE\s+l([A-Z]?)$',  r'ZONE 1\1', t)
    t = re.sub(r'^ZONE\s+G([A-Z]?)$',  r'ZONE 6\1', t)
    t = re.sub(r'^ZONE\s+B([A-Z]?)$',  r'ZONE 8\1', t)

    # Normalize wall labels
    t = re.sub(r'\bEW,\b', 'EW', t)
    t = t.replace('1W','IW').replace('lW','IW').replace('lw','IW').replace('|W','IW')
    t = t.replace('EW.','EW').replace('IW.','IW')

    # Dash→equals for spec labels: T-9→T=9, H-10→H=10, H-3.00→H=3.00
    t = re.sub(r'\b([TH])-(\d)', r'\1=\2', t)
    t = re.sub(r'\b([DW])-(\d)', r'\1=\2', t)

    # Fix garbled specs: TZ4→T=4, H-L0→H=10, T-G→T=9
    t = re.sub(r'\bTZ(\d)', r'T=\1', t)
    t = re.sub(r'H-L(\d)', r'H=1\1', t)
    t = t.replace('T-G','T=9').replace('T=G','T=9')

    # Fix W-4→W=4, D-3→D=3
    t = re.sub(r'\bW-(\d)', r'W=\1', t)
    t = re.sub(r'\bD-(\d)', r'D=\1', t)

    # D(MAIN) normalization
    t = re.sub(r'D\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
    t = re.sub(r'D\s*\(\s*EXT.*?\)', 'D(EXT)', t)

    # Fix dimension letter confusion: W=AX4→W=4X4
    t = re.sub(r'\b([DW]=\d*)([A-OQ-Z])(X)', lambda m: m.group(1)+'4'+m.group(3), t)
    t = re.sub(r'(X)([A-OQ-Z])\b', r'\g<1>4', t)

    t = re.sub(r'\s+', ' ', t)
    return t.strip()

for e in elements: e["text"] = clean(e["text"])

img_h = raw_images[0].shape[0] if raw_images else 1000
img_w = raw_images[0].shape[1] if raw_images else 1000

# Remove legend/footer text (bottom 20% of image)
legend_y = img_h * 0.80
LEGEND_KW = [
    "= EXTERNAL WALL","= INTERNAL WALL","EW =","IW =","EW:","IW:",
    "ROOM TINTS","W = WINDOW","D = DOOR","T = THICKNESS","H = HEIGHT",
    "BEDROOMS / STUDY","BATHROOMS / LIVING","KITCHEN / DINING",
    "BEDROOMS","BATHROOMS","KITCHEN/DINING",
]
elements = [e for e in elements if not (
    e["y"] > legend_y and any(kw in e["text"] for kw in LEGEND_KW)
)]

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def dist(a,b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
def is_dup(x,y,pos,th=50): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)
def nearby(base,r=220): return [e for e in elements if dist(base,e)<r]

def matches_any(text, patterns):
    return any(re.match(p, text) for p in patterns)

def extract_dim(text, patterns):
    for p in patterns:
        m = re.match(p, text)
        if m and m.lastindex and m.lastindex >= 2:
            try: return float(m.group(1)), float(m.group(2))
            except: pass
    return None, None

def find_val(elems, pat):
    for e in elems:
        txt = re.sub(r'\b([TH])-(\d)', r'\1=\2', e["text"])
        m = re.search(pat, txt)
        if m:
            try: return float(m.group(1))
            except: pass
    return None

def find_dir(elems):
    for e in elems:
        for d in ["NORTH","SOUTH","EAST","WEST","MIDDLE","CENTER"]:
            if d in e["text"]: return d.lower()
    return None

def pos_by_location(x, y):
    m = 0.15
    if   y < img_h*m: return "north"
    elif y > img_h*(1-m): return "south"
    elif x < img_w*m: return "west"
    elif x > img_w*(1-m): return "east"
    return "inner"

def parse_feet_inches(text):
    if not text: return None
    text = str(text).replace(" ","")
    fm = re.search(r"(\d+)'", text)
    if not fm: return None
    feet   = int(fm.group(1))
    im     = re.search(r"'(\d+)", text)
    fracm  = re.search(r"(\d+)/(\d+)", text)
    inches = int(im.group(1)) if im else 0
    frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
    if inches > 12: inches = int(str(inches)[0])
    return math.ceil(feet+(inches+frac)/12)

def to_ft(val, unit):
    if val is None: return None
    return round(val*3.281,1) if unit=="m" else round(val,1)

# ══════════════════════════════════════════════
#  DETECT UNIT (metric / imperial)
# ══════════════════════════════════════════════
def detect_unit():
    mhits = ihits = 0
    for e in elements:
        t = e["text"]
        if re.search(r"\d+\.\d{2}", t): mhits += 1
        if re.search(r"\b\d+\.?\d*\s*M\b", t): mhits += 1
        if re.search(r"\d+'", t): ihits += 1
        if re.search(r"\d+X\d+", t) and not re.search(r"\d+\.\d+", t): ihits += 1
    return "m" if mhits > ihits else "ft"

drawing_unit = detect_unit()

# ══════════════════════════════════════════════
#  FIND NEARBY DIMENSION (for labeled W1/D1 items)
# ══════════════════════════════════════════════
def find_nearby_dim(base, radius=300):
    """Search radius for a WxH pattern near a label."""
    for e in sorted([e for e in elements if dist(base,e)<radius], key=lambda e:dist(base,e)):
        t = e["text"]
        m = re.search(r'(\d+\.?\d*)\s*X\s*(\d+\.?\d*)', t)
        if m:
            w,h = float(m.group(1)), float(m.group(2))
            if 0.3<=w<=30 and 0.3<=h<=30:
                return w, h
        m = re.search(r"(\d+'?\d*[\"']?)\s*X\s*(\d+'?\d*[\"']?)", t)
        if m:
            w = parse_feet_inches(m.group(1)) or float(re.search(r"\d+",m.group(1)).group())
            h = parse_feet_inches(m.group(2)) or float(re.search(r"\d+",m.group(2)).group())
            if w and h: return float(w), float(h)
    return None, None

# ══════════════════════════════════════════════
#  ZONE DIMENSION PARSER
# ══════════════════════════════════════════════
def zone_dims(base, r=500):
    for e in elements:
        if dist(base,e) > r: continue
        t = e["text"]
        # Metric decimal: 3.50 X 4.25
        m = re.search(r'(\d+\.\d+)\s*X\s*(\d+\.\d+)', t)
        if m:
            ww = to_ft(float(m.group(1)), drawing_unit)
            ll = to_ft(float(m.group(2)), drawing_unit)
            if ww and ll and 4<=ww<=200 and 4<=ll<=200:
                return round(min(ww,ll),1), round(max(ww,ll),1)
        # Imperial feet: 12'0"X15'0"
        m = re.search(r"(\d+['']\d+[\"']?)\s*X\s*(\d+['']\d+[\"']?)", t)
        if m:
            ww = parse_feet_inches(m.group(1))
            ll = parse_feet_inches(m.group(2))
            if ww and ll and 4<=ww<=100 and 4<=ll<=100:
                return min(ww,ll), max(ww,ll)
        # Imperial no separator: 12'0"15'0"
        m = re.search(r"(\d+['']\d+)[\"']\s*(\d+['']\d+)", t)
        if m:
            ww = parse_feet_inches(m.group(1))
            ll = parse_feet_inches(m.group(2))
            if ww and ll and 4<=ww<=100 and 4<=ll<=100:
                return min(ww,ll), max(ww,ll)
        # Simple integers NxM (feet)
        m = re.search(r'(\d{1,3})\s*X\s*(\d{1,3})', t)
        if m:
            ww,ll = int(m.group(1)), int(m.group(2))
            if 4<=ww<=100 and 4<=ll<=100 and not matches_any(t, WINDOW_PATTERNS+DOOR_PATTERNS):
                return min(ww,ll), max(ww,ll)
    return None, None

# ══════════════════════════════════════════════
#  AUTO PIXEL-PER-FOOT SCALE
#  Detects real scale from drawing content
# ══════════════════════════════════════════════
def auto_px_per_ft():
    # Strategy A: find overall building dimension text
    for e in elements:
        t = e["text"]
        # "10.00 M" or "10.00M" total width label
        m = re.search(r'(\d+\.\d+)\s*M\b', t)
        if m:
            total_ft = float(m.group(1)) * 3.281
            if 10 <= total_ft <= 500:
                return img_w / total_ft
        # "50'0"" total dimension
        m = re.search(r"(\d{2,3})'0\"?", t)
        if m:
            total_ft = float(m.group(1))
            if 20 <= total_ft <= 500:
                return img_w / total_ft

    # Strategy B: use two zone labels with known dims
    zone_pts = []
    for e in elements:
        if re.match(r"^ZONE\s*\d+[A-Z]?$", e["text"]):
            wft, lft = zone_dims(e)
            if wft and lft:
                zone_pts.append((e, max(wft,lft)))

    if len(zone_pts) >= 2:
        best_d, best_scale = 0, None
        for i in range(len(zone_pts)):
            for j in range(i+1, len(zone_pts)):
                ea, fa = zone_pts[i]
                eb, fb = zone_pts[j]
                px_d = dist(ea, eb)
                est_ft = (fa + fb) / 1.5
                if px_d > best_d and est_ft > 0:
                    s = px_d / est_ft
                    if 5 < s < 600:
                        best_d, best_scale = px_d, s
        if best_scale:
            return best_scale

    # Strategy C: fallback by unit
    return img_w / (32.8 if drawing_unit=="m" else 40.0)

# ══════════════════════════════════════════════
#  STRATEGY 1 — Text label wall detection
# ══════════════════════════════════════════════
ew_anchors, ew_seen = [], []
iw_anchors, iw_seen = [], []

for e in elements:
    t = e["text"]
    if matches_any(t, EW_PATTERNS):
        if not is_dup(e["x"],e["y"],ew_seen):
            ew_anchors.append(dict(e)); ew_seen.append((e["x"],e["y"]))
    elif matches_any(t, IW_PATTERNS):
        if not is_dup(e["x"],e["y"],iw_seen):
            iw_anchors.append(dict(e)); iw_seen.append((e["x"],e["y"]))

external_walls = {}
internal_walls = {}

def build_from_text():
    for i, anchor in enumerate(ew_anchors):
        eid = f"ew{i+1}"
        nb  = nearby(anchor, 280)
        tv  = find_val(nb, r"T[=\s](\d+\.?\d*)")
        hv  = find_val(nb, r"H[=\s](\d+\.?\d*)")
        td  = round(tv*39.37) if tv and drawing_unit=="m" else (int(tv) if tv else 9)
        hd  = round(hv*3.281) if hv and drawing_unit=="m" else (int(hv) if hv else 10)
        pos = find_dir(nb) or pos_by_location(anchor["x"], anchor["y"])
        external_walls[eid] = {"id":eid,"position":pos,"length_ft":None,
            "thickness_in":td,"height_ft":hd,
            "connected":{"windows":[],"doors":[],"internal_walls":[]}}
        anchor["id"] = eid

    for i, anchor in enumerate(iw_anchors):
        iwid = f"iw{i+1}"
        nb   = nearby(anchor, 280)
        tv   = find_val(nb, r"T[=\s](\d+\.?\d*)")
        hv   = find_val(nb, r"H[=\s](\d+\.?\d*)")
        td   = round(tv*39.37) if tv and drawing_unit=="m" else (int(tv) if tv else 4)
        hd   = round(hv*3.281) if hv and drawing_unit=="m" else (int(hv) if hv else 10)
        pos  = find_dir(nb)
        conn = []
        for ea in sorted(ew_anchors, key=lambda a: dist(anchor,a))[:2]:
            conn.append(ea["id"])
            if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
                external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
        internal_walls[iwid] = {"id":iwid,"position":pos,"thickness_in":td,
            "height_ft":hd,"length_ft":None,"connects_external":conn,
            "connected":{"doors":[],"windows":[]}}
        anchor["id"] = iwid

# ══════════════════════════════════════════════
#  STRATEGY 2 — OpenCV line detection
# ══════════════════════════════════════════════
def build_from_opencv():
    img   = raw_images[0]
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    gray  = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    edges = cv2.Canny(gray,40,120,apertureSize=3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,minLineLength=img_w//12,maxLineGap=25)
    hl,vl=[],[]
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2=l[0]
            ang=abs(math.degrees(math.atan2(y2-y1,x2-x1)))
            ln=math.hypot(x2-x1,y2-y1)
            if ang<15 or ang>165: hl.append((min(x1,x2),max(x1,x2),(y1+y2)//2,ln))
            elif 75<ang<105:      vl.append(((x1+x2)//2,min(y1,y2),max(y1,y2),ln))

    def cluster(lines,axis,gap=35):
        if not lines: return []
        lines=sorted(lines,key=lambda l:l[2] if axis=='h' else l[0])
        out=[lines[0]]
        for ln in lines[1:]:
            c=ln[2] if axis=='h' else ln[0]; p=out[-1][2] if axis=='h' else out[-1][0]
            if abs(c-p)<gap: out[-1]=ln if ln[3]>out[-1][3] else out[-1]
            else: out.append(ln)
        return out

    ec=ic=0; bm=0.12
    for (x1,x2,y,l) in cluster(hl,'h'):
        lft=round(l/(img_w/30))
        if y<img_h*bm or y>img_h*(1-bm):
            ec+=1; eid=f"ew{ec}"; pos="north" if y<img_h/2 else "south"
            external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
            ew_anchors.append({"x":(x1+x2)//2,"y":y,"id":eid})
        else:
            ic+=1; iwid=f"iw{ic}"
            internal_walls[iwid]={"id":iwid,"position":"horizontal","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
            iw_anchors.append({"x":(x1+x2)//2,"y":y,"id":iwid})
    for (x,y1,y2,l) in cluster(vl,'v'):
        lft=round(l/(img_h/30))
        if x<img_w*bm or x>img_w*(1-bm):
            ec+=1; eid=f"ew{ec}"; pos="west" if x<img_w/2 else "east"
            external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
            ew_anchors.append({"x":x,"y":(y1+y2)//2,"id":eid})
        else:
            ic+=1; iwid=f"iw{ic}"
            internal_walls[iwid]={"id":iwid,"position":"vertical","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
            iw_anchors.append({"x":x,"y":(y1+y2)//2,"id":iwid})
    for iwa in iw_anchors:
        iwid=iwa["id"]
        for ea in sorted(ew_anchors,key=lambda a:dist(iwa,a))[:2]:
            if ea["id"] not in internal_walls[iwid]["connects_external"]: internal_walls[iwid]["connects_external"].append(ea["id"])
            if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]: external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
    return ec>0

# ══════════════════════════════════════════════
#  STRATEGY 3 — Auto 4 boundary walls
# ══════════════════════════════════════════════
def build_fallback():
    for pos,eid,x,y in [("north","ew1",img_w//2,5),("east","ew2",img_w-5,img_h//2),
                         ("south","ew3",img_w//2,img_h-5),("west","ew4",5,img_h//2)]:
        external_walls[eid]={"id":eid,"position":pos,"length_ft":None,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
        ew_anchors.append({"x":x,"y":y,"id":eid})

# Run strategy chain
strategy = "auto_boundary"
if ew_anchors or iw_anchors:
    build_from_text(); strategy="text_labels"
elif raw_images and build_from_opencv():
    strategy="opencv_lines"
else:
    build_fallback()

# ══════════════════════════════════════════════
#  DOORS
# ══════════════════════════════════════════════
door_list, door_seen = [], []
for e in elements:
    t=e["text"]
    w,h=extract_dim(t,DOOR_PATTERNS)
    if w and h:
        if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
        if not is_dup(e["x"],e["y"],door_seen):
            door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
            door_seen.append((e["x"],e["y"]))
    elif re.match(r"^D\d+$|^DOOR\d+$|^D\(.*\)$|^DMAIN$", t) and not is_dup(e["x"],e["y"],door_seen):
        dw,dh = find_nearby_dim(e, 300)
        if dw and dh and drawing_unit=="m" and dw<5: dw=round(dw*3.281,1); dh=round(dh*3.281,1)
        dw = dw or (4.0 if "MAIN" in t else 3.0)
        dh = dh or 7.0
        door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":dw,"height_ft":dh})
        door_seen.append((e["x"],e["y"]))

# ══════════════════════════════════════════════
#  WINDOWS
# ══════════════════════════════════════════════
window_list, win_seen = [], []
for e in elements:
    t=e["text"]
    w,h=extract_dim(t,WINDOW_PATTERNS)
    if w and h:
        if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
        if not is_dup(e["x"],e["y"],win_seen):
            window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
            win_seen.append((e["x"],e["y"]))
    elif re.match(r"^W\d+$|^WIN\d+$|^WINDOW\d+$", t) and not is_dup(e["x"],e["y"],win_seen):
        wv,hv = find_nearby_dim(e, 300)
        if wv and hv and drawing_unit=="m" and wv<5: wv=round(wv*3.281,1); hv=round(hv*3.281,1)
        wv = wv or 4.0; hv = hv or 4.0
        window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":wv,"height_ft":hv})
        win_seen.append((e["x"],e["y"]))

# ══════════════════════════════════════════════
#  ASSIGN on_wall + BACK-FILL
# ══════════════════════════════════════════════
def nearest_wall(pos):
    best,bd=None,999999
    for a in ew_anchors+iw_anchors:
        d=dist(pos,a)
        if d<bd: bd,best=d,a["id"]
    return best

doors_out={}
for d in door_list:
    ow=nearest_wall({"x":d["x"],"y":d["y"]})
    doors_out[d["id"]]={"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],"on_wall":ow}
    if ow:
        wl=external_walls.get(ow) or internal_walls.get(ow)
        if wl and d["id"] not in wl["connected"]["doors"]: wl["connected"]["doors"].append(d["id"])

windows_out={}
for w in window_list:
    ow=nearest_wall({"x":w["x"],"y":w["y"]})
    windows_out[w["id"]]={"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],"on_wall":ow}
    if ow:
        wl=external_walls.get(ow) or internal_walls.get(ow)
        if wl and w["id"] not in wl["connected"]["windows"]: wl["connected"]["windows"].append(w["id"])

# ══════════════════════════════════════════════
#  ZONES
# ══════════════════════════════════════════════
ZONE_PAT = re.compile(r"^ZONE\s*\d+[A-Z]?$")
zones_out = {}

def nearest_room(base, r=400):
    for e in sorted(elements, key=lambda e: dist(base,e)):
        if dist(base,e)>r: break
        for kw in ROOM_KEYWORDS:
            if kw in e["text"]: return e["text"]
    return "UNKNOWN"

px_ft = auto_px_per_ft()   # compute once for all zones

for e in elements:
    if ZONE_PAT.match(e["text"]):
        zid  = re.sub(r"\s+","",e["text"])
        name = nearest_room(e)
        wft, lft = zone_dims(e)
        area = round(wft*lft,2) if wft and lft else None

        # Zone proximity radius: based on zone size in pixels
        # Larger zones need bigger search radius for walls
        if area:
            zone_side_px = math.sqrt(area) * px_ft
            zp = zone_side_px * 0.75
        else:
            zp = img_w * 0.28   # fallback: 28% of image width

        conn_ew = list(dict.fromkeys(a["id"] for a in ew_anchors if dist(e,a)<zp*1.4))
        conn_iw = list(dict.fromkeys(a["id"] for a in iw_anchors if dist(e,a)<zp*1.2))
        conn_d  = list(dict.fromkeys(d["id"] for d in door_list   if dist(e,{"x":d["x"],"y":d["y"]})<zp))
        conn_w  = list(dict.fromkeys(w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp))

        zones_out[zid] = {
            "id":zid,"label":name,"width_ft":wft,"length_ft":lft,"area_sqft":area,
            "connected_external_walls":conn_ew,
            "connected_internal_walls":conn_iw,
            "total_walls_connected":len(conn_ew)+len(conn_iw),
            "doors":conn_d,"windows":conn_w
        }

# ══════════════════════════════════════════════
#  DEDUP — no double counting
# ══════════════════════════════════════════════
all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])

for zid in zones_out:
    z = zones_out[zid]
    z["connected_external_walls"] = list(dict.fromkeys(z["connected_external_walls"]))
    z["connected_internal_walls"] = list(dict.fromkeys(z["connected_internal_walls"]))
    z["doors"]   = list(dict.fromkeys(z["doors"]))
    z["windows"] = list(dict.fromkeys(z["windows"]))
    z["total_walls_connected"] = len(z["connected_external_walls"]) + len(z["connected_internal_walls"])

total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]), 2)

# ══════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════
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
        "drawing_unit":            drawing_unit,
        "detection_strategy":      strategy,
        "px_per_ft":               round(px_ft, 2)
    }
}))
# import cv2, easyocr, re, json, numpy as np, math, sys
# from pdf2image import convert_from_path

# MIN_CONF   = 0.28
# reader     = easyocr.Reader(['en'], gpu=False)
# input_path = sys.argv[1]

# # ══════════════════════════════════════════════
# #  UNIVERSAL LABEL PATTERNS
# # ══════════════════════════════════════════════
# EW_PATTERNS = [
#     r"^EW[\s,=T\d\.]*$", r"^EW\d+[A-Z]?$", r"^E\.W\.?\d*$",
#     r"^E\d+$", r"^EXT[\s\-]?WALL\d*$", r"^EXTERNAL[\s\-]?WALL\d*$",
#     r"^OUTER[\s\-]?WALL\d*$", r"^BOUNDARY[\s\-]?WALL\d*$",
#     r"^MUR[\s\-]?EXT\d*$", r"^MUR[\s\-]?EXTERIEUR\d*$",
# ]
# IW_PATTERNS = [
#     r"^IW[\s,=T\d\.]*$", r"^IW\d+[A-Z]?$", r"^I\.W\.?\d*$",
#     r"^IW[\s\(].*$",
#     r"^I\d+$", r"^INT[\s\-]?WALL\d*$", r"^INTERNAL[\s\-]?WALL\d*$",
#     r"^INNER[\s\-]?WALL\d*$", r"^PARTITION[\s\-]?\d*$",
#     r"^DIVIDER[\s\-]?\d*$", r"^PW\d*$",
#     r"^MUR[\s\-]?INT\d*$", r"^CLOISON\d*$",
# ]
# DOOR_PATTERNS = [
#     r"^D[=\-\s]?(\d+\.?\d*)[xX×X\xd7](\d+\.?\d*)$",
#     r"^DOOR[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
#     r"^DR[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
#     r"^D[\s\(].*?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*).*$",
#     r"^D\(MAIN\).*?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*).*$",
#     r"^D\d+$", r"^DOOR\d*$", r"^DR\d*$", r"^PORTE\d*$",
#     r"^D\(.*\)$", r"^DMAIN$", r"^D\(MAIN\)$",
# ]
# WINDOW_PATTERNS = [
#     r"^W[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
#     r"^W[=\-\s]?(\d+['']\d*[\"']?)\s*[xX×\xd7]\s*(\d+['']\d*[\"']?)$",
#     r"^W[=\-]?(\d+)'?(\d*)[\"']?[xX×\xd7](\d+)[''\"']?$",
#     r"^WIN[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
#     r"^WINDOW[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
#     r"^W\d+[\s\n]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
#     r"^W\d+$", r"^WIN\d*$", r"^WINDOW\d*$", r"^FENETRE\d*$",
# ]
# ROOM_KEYWORDS = [
#     "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL",
#     "BALCONY","STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM",
#     "C.BEDROOM","MASTER","DRAWING","UTILITY","GARAGE","PASSAGE",
#     "WASH","BATH","FAMILY","PANTRY","LAUNDRY","TERRACE","VERANDAH",
#     "ENTRANCE","FOYER","MBEDROOM","CBEDROOM","BEDROOM 2","BEDROOM 3",
#     "LIVING ROOM","DINING ROOM","STORE ROOM","PRAYER","POWDER",
#     "CHAMBRE","SALON","CUISINE","SALLE","BAIN","TOILETTE","WC",
#     "COULOIR","ENTREE","BUREAU","DRESSING","BUANDERIE","SEJOUR",
#     "TERRASSE","CELLIER","PLACARD","DEGAGEMENT","SALLE DE BAIN",
#     "SALLE A MANGER","SALON PRIVE","CHAMBRE ENFANT","DOUCHE","MAGASIN",
# ]

# DIM_SEP = r"[xX×\xd7\*]"   # universal dimension separator (x, X, ×, *)

# # ══════════════════════════════════════════════
# #  LOAD
# # ══════════════════════════════════════════════
# if input_path.lower().endswith(".pdf"):
#     pages = convert_from_path(input_path, dpi=300)
# else:
#     img   = cv2.imread(input_path)
#     pages = [img] if img is not None else []

# elements   = []
# raw_images = []

# def preprocess(img):
#     if not isinstance(img, np.ndarray):
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
#     _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     return th

# for page in pages:
#     image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
#     raw_images.append(image)
#     raw_results = reader.readtext(preprocess(image))

#     # Merge split detections on same line
#     merged   = []
#     used     = set()
#     raw_sorted = sorted(raw_results, key=lambda r: (round(r[0][0][1]/20), r[0][0][0]))
#     for i, (bbox, text, prob) in enumerate(raw_sorted):
#         if i in used or prob < MIN_CONF: continue
#         x = int(sum([p[0] for p in bbox]) / 4)
#         y = int(sum([p[1] for p in bbox]) / 4)
#         cur_text = text.upper().strip()

#         if i+1 < len(raw_sorted) and i+1 not in used:
#             bbox2, text2, prob2 = raw_sorted[i+1]
#             x2 = int(sum([p[0] for p in bbox2]) / 4)
#             y2 = int(sum([p[1] for p in bbox2]) / 4)
#             same_line = abs(y2-y) < 30
#             close_x   = abs(x2-x) < 200

#             if same_line and close_x and prob2 >= MIN_CONF:
#                 next_text    = text2.upper().strip()
#                 with_space   = cur_text + " " + next_text
#                 without_space= cur_text + next_text

#                 # Rule 1: Any "ZONE" + digit/label → "ZONE 1", "ZONE 1A" etc.
#                 if re.match(r"^ZONE$", cur_text) and re.match(r"^\d+[A-Z]?$", next_text):
#                     merged.append({"text": "ZONE " + next_text, "x": (x+x2)//2, "y": (y+y2)//2})
#                     used.add(i); used.add(i+1); continue

#                 # Rule 2: "ZONE N" + letter → "ZONE 1A", "ZONE 4B" etc.
#                 if re.match(r"^ZONE \d+$", cur_text) and re.match(r"^[A-Z]$", next_text):
#                     merged.append({"text": cur_text + next_text, "x": (x+x2)//2, "y": (y+y2)//2})
#                     used.add(i); used.add(i+1); continue

#                 # Rule 3: Dimension labels — "W4" + "X4" or "W-4" + "6X4"
#                 if re.search(r"[WD].*\d+.*[Xx*×].*\d+", without_space):
#                     merged.append({"text": without_space, "x": (x+x2)//2, "y": (y+y2)//2})
#                     used.add(i); used.add(i+1); continue

#                 # Rule 4: Any label + dimension — "W1" + "1.35X1.20"
#                 if re.match(r"^[WD]\d+$", cur_text) and re.search(r"\d+[.*X]\d+", next_text):
#                     merged.append({"text": with_space, "x": (x+x2)//2, "y": (y+y2)//2})
#                     used.add(i); used.add(i+1); continue

#                 # Rule 5: IW/EW + spec — "IW" + "T=4" or "EW" + "T=9"
#                 if re.match(r"^(IW|EW)$", cur_text) and re.search(r"T[=\-]", next_text):
#                     merged.append({"text": with_space, "x": (x+x2)//2, "y": (y+y2)//2})
#                     used.add(i); used.add(i+1); continue

#         merged.append({"text": cur_text, "x": x, "y": y})
#         used.add(i)
#     elements.extend(merged)

# # ══════════════════════════════════════════════
# #  CLEAN — normalize ALL OCR noise
# # ══════════════════════════════════════════════
# def clean(t):
#     """
#     Universal OCR noise cleaner.
#     Fixes ALL known misreads regardless of drawing source.
#     """
#     t = t.upper().strip()

#     # ── 1. Normalize dimension separators ───────────────────────────────
#     # All of these mean "times" in dimension context
#     t = t.replace('\xd7','X')   # Unicode ×
#     t = t.replace('×','X')        # Unicode × (different encoding)
#     t = t.replace('✕','X')        # Unicode cross
#     t = t.replace('*','X')        # Asterisk used as multiplier
#     t = t.replace('·','X')        # Middle dot

#     # ── 2. Fix digit/letter confusion (global rules) ─────────────────────
#     # OCR commonly confuses:
#     #   0 ↔ O (zero vs letter O)
#     #   1 ↔ I or l (one vs letter I/l)
#     #   5 ↔ S (five vs letter S)
#     #   4 ↔ A (four vs letter A)
#     #   6 ↔ G or b (six vs G/b)
#     #   8 ↔ B (eight vs letter B)

#     # Fix decimal values: T=O.10 → T=0.10, H=2.1O → H=2.10
#     t = re.sub(r'([TH]=\d*\.)(\d*)O', r'\g<1>\g<2>0', t)   # trailing O → 0
#     t = re.sub(r'([TH]=)O\.', r'\g<1>0.', t)                  # leading O → 0

#     # Fix H-3.OO → H=3.00, H-2.1O → H=2.10
#     t = re.sub(r'([TH][-=]\d+\.\d*)O', lambda m: m.group(0)[:-1]+'0', t)

#     # ── 3. Fix door label: D read as 0 by OCR ────────────────────────────
#     # '01' → 'D1', '02' → 'D2' etc. (only at start of string)
#     # '0 (MAIN)' → 'D(MAIN)', '0(MAIN)' → 'D(MAIN)'
#     t = re.sub(r'^0\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
#     t = re.sub(r'^0\s*\(', 'D(', t)
#     t = re.sub(r'^0(\d+)$', r'D\1', t)          # 01 → D1, 07 → D7

#     # ── 4. Fix window labels: letter/digit confusion ──────────────────────
#     # WA→W4, WS→W5, WB→W6, WG→W6, WI→W1, Wl→W1
#     t = re.sub(r'^W([AAQO])$', 'W4', t)          # WA, WQ → W4
#     t = re.sub(r'^W([S5])$',   'W5', t)           # WS → W5
#     t = re.sub(r'^W([Bb8B])$', 'W8', t)           # WB → W8
#     t = re.sub(r'^W([Gg6])$',  'W6', t)           # WG → W6
#     t = re.sub(r'^W([Il1])$',  'W1', t)           # WI,Wl → W1

#     # ── 5. Fix ZONE label: letter/digit confusion ─────────────────────────
#     # ZONE SA → ZONE 5A, ZONE lA → ZONE 1A, ZONE IB → ZONE 1B
#     t = re.sub(r'^ZONE\s+S([A-Z]?)$', r'ZONE 5\1', t)   # SA→5A, SB→5B
#     t = re.sub(r'^ZONE\s+I([A-Z]?)$', r'ZONE 1\1', t)   # IA→1A
#     t = re.sub(r'^ZONE\s+l([A-Z]?)$', r'ZONE 1\1', t)   # lA→1A
#     t = re.sub(r'^ZONE\s+O([A-Z]?)$', r'ZONE 0\1', t)   # O→0

#     # ── 6. Normalize wall labels ──────────────────────────────────────────
#     t = re.sub(r"\bEW,\b","EW",t)
#     t = t.replace("1W","IW").replace("lW","IW").replace("lw","IW").replace("|W","IW")
#     t = t.replace("EW.","EW").replace("IW.","IW")

#     # ── 7. Normalize dash → equals for spec labels ────────────────────────
#     # T-9 → T=9, H-10 → H=10, T-0.23 → T=0.23, H-3.00 → H=3.00
#     t = re.sub(r'\b([TH])-(\d)', r'\1=\2', t)
#     t = re.sub(r'\b([DW])-(\d)', r'\1=\2', t)

#     # ── 8. Fix common garbled patterns ───────────────────────────────────
#     t = re.sub(r'\bTZ(\d)', r'T=\1', t)         # TZ4 → T=4
#     t = re.sub(r'H-L(\d)', r'H=1\1', t)          # H-L0 → H=10
#     t = t.replace("T-G","T=9").replace("T=G","T=9")# T-G → T=9
#     t = re.sub(r'\bW-(\d)', r'W=\1', t)         # W-4 → W=4
#     t = re.sub(r'\bD-(\d)', r'D=\1', t)         # D-3 → D=3

#     # ── 9. Fix D(MAIN) variations ─────────────────────────────────────────
#     t = re.sub(r'D\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
#     t = re.sub(r'D\s*\(\s*EXT.*?\)', 'D(EXT)', t)

#     # ── 10. Fix dimension letter/digit confusion ──────────────────────────
#     # In patterns like "D=AX7" or "W=4XA"
#     t = re.sub(r'\b([DW]=)(\d*)([A-OQ-Z])(X)', lambda m: m.group(1)+m.group(2)+'4'+m.group(4), t)
#     t = re.sub(r'(X)([A-OQ-Z])(\b)', lambda m: m.group(1)+'4', t)

#     # ── 11. General whitespace cleanup ───────────────────────────────────
#     t = re.sub(r"\s+"," ",t)
#     return t.strip()


# for e in elements: e["text"] = clean(e["text"])

# img_h = raw_images[0].shape[0] if raw_images else 1000
# img_w = raw_images[0].shape[1] if raw_images else 1000

# # Remove legend/footer labels (bottom 20% = legend area)
# legend_y = img_h * 0.80
# LEGEND_KEYWORDS = [
#     "= EXTERNAL WALL","= INTERNAL WALL","EXTERNAL WALL","EW =","IW =",
#     "EW:","ROOM TINTS","BEDROOMS","BATHROOMS","KITCHEN/DINING",
#     "W = WINDOW","D = DOOR","T = THICKNESS","H = HEIGHT",
#     "BEDROOMS / STUDY","BATHROOMS / LIVING","KITCHEN / DINING",
# ]
# elements = [e for e in elements if not (
#     e["y"] > legend_y and
#     any(kw in e["text"] for kw in LEGEND_KEYWORDS)
# )]

# # ══════════════════════════════════════════════
# #  HELPERS
# # ══════════════════════════════════════════════
# def dist(a,b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
# def is_dup(x,y,pos,th=50): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)
# def nearby(base,r=220): return [e for e in elements if dist(base,e)<r]

# def matches_any(text, patterns):
#     return any(re.match(p, text) for p in patterns)

# def extract_dim_float(text, patterns):
#     """Extract WxH dimensions — handles x, X, × all normalized to X after clean()."""
#     for p in patterns:
#         m = re.match(p, text)
#         if m and m.lastindex and m.lastindex >= 2:
#             try: return float(m.group(1)), float(m.group(2))
#             except: pass
#     return None, None

# def find_nearby_dim(base, radius=250):
#     """
#     Search nearby text for dimension pattern like:
#     1.35 X 1.20  or  3.50 X 4.25  or  12'0"X15'0"
#     Returns (w, h) as floats or (None, None)
#     """
#     for e in sorted([e for e in elements if dist(base,e)<radius], key=lambda e: dist(base,e)):
#         t = e["text"]
#         # Metric decimal: 1.35 X 1.20
#         m = re.search(r'(\d+\.\d+)\s*[X\*]\s*(\d+\.\d+)', t)
#         if m:
#             return float(m.group(1)), float(m.group(2))
#         # Imperial feet: 12'0"X15'0"
#         m = re.search(r"(\d+'\d*[\"']?)\s*X\s*(\d+'\d*[\"']?)", t)
#         if m:
#             ww = parse_feet_inches(m.group(1))
#             ll = parse_feet_inches(m.group(2))
#             if ww and ll: return float(ww), float(ll)
#         # Simple integers: 3X7
#         m = re.search(r'\b(\d+\.?\d*)\s*[X*]\s*(\d+\.?\d*)\b', t)
#         if m:
#             ww, ll = float(m.group(1)), float(m.group(2))
#             if 0.3 <= ww <= 80 and 0.3 <= ll <= 80:
#                 return ww, ll
#     return None, None

# def find_val_float(elems, pat):
#     for e in elems:
#         txt = re.sub(r'\b([TH])-(\d)', r'\1=\2', e["text"])
#         m = re.search(pat, txt)
#         if m:
#             try: return float(m.group(1))
#             except: pass
#     return None

# def find_dir(elems):
#     dirs = ["NORTH","SOUTH","EAST","WEST","MIDDLE","CENTER","TOP","BOTTOM","LEFT","RIGHT"]
#     for e in elems:
#         for d in dirs:
#             if d in e["text"]: return d.lower()
#     return None

# def pos_by_location(x, y):
#     m = 0.15
#     if   y < img_h*m:      return "north"
#     elif y > img_h*(1-m):  return "south"
#     elif x < img_w*m:      return "west"
#     elif x > img_w*(1-m):  return "east"
#     return "inner"

# def parse_feet_inches(text):
#     if not text: return None
#     text = str(text).replace(" ","")
#     fm   = re.search(r"(\d+)'", text)
#     if not fm: return None
#     feet   = int(fm.group(1))
#     im     = re.search(r"'(\d+)", text)
#     fracm  = re.search(r"(\d+)/(\d+)", text)
#     inches = int(im.group(1)) if im else 0
#     frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
#     if inches > 12: inches = int(str(inches)[0])
#     return math.ceil(feet+(inches+frac)/12)

# def to_ft(val, unit):
#     """Convert value to feet if needed."""
#     if val is None: return None
#     if unit == "m": return round(val * 3.281, 1)
#     return round(val, 1)

# # ══════════════════════════════════════════════
# #  DETECT DRAWING UNIT
# # ══════════════════════════════════════════════
# def detect_unit():
#     metric_hits = 0; imperial_hits = 0
#     for e in elements:
#         t = e["text"]
#         if re.search(r"\d+\.\d{2}", t): metric_hits += 1
#         if re.search(r"\b\d+M\b|\d+\.?\d*\s*M\b", t): metric_hits += 1
#         if re.search(r"\d+'", t): imperial_hits += 1
#         if re.search(r"\d+X\d+", t) and not re.search(r"\d+\.\d+", t): imperial_hits += 1
#     return "m" if metric_hits > imperial_hits else "ft"

# drawing_unit = detect_unit()

# # ══════════════════════════════════════════════
# #  ZONE DIMENSION PARSER — all formats
# # ══════════════════════════════════════════════
# def zone_dims(base, r=500):
#     for e in elements:
#         if dist(base,e) > r: continue
#         t = e["text"]

#         # Metric decimal: 3.50 X 4.25
#         m = re.search(r'(\d+\.\d+)\s*[X\*]\s*(\d+\.\d+)', t)
#         if m:
#             ww = to_ft(float(m.group(1)), drawing_unit)
#             ll = to_ft(float(m.group(2)), drawing_unit)
#             if ww and ll and 4<=ww<=150 and 4<=ll<=150:
#                 return round(min(ww,ll),1), round(max(ww,ll),1)

#         # Imperial: 12'0"X15'0"
#         m = re.search(r"(\d+['']\d+[\"']?)\s*X\s*(\d+['']\d+[\"']?)", t)
#         if m:
#             ww = parse_feet_inches(m.group(1))
#             ll = parse_feet_inches(m.group(2))
#             if ww and ll and 4<=ww<=100 and 4<=ll<=100:
#                 return min(ww,ll), max(ww,ll)

#         # Imperial no separator: 12'0"15'0"
#         m = re.search(r"(\d+['']\d+)[\"']\s*(\d+['']\d+)", t)
#         if m:
#             ww = parse_feet_inches(m.group(1))
#             ll = parse_feet_inches(m.group(2))
#             if ww and ll and 4<=ww<=100 and 4<=ll<=100:
#                 return min(ww,ll), max(ww,ll)

#         # Simple integers NxM
#         m2 = re.search(r'(\d{1,2})\s*[X\*]\s*(\d{1,2})', t)
#         if m2:
#             ww,ll = int(m2.group(1)), int(m2.group(2))
#             if 5<=ww<=80 and 5<=ll<=80 and not matches_any(t, WINDOW_PATTERNS+DOOR_PATTERNS):
#                 return min(ww,ll), max(ww,ll)

#     return None, None

# # ══════════════════════════════════════════════
# #  STRATEGY 1 — Text label detection
# # ══════════════════════════════════════════════
# ew_anchors, ew_seen = [], []
# iw_anchors, iw_seen = [], []

# for e in elements:
#     t = e["text"]
#     if matches_any(t, EW_PATTERNS):
#         if not is_dup(e["x"],e["y"],ew_seen):
#             ew_anchors.append(dict(e)); ew_seen.append((e["x"],e["y"]))
#     elif matches_any(t, IW_PATTERNS):
#         if not is_dup(e["x"],e["y"],iw_seen):
#             iw_anchors.append(dict(e)); iw_seen.append((e["x"],e["y"]))

# external_walls = {}
# internal_walls = {}

# def build_from_text_labels():
#     for i, anchor in enumerate(ew_anchors):
#         eid = f"ew{i+1}"
#         nb  = nearby(anchor, 280)
#         t_v = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
#         h_v = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
#         if t_v and drawing_unit == "m":
#             t_display = round(t_v * 39.37)
#             h_display = round(h_v * 3.281) if h_v else 10
#         else:
#             t_display = int(t_v) if t_v else 9
#             h_display = int(h_v) if h_v else 10
#         d = find_dir(nb) or pos_by_location(anchor["x"], anchor["y"])
#         external_walls[eid] = {
#             "id":eid, "position":d,
#             "length_ft":None, "thickness_in":t_display, "height_ft":h_display,
#             "connected":{"windows":[],"doors":[],"internal_walls":[]}
#         }
#         anchor["id"] = eid

#     for i, anchor in enumerate(iw_anchors):
#         iwid = f"iw{i+1}"
#         nb   = nearby(anchor, 280)
#         t_v  = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
#         h_v  = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
#         if t_v and drawing_unit == "m":
#             t_display = round(t_v * 39.37)
#             h_display = round(h_v * 3.281) if h_v else 10
#         else:
#             t_display = int(t_v) if t_v else 4
#             h_display = int(h_v) if h_v else 10
#         d    = find_dir(nb)
#         conn = []
#         for ea in sorted(ew_anchors, key=lambda a: dist(anchor,a))[:2]:
#             conn.append(ea["id"])
#             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
#                 external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
#         internal_walls[iwid] = {
#             "id":iwid, "position":d,
#             "thickness_in":t_display, "height_ft":h_display, "length_ft":None,
#             "connects_external":conn,
#             "connected":{"doors":[],"windows":[]}
#         }
#         anchor["id"] = iwid

# # ══════════════════════════════════════════════
# #  STRATEGY 2 — OpenCV line detection
# # ══════════════════════════════════════════════
# def build_from_opencv():
#     img   = raw_images[0]
#     gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
#     kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#     gray  = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
#     edges = cv2.Canny(gray,40,120,apertureSize=3)
#     lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,
#                              minLineLength=img_w//12,maxLineGap=25)
#     h_lines,v_lines=[],[]
#     if lines is not None:
#         for line in lines:
#             x1,y1,x2,y2=line[0]
#             angle=abs(math.degrees(math.atan2(y2-y1,x2-x1)))
#             length=math.hypot(x2-x1,y2-y1)
#             if angle<15 or angle>165: h_lines.append((min(x1,x2),max(x1,x2),(y1+y2)//2,length))
#             elif 75<angle<105:        v_lines.append(((x1+x2)//2,min(y1,y2),max(y1,y2),length))

#     def cluster(lines,axis,gap=35):
#         if not lines: return []
#         lines=sorted(lines,key=lambda l:l[2] if axis=='h' else l[0])
#         out=[lines[0]]
#         for ln in lines[1:]:
#             coord=ln[2] if axis=='h' else ln[0]; prev=out[-1][2] if axis=='h' else out[-1][0]
#             if abs(coord-prev)<gap: out[-1]=ln if ln[3]>out[-1][3] else out[-1]
#             else: out.append(ln)
#         return out

#     ew_c=0; iw_c=0; bm=0.12
#     for (x1,x2,y,l) in cluster(h_lines,'h'):
#         lft=round(l/(img_w/30))
#         if y<img_h*bm or y>img_h*(1-bm):
#             ew_c+=1; eid=f"ew{ew_c}"; pos="north" if y<img_h/2 else "south"
#             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
#             ew_anchors.append({"x":(x1+x2)//2,"y":y,"id":eid})
#         else:
#             iw_c+=1; iwid=f"iw{iw_c}"
#             internal_walls[iwid]={"id":iwid,"position":"horizontal","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
#             iw_anchors.append({"x":(x1+x2)//2,"y":y,"id":iwid})
#     for (x,y1,y2,l) in cluster(v_lines,'v'):
#         lft=round(l/(img_h/30))
#         if x<img_w*bm or x>img_w*(1-bm):
#             ew_c+=1; eid=f"ew{ew_c}"; pos="west" if x<img_w/2 else "east"
#             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
#             ew_anchors.append({"x":x,"y":(y1+y2)//2,"id":eid})
#         else:
#             iw_c+=1; iwid=f"iw{iw_c}"
#             internal_walls[iwid]={"id":iwid,"position":"vertical","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
#             iw_anchors.append({"x":x,"y":(y1+y2)//2,"id":iwid})
#     for iwa in iw_anchors:
#         iwid=iwa["id"]
#         for ea in sorted(ew_anchors,key=lambda a:dist(iwa,a))[:2]:
#             if ea["id"] not in internal_walls[iwid]["connects_external"]: internal_walls[iwid]["connects_external"].append(ea["id"])
#             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]: external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
#     return ew_c>0

# # ══════════════════════════════════════════════
# #  STRATEGY 3 — Auto 4 boundary walls
# # ══════════════════════════════════════════════
# def build_fallback():
#     for pos,eid,x,y in [("north","ew1",img_w//2,5),("east","ew2",img_w-5,img_h//2),
#                          ("south","ew3",img_w//2,img_h-5),("west","ew4",5,img_h//2)]:
#         external_walls[eid]={"id":eid,"position":pos,"length_ft":None,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
#         ew_anchors.append({"x":x,"y":y,"id":eid})

# # ══════════════════════════════════════════════
# #  RUN STRATEGY CHAIN
# # ══════════════════════════════════════════════
# strategy = "auto_boundary"
# if ew_anchors or iw_anchors:
#     build_from_text_labels(); strategy="text_labels"
# elif raw_images and build_from_opencv():
#     strategy="opencv_lines"
# else:
#     build_fallback()

# # ══════════════════════════════════════════════
# #  DOORS — universal detection
# # ══════════════════════════════════════════════
# door_list, door_seen = [], []
# for e in elements:
#     t=e["text"]
#     w,h=extract_dim_float(t,DOOR_PATTERNS)
#     if w and h:
#         if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
#         if not is_dup(e["x"],e["y"],door_seen):
#             door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
#             door_seen.append((e["x"],e["y"]))
#     # Numbered door labels: D1, D(Main) — look for nearby dimensions
#     elif re.match(r"^D\d+$|^DOOR\d+$|^DR\d+$|^PORTE\d*$|^D\(.*\)$|^DMAIN$", t) and not is_dup(e["x"],e["y"],door_seen):
#         dw, dh = find_nearby_dim(e, radius=250)
#         if dw and dh:
#             if drawing_unit=="m" and dw<5: dw=round(dw*3.281,1); dh=round(dh*3.281,1)
#         else:
#             dw, dh = (4.0, 7.0) if "MAIN" in t else (3.0, 7.0)
#         door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":dw,"height_ft":dh})
#         door_seen.append((e["x"],e["y"]))

# # ══════════════════════════════════════════════
# #  WINDOWS — universal detection
# # ══════════════════════════════════════════════
# window_list, win_seen = [], []
# for e in elements:
#     t=e["text"]
#     w,h=extract_dim_float(t,WINDOW_PATTERNS)
#     if w and h:
#         if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
#         if not is_dup(e["x"],e["y"],win_seen):
#             window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
#             win_seen.append((e["x"],e["y"]))
#     # Numbered window labels: W1, W2 — look for nearby dimensions
#     elif re.match(r"^W\d+$|^WIN\d+$|^WINDOW\d+$|^FENETRE\d*$", t) and not is_dup(e["x"],e["y"],win_seen):
#         wv, hv = find_nearby_dim(e, radius=250)
#         if wv and hv:
#             if drawing_unit=="m" and wv<5: wv=round(wv*3.281,1); hv=round(hv*3.281,1)
#         else:
#             wv, hv = 4.0, 4.0
#         window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":wv,"height_ft":hv})
#         win_seen.append((e["x"],e["y"]))

# # ══════════════════════════════════════════════
# #  ASSIGN on_wall + BACK-FILL
# # ══════════════════════════════════════════════
# def nearest_wall(pos):
#     best,bd=None,999999
#     for a in ew_anchors+iw_anchors:
#         d=dist(pos,a)
#         if d<bd: bd,best=d,a["id"]
#     return best

# doors_out={}
# for d in door_list:
#     ow=nearest_wall({"x":d["x"],"y":d["y"]})
#     doors_out[d["id"]]={"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],"on_wall":ow}
#     if ow:
#         wl=external_walls.get(ow) or internal_walls.get(ow)
#         if wl and d["id"] not in wl["connected"]["doors"]: wl["connected"]["doors"].append(d["id"])

# windows_out={}
# for w in window_list:
#     ow=nearest_wall({"x":w["x"],"y":w["y"]})
#     windows_out[w["id"]]={"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],"on_wall":ow}
#     if ow:
#         wl=external_walls.get(ow) or internal_walls.get(ow)
#         if wl and w["id"] not in wl["connected"]["windows"]: wl["connected"]["windows"].append(w["id"])


# # ══════════════════════════════════════════════
# #  AUTO PIXEL-PER-FOOT SCALE DETECTION
# #  Computes real px/ft from dimension labels in
# #  the drawing itself — works for any scale/size
# # ══════════════════════════════════════════════
# def _auto_px_per_ft():
#     """
#     Find two zone labels with known dimensions and known pixel distance
#     to compute actual px_per_ft for THIS drawing.
#     Falls back to img_w / estimated_total_ft.
#     """
#     # Strategy A: find overall building dimension label (e.g. "10.00 m" or "50'0"")
#     for e in elements:
#         t = e["text"]
#         # Metric: "10.00 M" or "10.00M"
#         m = re.search(r"(\d+\.\d+)\s*M", t)
#         if m:
#             total_m  = float(m.group(1))
#             total_ft = total_m * 3.281
#             # This label is near the edge — use img_w as the pixel span
#             return img_w / total_ft
#         # Imperial: "50'0"" total dimension
#         m = re.search(r"(\d+)'\d*\s*[xX*]\s*(\d+)'\d*", t)
#         if m:
#             w_ft = float(m.group(1))
#             h_ft = float(m.group(2))
#             total_ft = max(w_ft, h_ft)
#             return img_w / total_ft

#     # Strategy B: use two zone labels whose real distance we can estimate
#     # Pick zones with known area and compute from pixel positions
#     zone_entries = []
#     for e in elements:
#         if re.match(r"^ZONE\s*\d+[A-Z]?$", e["text"]):
#             wft, lft = zone_dims(e)
#             if wft and lft:
#                 zone_entries.append((e, wft, lft))

#     if len(zone_entries) >= 2:
#         # Use the two zones furthest apart
#         best_d, best_scale = 0, None
#         for i in range(len(zone_entries)):
#             for j in range(i+1, len(zone_entries)):
#                 ea, wa, la = zone_entries[i]
#                 eb, wb, lb = zone_entries[j]
#                 px_dist = dist(ea, eb)
#                 # Estimated real distance = average of their dimensions
#                 real_ft = (max(wa,la) + max(wb,lb)) / 2
#                 if px_dist > best_d and real_ft > 0:
#                     best_d = px_dist
#                     best_scale = px_dist / real_ft
#         if best_scale and 5 < best_scale < 500:
#             return best_scale

#     # Strategy C: fallback based on unit
#     if drawing_unit == "m":
#         return img_w / 32.8   # 10m = 32.8ft typical
#     return img_w / 40.0        # 40ft typical imperial

# # ══════════════════════════════════════════════
# #  ZONES
# # ══════════════════════════════════════════════
# ZONE_PAT=re.compile(r"^ZONE\s*\d+[A-Z]?$")
# zones_out={}

# def nearest_room(base,r=400):
#     for e in sorted(elements,key=lambda e:dist(base,e)):
#         if dist(base,e)>r: break
#         for kw in ROOM_KEYWORDS:
#             if kw in e["text"]: return e["text"]
#     return "UNKNOWN"

# for e in elements:
#     if ZONE_PAT.match(e["text"]):
#         zid=re.sub(r"\s+","",e["text"])
#         name=nearest_room(e)
#         wft,lft=zone_dims(e)
#         area=round(wft*lft,2) if wft and lft else None
#         px_ft = _auto_px_per_ft()
#         zp = math.sqrt(area) * px_ft * 0.7 if area else img_w * 0.30

#         conn_ew=list(dict.fromkeys(a["id"] for a in ew_anchors if dist(e,a)<zp*1.3))
#         conn_iw=list(dict.fromkeys(a["id"] for a in iw_anchors if dist(e,a)<zp*1.1))
#         conn_d =list(dict.fromkeys(d["id"] for d in door_list   if dist(e,{"x":d["x"],"y":d["y"]})<zp))
#         conn_w =list(dict.fromkeys(w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp))

#         zones_out[zid]={
#             "id":zid,"label":name,"width_ft":wft,"length_ft":lft,"area_sqft":area,
#             "connected_external_walls":conn_ew,
#             "connected_internal_walls":conn_iw,
#             "total_walls_connected":len(conn_ew)+len(conn_iw),
#             "doors":conn_d,"windows":conn_w
#         }

# # ══════════════════════════════════════════════
# #  DEDUP SUMMARY
# # ══════════════════════════════════════════════
# all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
# all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])

# for zid in zones_out:
#     zones_out[zid]["connected_external_walls"] = list(dict.fromkeys(zones_out[zid]["connected_external_walls"]))
#     zones_out[zid]["connected_internal_walls"] = list(dict.fromkeys(zones_out[zid]["connected_internal_walls"]))
#     zones_out[zid]["doors"]   = list(dict.fromkeys(zones_out[zid]["doors"]))
#     zones_out[zid]["windows"] = list(dict.fromkeys(zones_out[zid]["windows"]))
#     zones_out[zid]["total_walls_connected"] = (
#         len(zones_out[zid]["connected_external_walls"]) +
#         len(zones_out[zid]["connected_internal_walls"])
#     )

# total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]), 2)

# # ══════════════════════════════════════════════
# #  OUTPUT
# # ══════════════════════════════════════════════
# print(json.dumps({
#     "external_walls": external_walls,
#     "internal_walls": internal_walls,
#     "doors":    doors_out,
#     "windows":  windows_out,
#     "zones":    zones_out,
#     "summary": {
#         "total_external_walls":    len(external_walls),
#         "total_internal_walls":    len(internal_walls),
#         "total_doors":             len(doors_out),
#         "total_windows":           len(windows_out),
#         "total_zones":             len(zones_out),
#         "total_area_sqft":         total_area,
#         "unique_doors_in_zones":   len(all_zone_doors),
#         "unique_windows_in_zones": len(all_zone_windows),
#         "drawing_unit":            drawing_unit,
#         "detection_strategy":      strategy
#     }
# }))







# # import cv2, easyocr, re, json, numpy as np, math, sys
# # from pdf2image import convert_from_path
 
# # MIN_CONF   = 0.28
# # reader     = easyocr.Reader(['en'], gpu=False)
# # input_path = sys.argv[1]
 
# # # ══════════════════════════════════════════════
# # #  UNIVERSAL LABEL PATTERNS
# # # ══════════════════════════════════════════════
# # EW_PATTERNS = [
# #     r"^EW[\s,=T\d\.]*$", r"^EW\d+[A-Z]?$", r"^E\.W\.?\d*$",
# #     r"^E\d+$", r"^EXT[\s\-]?WALL\d*$", r"^EXTERNAL[\s\-]?WALL\d*$",
# #     r"^OUTER[\s\-]?WALL\d*$", r"^BOUNDARY[\s\-]?WALL\d*$",
# #     r"^MUR[\s\-]?EXT\d*$", r"^MUR[\s\-]?EXTERIEUR\d*$",
# # ]
# # IW_PATTERNS = [
# #     r"^IW[\s,=T\d\.]*$", r"^IW\d+[A-Z]?$", r"^I\.W\.?\d*$",
# #     r"^IW[\s\(].*$",
# #     r"^I\d+$", r"^INT[\s\-]?WALL\d*$", r"^INTERNAL[\s\-]?WALL\d*$",
# #     r"^INNER[\s\-]?WALL\d*$", r"^PARTITION[\s\-]?\d*$",
# #     r"^DIVIDER[\s\-]?\d*$", r"^PW\d*$",
# #     r"^MUR[\s\-]?INT\d*$", r"^CLOISON\d*$",
# # ]
# # DOOR_PATTERNS = [
# #     r"^D[=\-\s]?(\d+\.?\d*)[xX×X\xd7](\d+\.?\d*)$",
# #     r"^DOOR[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# #     r"^DR[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# #     r"^D[\s\(].*?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*).*$",
# #     r"^D\(MAIN\).*?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*).*$",
# #     r"^D\d+$", r"^DOOR\d*$", r"^DR\d*$", r"^PORTE\d*$",
# #     r"^D\(.*\)$", r"^DMAIN$", r"^D\(MAIN\)$",
# # ]
# # WINDOW_PATTERNS = [
# #     r"^W[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# #     r"^W[=\-\s]?(\d+['']\d*[\"']?)\s*[xX×\xd7]\s*(\d+['']\d*[\"']?)$",
# #     r"^W[=\-]?(\d+)'?(\d*)[\"']?[xX×\xd7](\d+)[''\"']?$",
# #     r"^WIN[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# #     r"^WINDOW[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# #     r"^W\d+[\s\n]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# #     r"^W\d+$", r"^WIN\d*$", r"^WINDOW\d*$", r"^FENETRE\d*$",
# # ]
# # ROOM_KEYWORDS = [
# #     "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL",
# #     "BALCONY","STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM",
# #     "C.BEDROOM","MASTER","DRAWING","UTILITY","GARAGE","PASSAGE",
# #     "WASH","BATH","FAMILY","PANTRY","LAUNDRY","TERRACE","VERANDAH",
# #     "ENTRANCE","FOYER","MBEDROOM","CBEDROOM","BEDROOM 2","BEDROOM 3",
# #     "LIVING ROOM","DINING ROOM","STORE ROOM","PRAYER","POWDER",
# #     "CHAMBRE","SALON","CUISINE","SALLE","BAIN","TOILETTE","WC",
# #     "COULOIR","ENTREE","BUREAU","DRESSING","BUANDERIE","SEJOUR",
# #     "TERRASSE","CELLIER","PLACARD","DEGAGEMENT","SALLE DE BAIN",
# #     "SALLE A MANGER","SALON PRIVE","CHAMBRE ENFANT","DOUCHE","MAGASIN",
# # ]
 
# # DIM_SEP = r"[xX×\xd7\*]"   # universal dimension separator (x, X, ×, *)
 
# # # ══════════════════════════════════════════════
# # #  LOAD
# # # ══════════════════════════════════════════════
# # if input_path.lower().endswith(".pdf"):
# #     pages = convert_from_path(input_path, dpi=300)
# # else:
# #     img   = cv2.imread(input_path)
# #     pages = [img] if img is not None else []
 
# # elements   = []
# # raw_images = []
 
# # def preprocess(img):
# #     if not isinstance(img, np.ndarray):
# #         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
# #     gray = cv2.bilateralFilter(gray, 9, 75, 75)
# #     _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# #     return th
 
# # for page in pages:
# #     image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
# #     raw_images.append(image)
# #     raw_results = reader.readtext(preprocess(image))
 
# #     # Merge split detections on same line
# #     merged   = []
# #     used     = set()
# #     raw_sorted = sorted(raw_results, key=lambda r: (round(r[0][0][1]/20), r[0][0][0]))
# #     for i, (bbox, text, prob) in enumerate(raw_sorted):
# #         if i in used or prob < MIN_CONF: continue
# #         x = int(sum([p[0] for p in bbox]) / 4)
# #         y = int(sum([p[1] for p in bbox]) / 4)
# #         cur_text = text.upper().strip()
 
# #         if i+1 < len(raw_sorted) and i+1 not in used:
# #             bbox2, text2, prob2 = raw_sorted[i+1]
# #             x2 = int(sum([p[0] for p in bbox2]) / 4)
# #             y2 = int(sum([p[1] for p in bbox2]) / 4)
# #             same_line = abs(y2-y) < 30
# #             close_x   = abs(x2-x) < 200
 
# #             if same_line and close_x and prob2 >= MIN_CONF:
# #                 next_text    = text2.upper().strip()
# #                 with_space   = cur_text + " " + next_text
# #                 without_space= cur_text + next_text
 
# #                 # Rule 1: Any "ZONE" + digit/label → "ZONE 1", "ZONE 1A" etc.
# #                 if re.match(r"^ZONE$", cur_text) and re.match(r"^\d+[A-Z]?$", next_text):
# #                     merged.append({"text": "ZONE " + next_text, "x": (x+x2)//2, "y": (y+y2)//2})
# #                     used.add(i); used.add(i+1); continue
 
# #                 # Rule 2: "ZONE N" + letter → "ZONE 1A", "ZONE 4B" etc.
# #                 if re.match(r"^ZONE \d+$", cur_text) and re.match(r"^[A-Z]$", next_text):
# #                     merged.append({"text": cur_text + next_text, "x": (x+x2)//2, "y": (y+y2)//2})
# #                     used.add(i); used.add(i+1); continue
 
# #                 # Rule 3: Dimension labels — "W4" + "X4" or "W-4" + "6X4"
# #                 if re.search(r"[WD].*\d+.*[Xx*×].*\d+", without_space):
# #                     merged.append({"text": without_space, "x": (x+x2)//2, "y": (y+y2)//2})
# #                     used.add(i); used.add(i+1); continue
 
# #                 # Rule 4: Any label + dimension — "W1" + "1.35X1.20"
# #                 if re.match(r"^[WD]\d+$", cur_text) and re.search(r"\d+[.*X]\d+", next_text):
# #                     merged.append({"text": with_space, "x": (x+x2)//2, "y": (y+y2)//2})
# #                     used.add(i); used.add(i+1); continue
 
# #                 # Rule 5: IW/EW + spec — "IW" + "T=4" or "EW" + "T=9"
# #                 if re.match(r"^(IW|EW)$", cur_text) and re.search(r"T[=\-]", next_text):
# #                     merged.append({"text": with_space, "x": (x+x2)//2, "y": (y+y2)//2})
# #                     used.add(i); used.add(i+1); continue
 
# #         merged.append({"text": cur_text, "x": x, "y": y})
# #         used.add(i)
# #     elements.extend(merged)
 
# # # ══════════════════════════════════════════════
# # #  CLEAN — normalize ALL OCR noise
# # # ══════════════════════════════════════════════
# # def clean(t):
# #     """
# #     Universal OCR noise cleaner.
# #     Fixes ALL known misreads regardless of drawing source.
# #     """
# #     t = t.upper().strip()
 
# #     # ── 1. Normalize dimension separators ───────────────────────────────
# #     # All of these mean "times" in dimension context
# #     t = t.replace('\xd7','X')   # Unicode ×
# #     t = t.replace('×','X')        # Unicode × (different encoding)
# #     t = t.replace('✕','X')        # Unicode cross
# #     t = t.replace('*','X')        # Asterisk used as multiplier
# #     t = t.replace('·','X')        # Middle dot
 
# #     # ── 2. Fix digit/letter confusion (global rules) ─────────────────────
# #     # OCR commonly confuses:
# #     #   0 ↔ O (zero vs letter O)
# #     #   1 ↔ I or l (one vs letter I/l)
# #     #   5 ↔ S (five vs letter S)
# #     #   4 ↔ A (four vs letter A)
# #     #   6 ↔ G or b (six vs G/b)
# #     #   8 ↔ B (eight vs letter B)
 
# #     # Fix decimal values: T=O.10 → T=0.10, H=2.1O → H=2.10
# #     t = re.sub(r'([TH]=\d*\.)(\d*)O', r'\g<1>\g<2>0', t)   # trailing O → 0
# #     t = re.sub(r'([TH]=)O\.', r'\g<1>0.', t)                  # leading O → 0
 
# #     # Fix H-3.OO → H=3.00, H-2.1O → H=2.10
# #     t = re.sub(r'([TH][-=]\d+\.\d*)O', lambda m: m.group(0)[:-1]+'0', t)
 
# #     # ── 3. Fix door label: D read as 0 by OCR ────────────────────────────
# #     # '01' → 'D1', '02' → 'D2' etc. (only at start of string)
# #     # '0 (MAIN)' → 'D(MAIN)', '0(MAIN)' → 'D(MAIN)'
# #     t = re.sub(r'^0\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
# #     t = re.sub(r'^0\s*\(', 'D(', t)
# #     t = re.sub(r'^0(\d+)$', r'D\1', t)          # 01 → D1, 07 → D7
 
# #     # ── 4. Fix window labels: letter/digit confusion ──────────────────────
# #     # WA→W4, WS→W5, WB→W6, WG→W6, WI→W1, Wl→W1
# #     t = re.sub(r'^W([AAQO])$', 'W4', t)          # WA, WQ → W4
# #     t = re.sub(r'^W([S5])$',   'W5', t)           # WS → W5
# #     t = re.sub(r'^W([Bb8B])$', 'W8', t)           # WB → W8
# #     t = re.sub(r'^W([Gg6])$',  'W6', t)           # WG → W6
# #     t = re.sub(r'^W([Il1])$',  'W1', t)           # WI,Wl → W1
 
# #     # ── 5. Fix ZONE label: letter/digit confusion ─────────────────────────
# #     # ZONE SA → ZONE 5A, ZONE lA → ZONE 1A, ZONE IB → ZONE 1B
# #     t = re.sub(r'^ZONE\s+S([A-Z]?)$', r'ZONE 5\1', t)   # SA→5A, SB→5B
# #     t = re.sub(r'^ZONE\s+I([A-Z]?)$', r'ZONE 1\1', t)   # IA→1A
# #     t = re.sub(r'^ZONE\s+l([A-Z]?)$', r'ZONE 1\1', t)   # lA→1A
# #     t = re.sub(r'^ZONE\s+O([A-Z]?)$', r'ZONE 0\1', t)   # O→0
 
# #     # ── 6. Normalize wall labels ──────────────────────────────────────────
# #     t = re.sub(r"\bEW,\b","EW",t)
# #     t = t.replace("1W","IW").replace("lW","IW").replace("lw","IW").replace("|W","IW")
# #     t = t.replace("EW.","EW").replace("IW.","IW")
 
# #     # ── 7. Normalize dash → equals for spec labels ────────────────────────
# #     # T-9 → T=9, H-10 → H=10, T-0.23 → T=0.23, H-3.00 → H=3.00
# #     t = re.sub(r'\b([TH])-(\d)', r'\1=\2', t)
# #     t = re.sub(r'\b([DW])-(\d)', r'\1=\2', t)
 
# #     # ── 8. Fix common garbled patterns ───────────────────────────────────
# #     t = re.sub(r'\bTZ(\d)', r'T=\1', t)         # TZ4 → T=4
# #     t = re.sub(r'H-L(\d)', r'H=1\1', t)          # H-L0 → H=10
# #     t = t.replace("T-G","T=9").replace("T=G","T=9")# T-G → T=9
# #     t = re.sub(r'\bW-(\d)', r'W=\1', t)         # W-4 → W=4
# #     t = re.sub(r'\bD-(\d)', r'D=\1', t)         # D-3 → D=3
 
# #     # ── 9. Fix D(MAIN) variations ─────────────────────────────────────────
# #     t = re.sub(r'D\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
# #     t = re.sub(r'D\s*\(\s*EXT.*?\)', 'D(EXT)', t)
 
# #     # ── 10. Fix dimension letter/digit confusion ──────────────────────────
# #     # In patterns like "D=AX7" or "W=4XA"
# #     t = re.sub(r'\b([DW]=)(\d*)([A-OQ-Z])(X)', lambda m: m.group(1)+m.group(2)+'4'+m.group(4), t)
# #     t = re.sub(r'(X)([A-OQ-Z])(\b)', lambda m: m.group(1)+'4', t)
 
# #     # ── 11. General whitespace cleanup ───────────────────────────────────
# #     t = re.sub(r"\s+"," ",t)
# #     return t.strip()
 
 
# # for e in elements: e["text"] = clean(e["text"])
 
# # img_h = raw_images[0].shape[0] if raw_images else 1000
# # img_w = raw_images[0].shape[1] if raw_images else 1000
 
# # # Remove legend/footer labels (bottom 20% = legend area)
# # legend_y = img_h * 0.80
# # LEGEND_KEYWORDS = [
# #     "= EXTERNAL WALL","= INTERNAL WALL","EXTERNAL WALL","EW =","IW =",
# #     "EW:","ROOM TINTS","BEDROOMS","BATHROOMS","KITCHEN/DINING",
# #     "W = WINDOW","D = DOOR","T = THICKNESS","H = HEIGHT",
# #     "BEDROOMS / STUDY","BATHROOMS / LIVING","KITCHEN / DINING",
# # ]
# # elements = [e for e in elements if not (
# #     e["y"] > legend_y and
# #     any(kw in e["text"] for kw in LEGEND_KEYWORDS)
# # )]
 
# # # ══════════════════════════════════════════════
# # #  HELPERS
# # # ══════════════════════════════════════════════
# # def dist(a,b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
# # def is_dup(x,y,pos,th=50): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)
# # def nearby(base,r=220): return [e for e in elements if dist(base,e)<r]
 
# # def matches_any(text, patterns):
# #     return any(re.match(p, text) for p in patterns)
 
# # def extract_dim_float(text, patterns):
# #     """Extract WxH dimensions — handles x, X, × all normalized to X after clean()."""
# #     for p in patterns:
# #         m = re.match(p, text)
# #         if m and m.lastindex and m.lastindex >= 2:
# #             try: return float(m.group(1)), float(m.group(2))
# #             except: pass
# #     return None, None
 
# # def find_nearby_dim(base, radius=250):
# #     """
# #     Search nearby text for dimension pattern like:
# #     1.35 X 1.20  or  3.50 X 4.25  or  12'0"X15'0"
# #     Returns (w, h) as floats or (None, None)
# #     """
# #     for e in sorted([e for e in elements if dist(base,e)<radius], key=lambda e: dist(base,e)):
# #         t = e["text"]
# #         # Metric decimal: 1.35 X 1.20
# #         m = re.search(r'(\d+\.\d+)\s*[X\*]\s*(\d+\.\d+)', t)
# #         if m:
# #             return float(m.group(1)), float(m.group(2))
# #         # Imperial feet: 12'0"X15'0"
# #         m = re.search(r"(\d+'\d*[\"']?)\s*X\s*(\d+'\d*[\"']?)", t)
# #         if m:
# #             ww = parse_feet_inches(m.group(1))
# #             ll = parse_feet_inches(m.group(2))
# #             if ww and ll: return float(ww), float(ll)
# #         # Simple integers: 3X7
# #         m = re.search(r'\b(\d+\.?\d*)\s*[X*]\s*(\d+\.?\d*)\b', t)
# #         if m:
# #             ww, ll = float(m.group(1)), float(m.group(2))
# #             if 0.3 <= ww <= 80 and 0.3 <= ll <= 80:
# #                 return ww, ll
# #     return None, None
 
# # def find_val_float(elems, pat):
# #     for e in elems:
# #         txt = re.sub(r'\b([TH])-(\d)', r'\1=\2', e["text"])
# #         m = re.search(pat, txt)
# #         if m:
# #             try: return float(m.group(1))
# #             except: pass
# #     return None
 
# # def find_dir(elems):
# #     dirs = ["NORTH","SOUTH","EAST","WEST","MIDDLE","CENTER","TOP","BOTTOM","LEFT","RIGHT"]
# #     for e in elems:
# #         for d in dirs:
# #             if d in e["text"]: return d.lower()
# #     return None
 
# # def pos_by_location(x, y):
# #     m = 0.15
# #     if   y < img_h*m:      return "north"
# #     elif y > img_h*(1-m):  return "south"
# #     elif x < img_w*m:      return "west"
# #     elif x > img_w*(1-m):  return "east"
# #     return "inner"
 
# # def parse_feet_inches(text):
# #     if not text: return None
# #     text = str(text).replace(" ","")
# #     fm   = re.search(r"(\d+)'", text)
# #     if not fm: return None
# #     feet   = int(fm.group(1))
# #     im     = re.search(r"'(\d+)", text)
# #     fracm  = re.search(r"(\d+)/(\d+)", text)
# #     inches = int(im.group(1)) if im else 0
# #     frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
# #     if inches > 12: inches = int(str(inches)[0])
# #     return math.ceil(feet+(inches+frac)/12)
 
# # def to_ft(val, unit):
# #     """Convert value to feet if needed."""
# #     if val is None: return None
# #     if unit == "m": return round(val * 3.281, 1)
# #     return round(val, 1)
 
# # # ══════════════════════════════════════════════
# # #  DETECT DRAWING UNIT
# # # ══════════════════════════════════════════════
# # def detect_unit():
# #     metric_hits = 0; imperial_hits = 0
# #     for e in elements:
# #         t = e["text"]
# #         if re.search(r"\d+\.\d{2}", t): metric_hits += 1
# #         if re.search(r"\b\d+M\b|\d+\.?\d*\s*M\b", t): metric_hits += 1
# #         if re.search(r"\d+'", t): imperial_hits += 1
# #         if re.search(r"\d+X\d+", t) and not re.search(r"\d+\.\d+", t): imperial_hits += 1
# #     return "m" if metric_hits > imperial_hits else "ft"
 
# # drawing_unit = detect_unit()
 
# # # ══════════════════════════════════════════════
# # #  ZONE DIMENSION PARSER — all formats
# # # ══════════════════════════════════════════════
# # def zone_dims(base, r=500):
# #     for e in elements:
# #         if dist(base,e) > r: continue
# #         t = e["text"]
 
# #         # Metric decimal: 3.50 X 4.25
# #         m = re.search(r'(\d+\.\d+)\s*[X\*]\s*(\d+\.\d+)', t)
# #         if m:
# #             ww = to_ft(float(m.group(1)), drawing_unit)
# #             ll = to_ft(float(m.group(2)), drawing_unit)
# #             if ww and ll and 4<=ww<=150 and 4<=ll<=150:
# #                 return round(min(ww,ll),1), round(max(ww,ll),1)
 
# #         # Imperial: 12'0"X15'0"
# #         m = re.search(r"(\d+['']\d+[\"']?)\s*X\s*(\d+['']\d+[\"']?)", t)
# #         if m:
# #             ww = parse_feet_inches(m.group(1))
# #             ll = parse_feet_inches(m.group(2))
# #             if ww and ll and 4<=ww<=100 and 4<=ll<=100:
# #                 return min(ww,ll), max(ww,ll)
 
# #         # Imperial no separator: 12'0"15'0"
# #         m = re.search(r"(\d+['']\d+)[\"']\s*(\d+['']\d+)", t)
# #         if m:
# #             ww = parse_feet_inches(m.group(1))
# #             ll = parse_feet_inches(m.group(2))
# #             if ww and ll and 4<=ww<=100 and 4<=ll<=100:
# #                 return min(ww,ll), max(ww,ll)
 
# #         # Simple integers NxM
# #         m2 = re.search(r'(\d{1,2})\s*[X\*]\s*(\d{1,2})', t)
# #         if m2:
# #             ww,ll = int(m2.group(1)), int(m2.group(2))
# #             if 5<=ww<=80 and 5<=ll<=80 and not matches_any(t, WINDOW_PATTERNS+DOOR_PATTERNS):
# #                 return min(ww,ll), max(ww,ll)
 
# #     return None, None
 
# # # ══════════════════════════════════════════════
# # #  STRATEGY 1 — Text label detection
# # # ══════════════════════════════════════════════
# # ew_anchors, ew_seen = [], []
# # iw_anchors, iw_seen = [], []
 
# # for e in elements:
# #     t = e["text"]
# #     if matches_any(t, EW_PATTERNS):
# #         if not is_dup(e["x"],e["y"],ew_seen):
# #             ew_anchors.append(dict(e)); ew_seen.append((e["x"],e["y"]))
# #     elif matches_any(t, IW_PATTERNS):
# #         if not is_dup(e["x"],e["y"],iw_seen):
# #             iw_anchors.append(dict(e)); iw_seen.append((e["x"],e["y"]))
 
# # external_walls = {}
# # internal_walls = {}
 
# # def build_from_text_labels():
# #     for i, anchor in enumerate(ew_anchors):
# #         eid = f"ew{i+1}"
# #         nb  = nearby(anchor, 280)
# #         t_v = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
# #         h_v = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
# #         if t_v and drawing_unit == "m":
# #             t_display = round(t_v * 39.37)
# #             h_display = round(h_v * 3.281) if h_v else 10
# #         else:
# #             t_display = int(t_v) if t_v else 9
# #             h_display = int(h_v) if h_v else 10
# #         d = find_dir(nb) or pos_by_location(anchor["x"], anchor["y"])
# #         external_walls[eid] = {
# #             "id":eid, "position":d,
# #             "length_ft":None, "thickness_in":t_display, "height_ft":h_display,
# #             "connected":{"windows":[],"doors":[],"internal_walls":[]}
# #         }
# #         anchor["id"] = eid
 
# #     for i, anchor in enumerate(iw_anchors):
# #         iwid = f"iw{i+1}"
# #         nb   = nearby(anchor, 280)
# #         t_v  = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
# #         h_v  = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
# #         if t_v and drawing_unit == "m":
# #             t_display = round(t_v * 39.37)
# #             h_display = round(h_v * 3.281) if h_v else 10
# #         else:
# #             t_display = int(t_v) if t_v else 4
# #             h_display = int(h_v) if h_v else 10
# #         d    = find_dir(nb)
# #         conn = []
# #         for ea in sorted(ew_anchors, key=lambda a: dist(anchor,a))[:2]:
# #             conn.append(ea["id"])
# #             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
# #                 external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
# #         internal_walls[iwid] = {
# #             "id":iwid, "position":d,
# #             "thickness_in":t_display, "height_ft":h_display, "length_ft":None,
# #             "connects_external":conn,
# #             "connected":{"doors":[],"windows":[]}
# #         }
# #         anchor["id"] = iwid
 
# # # ══════════════════════════════════════════════
# # #  STRATEGY 2 — OpenCV line detection
# # # ══════════════════════════════════════════════
# # def build_from_opencv():
# #     img   = raw_images[0]
# #     gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
# #     kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# #     gray  = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
# #     edges = cv2.Canny(gray,40,120,apertureSize=3)
# #     lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,
# #                              minLineLength=img_w//12,maxLineGap=25)
# #     h_lines,v_lines=[],[]
# #     if lines is not None:
# #         for line in lines:
# #             x1,y1,x2,y2=line[0]
# #             angle=abs(math.degrees(math.atan2(y2-y1,x2-x1)))
# #             length=math.hypot(x2-x1,y2-y1)
# #             if angle<15 or angle>165: h_lines.append((min(x1,x2),max(x1,x2),(y1+y2)//2,length))
# #             elif 75<angle<105:        v_lines.append(((x1+x2)//2,min(y1,y2),max(y1,y2),length))
 
# #     def cluster(lines,axis,gap=35):
# #         if not lines: return []
# #         lines=sorted(lines,key=lambda l:l[2] if axis=='h' else l[0])
# #         out=[lines[0]]
# #         for ln in lines[1:]:
# #             coord=ln[2] if axis=='h' else ln[0]; prev=out[-1][2] if axis=='h' else out[-1][0]
# #             if abs(coord-prev)<gap: out[-1]=ln if ln[3]>out[-1][3] else out[-1]
# #             else: out.append(ln)
# #         return out
 
# #     ew_c=0; iw_c=0; bm=0.12
# #     for (x1,x2,y,l) in cluster(h_lines,'h'):
# #         lft=round(l/(img_w/30))
# #         if y<img_h*bm or y>img_h*(1-bm):
# #             ew_c+=1; eid=f"ew{ew_c}"; pos="north" if y<img_h/2 else "south"
# #             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# #             ew_anchors.append({"x":(x1+x2)//2,"y":y,"id":eid})
# #         else:
# #             iw_c+=1; iwid=f"iw{iw_c}"
# #             internal_walls[iwid]={"id":iwid,"position":"horizontal","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
# #             iw_anchors.append({"x":(x1+x2)//2,"y":y,"id":iwid})
# #     for (x,y1,y2,l) in cluster(v_lines,'v'):
# #         lft=round(l/(img_h/30))
# #         if x<img_w*bm or x>img_w*(1-bm):
# #             ew_c+=1; eid=f"ew{ew_c}"; pos="west" if x<img_w/2 else "east"
# #             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# #             ew_anchors.append({"x":x,"y":(y1+y2)//2,"id":eid})
# #         else:
# #             iw_c+=1; iwid=f"iw{iw_c}"
# #             internal_walls[iwid]={"id":iwid,"position":"vertical","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
# #             iw_anchors.append({"x":x,"y":(y1+y2)//2,"id":iwid})
# #     for iwa in iw_anchors:
# #         iwid=iwa["id"]
# #         for ea in sorted(ew_anchors,key=lambda a:dist(iwa,a))[:2]:
# #             if ea["id"] not in internal_walls[iwid]["connects_external"]: internal_walls[iwid]["connects_external"].append(ea["id"])
# #             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]: external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
# #     return ew_c>0
 
# # # ══════════════════════════════════════════════
# # #  STRATEGY 3 — Auto 4 boundary walls
# # # ══════════════════════════════════════════════
# # def build_fallback():
# #     for pos,eid,x,y in [("north","ew1",img_w//2,5),("east","ew2",img_w-5,img_h//2),
# #                          ("south","ew3",img_w//2,img_h-5),("west","ew4",5,img_h//2)]:
# #         external_walls[eid]={"id":eid,"position":pos,"length_ft":None,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# #         ew_anchors.append({"x":x,"y":y,"id":eid})
 
# # # ══════════════════════════════════════════════
# # #  RUN STRATEGY CHAIN
# # # ══════════════════════════════════════════════
# # strategy = "auto_boundary"
# # if ew_anchors or iw_anchors:
# #     build_from_text_labels(); strategy="text_labels"
# # elif raw_images and build_from_opencv():
# #     strategy="opencv_lines"
# # else:
# #     build_fallback()
 
# # # ══════════════════════════════════════════════
# # #  DOORS — universal detection
# # # ══════════════════════════════════════════════
# # door_list, door_seen = [], []
# # for e in elements:
# #     t=e["text"]
# #     w,h=extract_dim_float(t,DOOR_PATTERNS)
# #     if w and h:
# #         if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
# #         if not is_dup(e["x"],e["y"],door_seen):
# #             door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
# #             door_seen.append((e["x"],e["y"]))
# #     # Numbered door labels: D1, D(Main) — look for nearby dimensions
# #     elif re.match(r"^D\d+$|^DOOR\d+$|^DR\d+$|^PORTE\d*$|^D\(.*\)$|^DMAIN$", t) and not is_dup(e["x"],e["y"],door_seen):
# #         dw, dh = find_nearby_dim(e, radius=250)
# #         if dw and dh:
# #             if drawing_unit=="m" and dw<5: dw=round(dw*3.281,1); dh=round(dh*3.281,1)
# #         else:
# #             dw, dh = (4.0, 7.0) if "MAIN" in t else (3.0, 7.0)
# #         door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":dw,"height_ft":dh})
# #         door_seen.append((e["x"],e["y"]))
 
# # # ══════════════════════════════════════════════
# # #  WINDOWS — universal detection
# # # ══════════════════════════════════════════════
# # window_list, win_seen = [], []
# # for e in elements:
# #     t=e["text"]
# #     w,h=extract_dim_float(t,WINDOW_PATTERNS)
# #     if w and h:
# #         if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
# #         if not is_dup(e["x"],e["y"],win_seen):
# #             window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
# #             win_seen.append((e["x"],e["y"]))
# #     # Numbered window labels: W1, W2 — look for nearby dimensions
# #     elif re.match(r"^W\d+$|^WIN\d+$|^WINDOW\d+$|^FENETRE\d*$", t) and not is_dup(e["x"],e["y"],win_seen):
# #         wv, hv = find_nearby_dim(e, radius=250)
# #         if wv and hv:
# #             if drawing_unit=="m" and wv<5: wv=round(wv*3.281,1); hv=round(hv*3.281,1)
# #         else:
# #             wv, hv = 4.0, 4.0
# #         window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":wv,"height_ft":hv})
# #         win_seen.append((e["x"],e["y"]))
 
# # # ══════════════════════════════════════════════
# # #  ASSIGN on_wall + BACK-FILL
# # # ══════════════════════════════════════════════
# # def nearest_wall(pos):
# #     best,bd=None,999999
# #     for a in ew_anchors+iw_anchors:
# #         d=dist(pos,a)
# #         if d<bd: bd,best=d,a["id"]
# #     return best
 
# # doors_out={}
# # for d in door_list:
# #     ow=nearest_wall({"x":d["x"],"y":d["y"]})
# #     doors_out[d["id"]]={"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],"on_wall":ow}
# #     if ow:
# #         wl=external_walls.get(ow) or internal_walls.get(ow)
# #         if wl and d["id"] not in wl["connected"]["doors"]: wl["connected"]["doors"].append(d["id"])
 
# # windows_out={}
# # for w in window_list:
# #     ow=nearest_wall({"x":w["x"],"y":w["y"]})
# #     windows_out[w["id"]]={"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],"on_wall":ow}
# #     if ow:
# #         wl=external_walls.get(ow) or internal_walls.get(ow)
# #         if wl and w["id"] not in wl["connected"]["windows"]: wl["connected"]["windows"].append(w["id"])
 
 
# # # ══════════════════════════════════════════════
# # #  AUTO PIXEL-PER-FOOT SCALE DETECTION
# # #  Computes real px/ft from dimension labels in
# # #  the drawing itself — works for any scale/size
# # # ══════════════════════════════════════════════
# # def _auto_px_per_ft():
# #     """
# #     Find two zone labels with known dimensions and known pixel distance
# #     to compute actual px_per_ft for THIS drawing.
# #     Falls back to img_w / estimated_total_ft.
# #     """
# #     # Strategy A: find overall building dimension label (e.g. "10.00 m" or "50'0"")
# #     for e in elements:
# #         t = e["text"]
# #         # Metric: "10.00 M" or "10.00M"
# #         m = re.search(r"(\d+\.\d+)\s*M", t)
# #         if m:
# #             total_m  = float(m.group(1))
# #             total_ft = total_m * 3.281
# #             # This label is near the edge — use img_w as the pixel span
# #             return img_w / total_ft
# #         # Imperial: "50'0"" total dimension
# #         m = re.search(r"(\d+)'\d*\s*[xX*]\s*(\d+)'\d*", t)
# #         if m:
# #             w_ft = float(m.group(1))
# #             h_ft = float(m.group(2))
# #             total_ft = max(w_ft, h_ft)
# #             return img_w / total_ft
 
# #     # Strategy B: use two zone labels whose real distance we can estimate
# #     # Pick zones with known area and compute from pixel positions
# #     zone_entries = []
# #     for e in elements:
# #         if re.match(r"^ZONE\s*\d+[A-Z]?$", e["text"]):
# #             wft, lft = zone_dims(e)
# #             if wft and lft:
# #                 zone_entries.append((e, wft, lft))
 
# #     if len(zone_entries) >= 2:
# #         # Use the two zones furthest apart
# #         best_d, best_scale = 0, None
# #         for i in range(len(zone_entries)):
# #             for j in range(i+1, len(zone_entries)):
# #                 ea, wa, la = zone_entries[i]
# #                 eb, wb, lb = zone_entries[j]
# #                 px_dist = dist(ea, eb)
# #                 # Estimated real distance = average of their dimensions
# #                 real_ft = (max(wa,la) + max(wb,lb)) / 2
# #                 if px_dist > best_d and real_ft > 0:
# #                     best_d = px_dist
# #                     best_scale = px_dist / real_ft
# #         if best_scale and 5 < best_scale < 500:
# #             return best_scale
 
# #     # Strategy C: fallback based on unit
# #     if drawing_unit == "m":
# #         return img_w / 32.8   # 10m = 32.8ft typical
# #     return img_w / 40.0        # 40ft typical imperial
 
# # # ══════════════════════════════════════════════
# # #  ZONES
# # # ══════════════════════════════════════════════
# # ZONE_PAT=re.compile(r"^ZONE\s*\d+[A-Z]?$")
# # zones_out={}
 
# # def nearest_room(base,r=400):
# #     for e in sorted(elements,key=lambda e:dist(base,e)):
# #         if dist(base,e)>r: break
# #         for kw in ROOM_KEYWORDS:
# #             if kw in e["text"]: return e["text"]
# #     return "UNKNOWN"
 
# # for e in elements:
# #     if ZONE_PAT.match(e["text"]):
# #         zid=re.sub(r"\s+","",e["text"])
# #         name=nearest_room(e)
# #         wft,lft=zone_dims(e)
# #         area=round(wft*lft,2) if wft and lft else None
# #         px_ft = _auto_px_per_ft()
# #         zp = math.sqrt(area) * px_ft * 0.7 if area else img_w * 0.30
 
# #         conn_ew=list(dict.fromkeys(a["id"] for a in ew_anchors if dist(e,a)<zp*1.3))
# #         conn_iw=list(dict.fromkeys(a["id"] for a in iw_anchors if dist(e,a)<zp*1.1))
# #         conn_d =list(dict.fromkeys(d["id"] for d in door_list   if dist(e,{"x":d["x"],"y":d["y"]})<zp))
# #         conn_w =list(dict.fromkeys(w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp))
 
# #         zones_out[zid]={
# #             "id":zid,"label":name,"width_ft":wft,"length_ft":lft,"area_sqft":area,
# #             "connected_external_walls":conn_ew,
# #             "connected_internal_walls":conn_iw,
# #             "total_walls_connected":len(conn_ew)+len(conn_iw),
# #             "doors":conn_d,"windows":conn_w
# #         }
 
# # # ══════════════════════════════════════════════
# # #  DEDUP SUMMARY
# # # ══════════════════════════════════════════════
# # all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
# # all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])
 
# # for zid in zones_out:
# #     zones_out[zid]["connected_external_walls"] = list(dict.fromkeys(zones_out[zid]["connected_external_walls"]))
# #     zones_out[zid]["connected_internal_walls"] = list(dict.fromkeys(zones_out[zid]["connected_internal_walls"]))
# #     zones_out[zid]["doors"]   = list(dict.fromkeys(zones_out[zid]["doors"]))
# #     zones_out[zid]["windows"] = list(dict.fromkeys(zones_out[zid]["windows"]))
# #     zones_out[zid]["total_walls_connected"] = (
# #         len(zones_out[zid]["connected_external_walls"]) +
# #         len(zones_out[zid]["connected_internal_walls"])
# #     )
 
# # total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]), 2)
 
# # # ══════════════════════════════════════════════
# # #  OUTPUT
# # # ══════════════════════════════════════════════
# # print(json.dumps({
# #     "external_walls": external_walls,
# #     "internal_walls": internal_walls,
# #     "doors":    doors_out,
# #     "windows":  windows_out,
# #     "zones":    zones_out,
# #     "summary": {
# #         "total_external_walls":    len(external_walls),
# #         "total_internal_walls":    len(internal_walls),
# #         "total_doors":             len(doors_out),
# #         "total_windows":           len(windows_out),
# #         "total_zones":             len(zones_out),
# #         "total_area_sqft":         total_area,
# #         "unique_doors_in_zones":   len(all_zone_doors),
# #         "unique_windows_in_zones": len(all_zone_windows),
# #         "drawing_unit":            drawing_unit,
# #         "detection_strategy":      strategy
# #     }
# # }))

# # # import cv2, easyocr, re, json, numpy as np, math, sys
# # # from pdf2image import convert_from_path
 
# # # MIN_CONF   = 0.28
# # # reader     = easyocr.Reader(['en'], gpu=False)
# # # input_path = sys.argv[1]
 
# # # # ══════════════════════════════════════════════
# # # #  UNIVERSAL LABEL PATTERNS
# # # # ══════════════════════════════════════════════
# # # EW_PATTERNS = [
# # #     r"^EW[\s,=T\d\.]*$", r"^EW\d+[A-Z]?$", r"^E\.W\.?\d*$",
# # #     r"^E\d+$", r"^EXT[\s\-]?WALL\d*$", r"^EXTERNAL[\s\-]?WALL\d*$",
# # #     r"^OUTER[\s\-]?WALL\d*$", r"^BOUNDARY[\s\-]?WALL\d*$",
# # #     r"^MUR[\s\-]?EXT\d*$", r"^MUR[\s\-]?EXTERIEUR\d*$",
# # # ]
# # # IW_PATTERNS = [
# # #     r"^IW[\s,=T\d\.]*$", r"^IW\d+[A-Z]?$", r"^I\.W\.?\d*$",
# # #     r"^IW[\s\(].*$",
# # #     r"^I\d+$", r"^INT[\s\-]?WALL\d*$", r"^INTERNAL[\s\-]?WALL\d*$",
# # #     r"^INNER[\s\-]?WALL\d*$", r"^PARTITION[\s\-]?\d*$",
# # #     r"^DIVIDER[\s\-]?\d*$", r"^PW\d*$",
# # #     r"^MUR[\s\-]?INT\d*$", r"^CLOISON\d*$",
# # # ]
# # # DOOR_PATTERNS = [
# # #     r"^D[=\-\s]?(\d+\.?\d*)[xX×X\xd7](\d+\.?\d*)$",
# # #     r"^DOOR[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # #     r"^DR[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # #     r"^D[\s\(].*?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*).*$",
# # #     r"^D\(MAIN\).*?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*).*$",
# # #     r"^D\d+$", r"^DOOR\d*$", r"^DR\d*$", r"^PORTE\d*$",
# # #     r"^D\(.*\)$", r"^DMAIN$", r"^D\(MAIN\)$",
# # # ]
# # # WINDOW_PATTERNS = [
# # #     r"^W[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # #     r"^W[=\-\s]?(\d+['']\d*[\"']?)\s*[xX×\xd7]\s*(\d+['']\d*[\"']?)$",
# # #     r"^W[=\-]?(\d+)'?(\d*)[\"']?[xX×\xd7](\d+)[''\"']?$",
# # #     r"^WIN[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # #     r"^WINDOW[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # #     r"^W\d+[\s\n]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # #     r"^W\d+$", r"^WIN\d*$", r"^WINDOW\d*$", r"^FENETRE\d*$",
# # # ]
# # # ROOM_KEYWORDS = [
# # #     "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL",
# # #     "BALCONY","STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM",
# # #     "C.BEDROOM","MASTER","DRAWING","UTILITY","GARAGE","PASSAGE",
# # #     "WASH","BATH","FAMILY","PANTRY","LAUNDRY","TERRACE","VERANDAH",
# # #     "ENTRANCE","FOYER","MBEDROOM","CBEDROOM","BEDROOM 2","BEDROOM 3",
# # #     "LIVING ROOM","DINING ROOM","STORE ROOM","PRAYER","POWDER",
# # #     "CHAMBRE","SALON","CUISINE","SALLE","BAIN","TOILETTE","WC",
# # #     "COULOIR","ENTREE","BUREAU","DRESSING","BUANDERIE","SEJOUR",
# # #     "TERRASSE","CELLIER","PLACARD","DEGAGEMENT","SALLE DE BAIN",
# # #     "SALLE A MANGER","SALON PRIVE","CHAMBRE ENFANT","DOUCHE","MAGASIN",
# # # ]
 
# # # DIM_SEP = r"[xX×\xd7\*]"   # universal dimension separator (x, X, ×, *)
 
# # # # ══════════════════════════════════════════════
# # # #  LOAD
# # # # ══════════════════════════════════════════════
# # # if input_path.lower().endswith(".pdf"):
# # #     pages = convert_from_path(input_path, dpi=300)
# # # else:
# # #     img   = cv2.imread(input_path)
# # #     pages = [img] if img is not None else []
 
# # # elements   = []
# # # raw_images = []
 
# # # def preprocess(img):
# # #     if not isinstance(img, np.ndarray):
# # #         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
# # #     gray = cv2.bilateralFilter(gray, 9, 75, 75)
# # #     _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # #     return th
 
# # # for page in pages:
# # #     image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
# # #     raw_images.append(image)
# # #     raw_results = reader.readtext(preprocess(image))
 
# # #     # Merge split detections on same line
# # #     merged   = []
# # #     used     = set()
# # #     raw_sorted = sorted(raw_results, key=lambda r: (round(r[0][0][1]/20), r[0][0][0]))
# # #     for i, (bbox, text, prob) in enumerate(raw_sorted):
# # #         if i in used or prob < MIN_CONF: continue
# # #         x = int(sum([p[0] for p in bbox]) / 4)
# # #         y = int(sum([p[1] for p in bbox]) / 4)
# # #         if i+1 < len(raw_sorted) and i+1 not in used:
# # #             bbox2, text2, prob2 = raw_sorted[i+1]
# # #             x2 = int(sum([p[0] for p in bbox2]) / 4)
# # #             y2 = int(sum([p[1] for p in bbox2]) / 4)
# # #             if abs(y2-y) < 25 and abs(x2-x) < 130 and prob2 >= MIN_CONF:
# # #                 merged_text = (text + text2).upper().strip()
# # #                 if re.search(r'[WD].*\d+.*[Xx×\xd7].*\d+', merged_text):
# # #                     merged.append({"text": merged_text, "x": (x+x2)//2, "y": (y+y2)//2})
# # #                     used.add(i); used.add(i+1)
# # #                     continue
# # #         merged.append({"text": text.upper().strip(), "x": x, "y": y})
# # #         used.add(i)
# # #     elements.extend(merged)
 
# # # # ══════════════════════════════════════════════
# # # #  CLEAN — normalize ALL OCR noise
# # # # ══════════════════════════════════════════════
# # # def clean(t):
# # #     """
# # #     Universal OCR noise cleaner.
# # #     Fixes ALL known misreads regardless of drawing source.
# # #     """
# # #     t = t.upper().strip()
 
# # #     # ── 1. Normalize dimension separators ───────────────────────────────
# # #     # All of these mean "times" in dimension context
# # #     t = t.replace('\xd7','X')   # Unicode ×
# # #     t = t.replace('×','X')        # Unicode × (different encoding)
# # #     t = t.replace('✕','X')        # Unicode cross
# # #     t = t.replace('*','X')        # Asterisk used as multiplier
# # #     t = t.replace('·','X')        # Middle dot
 
# # #     # ── 2. Fix digit/letter confusion (global rules) ─────────────────────
# # #     # OCR commonly confuses:
# # #     #   0 ↔ O (zero vs letter O)
# # #     #   1 ↔ I or l (one vs letter I/l)
# # #     #   5 ↔ S (five vs letter S)
# # #     #   4 ↔ A (four vs letter A)
# # #     #   6 ↔ G or b (six vs G/b)
# # #     #   8 ↔ B (eight vs letter B)
 
# # #     # Fix decimal values: T=O.10 → T=0.10, H=2.1O → H=2.10
# # #     t = re.sub(r'([TH]=\d*\.)(\d*)O', r'\g<1>\g<2>0', t)   # trailing O → 0
# # #     t = re.sub(r'([TH]=)O\.', r'\g<1>0.', t)                  # leading O → 0
 
# # #     # Fix H-3.OO → H=3.00, H-2.1O → H=2.10
# # #     t = re.sub(r'([TH][-=]\d+\.\d*)O', lambda m: m.group(0)[:-1]+'0', t)
 
# # #     # ── 3. Fix door label: D read as 0 by OCR ────────────────────────────
# # #     # '01' → 'D1', '02' → 'D2' etc. (only at start of string)
# # #     # '0 (MAIN)' → 'D(MAIN)', '0(MAIN)' → 'D(MAIN)'
# # #     t = re.sub(r'^0\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
# # #     t = re.sub(r'^0\s*\(', 'D(', t)
# # #     t = re.sub(r'^0(\d+)$', r'D\1', t)          # 01 → D1, 07 → D7
 
# # #     # ── 4. Fix window labels: letter/digit confusion ──────────────────────
# # #     # WA→W4, WS→W5, WB→W6, WG→W6, WI→W1, Wl→W1
# # #     t = re.sub(r'^W([AAQO])$', 'W4', t)          # WA, WQ → W4
# # #     t = re.sub(r'^W([S5])$',   'W5', t)           # WS → W5
# # #     t = re.sub(r'^W([Bb8B])$', 'W8', t)           # WB → W8
# # #     t = re.sub(r'^W([Gg6])$',  'W6', t)           # WG → W6
# # #     t = re.sub(r'^W([Il1])$',  'W1', t)           # WI,Wl → W1
 
# # #     # ── 5. Fix ZONE label: letter/digit confusion ─────────────────────────
# # #     # ZONE SA → ZONE 5A, ZONE lA → ZONE 1A, ZONE IB → ZONE 1B
# # #     t = re.sub(r'^ZONE\s+S([A-Z]?)$', r'ZONE 5\1', t)   # SA→5A, SB→5B
# # #     t = re.sub(r'^ZONE\s+I([A-Z]?)$', r'ZONE 1\1', t)   # IA→1A
# # #     t = re.sub(r'^ZONE\s+l([A-Z]?)$', r'ZONE 1\1', t)   # lA→1A
# # #     t = re.sub(r'^ZONE\s+O([A-Z]?)$', r'ZONE 0\1', t)   # O→0
 
# # #     # ── 6. Normalize wall labels ──────────────────────────────────────────
# # #     t = re.sub(r"\bEW,\b","EW",t)
# # #     t = t.replace("1W","IW").replace("lW","IW").replace("lw","IW").replace("|W","IW")
# # #     t = t.replace("EW.","EW").replace("IW.","IW")
 
# # #     # ── 7. Normalize dash → equals for spec labels ────────────────────────
# # #     # T-9 → T=9, H-10 → H=10, T-0.23 → T=0.23, H-3.00 → H=3.00
# # #     t = re.sub(r'\b([TH])-(\d)', r'\1=\2', t)
# # #     t = re.sub(r'\b([DW])-(\d)', r'\1=\2', t)
 
# # #     # ── 8. Fix common garbled patterns ───────────────────────────────────
# # #     t = re.sub(r'\bTZ(\d)', r'T=\1', t)         # TZ4 → T=4
# # #     t = re.sub(r'H-L(\d)', r'H=1\1', t)          # H-L0 → H=10
# # #     t = t.replace("T-G","T=9").replace("T=G","T=9")# T-G → T=9
# # #     t = re.sub(r'\bW-(\d)', r'W=\1', t)         # W-4 → W=4
# # #     t = re.sub(r'\bD-(\d)', r'D=\1', t)         # D-3 → D=3
 
# # #     # ── 9. Fix D(MAIN) variations ─────────────────────────────────────────
# # #     t = re.sub(r'D\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
# # #     t = re.sub(r'D\s*\(\s*EXT.*?\)', 'D(EXT)', t)
 
# # #     # ── 10. Fix dimension letter/digit confusion ──────────────────────────
# # #     # In patterns like "D=AX7" or "W=4XA"
# # #     t = re.sub(r'\b([DW]=)(\d*)([A-OQ-Z])(X)', lambda m: m.group(1)+m.group(2)+'4'+m.group(4), t)
# # #     t = re.sub(r'(X)([A-OQ-Z])(\b)', lambda m: m.group(1)+'4', t)
 
# # #     # ── 11. General whitespace cleanup ───────────────────────────────────
# # #     t = re.sub(r"\s+"," ",t)
# # #     return t.strip()
 
 
# # # for e in elements: e["text"] = clean(e["text"])
 
# # # img_h = raw_images[0].shape[0] if raw_images else 1000
# # # img_w = raw_images[0].shape[1] if raw_images else 1000
 
# # # # Remove legend/footer labels (bottom 20% = legend area)
# # # legend_y = img_h * 0.80
# # # LEGEND_KEYWORDS = [
# # #     "= EXTERNAL WALL","= INTERNAL WALL","EXTERNAL WALL","EW =","IW =",
# # #     "EW:","ROOM TINTS","BEDROOMS","BATHROOMS","KITCHEN/DINING",
# # #     "W = WINDOW","D = DOOR","T = THICKNESS","H = HEIGHT",
# # #     "BEDROOMS / STUDY","BATHROOMS / LIVING","KITCHEN / DINING",
# # # ]
# # # elements = [e for e in elements if not (
# # #     e["y"] > legend_y and
# # #     any(kw in e["text"] for kw in LEGEND_KEYWORDS)
# # # )]
 
# # # # ══════════════════════════════════════════════
# # # #  HELPERS
# # # # ══════════════════════════════════════════════
# # # def dist(a,b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
# # # def is_dup(x,y,pos,th=50): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)
# # # def nearby(base,r=220): return [e for e in elements if dist(base,e)<r]
 
# # # def matches_any(text, patterns):
# # #     return any(re.match(p, text) for p in patterns)
 
# # # def extract_dim_float(text, patterns):
# # #     """Extract WxH dimensions — handles x, X, × all normalized to X after clean()."""
# # #     for p in patterns:
# # #         m = re.match(p, text)
# # #         if m and m.lastindex and m.lastindex >= 2:
# # #             try: return float(m.group(1)), float(m.group(2))
# # #             except: pass
# # #     return None, None
 
# # # def find_nearby_dim(base, radius=250):
# # #     """
# # #     Search nearby text for dimension pattern like:
# # #     1.35 X 1.20  or  3.50 X 4.25  or  12'0"X15'0"
# # #     Returns (w, h) as floats or (None, None)
# # #     """
# # #     for e in sorted([e for e in elements if dist(base,e)<radius], key=lambda e: dist(base,e)):
# # #         t = e["text"]
# # #         # Metric decimal: 1.35 X 1.20
# # #         m = re.search(r'(\d+\.\d+)\s*[X\*]\s*(\d+\.\d+)', t)
# # #         if m:
# # #             return float(m.group(1)), float(m.group(2))
# # #         # Imperial feet: 12'0"X15'0"
# # #         m = re.search(r"(\d+'\d*[\"']?)\s*X\s*(\d+'\d*[\"']?)", t)
# # #         if m:
# # #             ww = parse_feet_inches(m.group(1))
# # #             ll = parse_feet_inches(m.group(2))
# # #             if ww and ll: return float(ww), float(ll)
# # #         # Simple integers: 3X7
# # #         m = re.search(r'\b(\d+\.?\d*)\s*[X*]\s*(\d+\.?\d*)\b', t)
# # #         if m:
# # #             ww, ll = float(m.group(1)), float(m.group(2))
# # #             if 0.3 <= ww <= 80 and 0.3 <= ll <= 80:
# # #                 return ww, ll
# # #     return None, None
 
# # # def find_val_float(elems, pat):
# # #     for e in elems:
# # #         txt = re.sub(r'\b([TH])-(\d)', r'\1=\2', e["text"])
# # #         m = re.search(pat, txt)
# # #         if m:
# # #             try: return float(m.group(1))
# # #             except: pass
# # #     return None
 
# # # def find_dir(elems):
# # #     dirs = ["NORTH","SOUTH","EAST","WEST","MIDDLE","CENTER","TOP","BOTTOM","LEFT","RIGHT"]
# # #     for e in elems:
# # #         for d in dirs:
# # #             if d in e["text"]: return d.lower()
# # #     return None
 
# # # def pos_by_location(x, y):
# # #     m = 0.15
# # #     if   y < img_h*m:      return "north"
# # #     elif y > img_h*(1-m):  return "south"
# # #     elif x < img_w*m:      return "west"
# # #     elif x > img_w*(1-m):  return "east"
# # #     return "inner"
 
# # # def parse_feet_inches(text):
# # #     if not text: return None
# # #     text = str(text).replace(" ","")
# # #     fm   = re.search(r"(\d+)'", text)
# # #     if not fm: return None
# # #     feet   = int(fm.group(1))
# # #     im     = re.search(r"'(\d+)", text)
# # #     fracm  = re.search(r"(\d+)/(\d+)", text)
# # #     inches = int(im.group(1)) if im else 0
# # #     frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
# # #     if inches > 12: inches = int(str(inches)[0])
# # #     return math.ceil(feet+(inches+frac)/12)
 
# # # def to_ft(val, unit):
# # #     """Convert value to feet if needed."""
# # #     if val is None: return None
# # #     if unit == "m": return round(val * 3.281, 1)
# # #     return round(val, 1)
 
# # # # ══════════════════════════════════════════════
# # # #  DETECT DRAWING UNIT
# # # # ══════════════════════════════════════════════
# # # def detect_unit():
# # #     metric_hits = 0; imperial_hits = 0
# # #     for e in elements:
# # #         t = e["text"]
# # #         if re.search(r"\d+\.\d{2}", t): metric_hits += 1
# # #         if re.search(r"\b\d+M\b|\d+\.?\d*\s*M\b", t): metric_hits += 1
# # #         if re.search(r"\d+'", t): imperial_hits += 1
# # #         if re.search(r"\d+X\d+", t) and not re.search(r"\d+\.\d+", t): imperial_hits += 1
# # #     return "m" if metric_hits > imperial_hits else "ft"
 
# # # drawing_unit = detect_unit()
 
# # # # ══════════════════════════════════════════════
# # # #  ZONE DIMENSION PARSER — all formats
# # # # ══════════════════════════════════════════════
# # # def zone_dims(base, r=500):
# # #     for e in elements:
# # #         if dist(base,e) > r: continue
# # #         t = e["text"]
 
# # #         # Metric decimal: 3.50 X 4.25
# # #         m = re.search(r'(\d+\.\d+)\s*[X\*]\s*(\d+\.\d+)', t)
# # #         if m:
# # #             ww = to_ft(float(m.group(1)), drawing_unit)
# # #             ll = to_ft(float(m.group(2)), drawing_unit)
# # #             if ww and ll and 4<=ww<=150 and 4<=ll<=150:
# # #                 return round(min(ww,ll),1), round(max(ww,ll),1)
 
# # #         # Imperial: 12'0"X15'0"
# # #         m = re.search(r"(\d+['']\d+[\"']?)\s*X\s*(\d+['']\d+[\"']?)", t)
# # #         if m:
# # #             ww = parse_feet_inches(m.group(1))
# # #             ll = parse_feet_inches(m.group(2))
# # #             if ww and ll and 4<=ww<=100 and 4<=ll<=100:
# # #                 return min(ww,ll), max(ww,ll)
 
# # #         # Imperial no separator: 12'0"15'0"
# # #         m = re.search(r"(\d+['']\d+)[\"']\s*(\d+['']\d+)", t)
# # #         if m:
# # #             ww = parse_feet_inches(m.group(1))
# # #             ll = parse_feet_inches(m.group(2))
# # #             if ww and ll and 4<=ww<=100 and 4<=ll<=100:
# # #                 return min(ww,ll), max(ww,ll)
 
# # #         # Simple integers NxM
# # #         m2 = re.search(r'(\d{1,2})\s*[X\*]\s*(\d{1,2})', t)
# # #         if m2:
# # #             ww,ll = int(m2.group(1)), int(m2.group(2))
# # #             if 5<=ww<=80 and 5<=ll<=80 and not matches_any(t, WINDOW_PATTERNS+DOOR_PATTERNS):
# # #                 return min(ww,ll), max(ww,ll)
 
# # #     return None, None
 
# # # # ══════════════════════════════════════════════
# # # #  STRATEGY 1 — Text label detection
# # # # ══════════════════════════════════════════════
# # # ew_anchors, ew_seen = [], []
# # # iw_anchors, iw_seen = [], []
 
# # # for e in elements:
# # #     t = e["text"]
# # #     if matches_any(t, EW_PATTERNS):
# # #         if not is_dup(e["x"],e["y"],ew_seen):
# # #             ew_anchors.append(dict(e)); ew_seen.append((e["x"],e["y"]))
# # #     elif matches_any(t, IW_PATTERNS):
# # #         if not is_dup(e["x"],e["y"],iw_seen):
# # #             iw_anchors.append(dict(e)); iw_seen.append((e["x"],e["y"]))
 
# # # external_walls = {}
# # # internal_walls = {}
 
# # # def build_from_text_labels():
# # #     for i, anchor in enumerate(ew_anchors):
# # #         eid = f"ew{i+1}"
# # #         nb  = nearby(anchor, 280)
# # #         t_v = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
# # #         h_v = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
# # #         if t_v and drawing_unit == "m":
# # #             t_display = round(t_v * 39.37)
# # #             h_display = round(h_v * 3.281) if h_v else 10
# # #         else:
# # #             t_display = int(t_v) if t_v else 9
# # #             h_display = int(h_v) if h_v else 10
# # #         d = find_dir(nb) or pos_by_location(anchor["x"], anchor["y"])
# # #         external_walls[eid] = {
# # #             "id":eid, "position":d,
# # #             "length_ft":None, "thickness_in":t_display, "height_ft":h_display,
# # #             "connected":{"windows":[],"doors":[],"internal_walls":[]}
# # #         }
# # #         anchor["id"] = eid
 
# # #     for i, anchor in enumerate(iw_anchors):
# # #         iwid = f"iw{i+1}"
# # #         nb   = nearby(anchor, 280)
# # #         t_v  = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
# # #         h_v  = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
# # #         if t_v and drawing_unit == "m":
# # #             t_display = round(t_v * 39.37)
# # #             h_display = round(h_v * 3.281) if h_v else 10
# # #         else:
# # #             t_display = int(t_v) if t_v else 4
# # #             h_display = int(h_v) if h_v else 10
# # #         d    = find_dir(nb)
# # #         conn = []
# # #         for ea in sorted(ew_anchors, key=lambda a: dist(anchor,a))[:2]:
# # #             conn.append(ea["id"])
# # #             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
# # #                 external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
# # #         internal_walls[iwid] = {
# # #             "id":iwid, "position":d,
# # #             "thickness_in":t_display, "height_ft":h_display, "length_ft":None,
# # #             "connects_external":conn,
# # #             "connected":{"doors":[],"windows":[]}
# # #         }
# # #         anchor["id"] = iwid
 
# # # # ══════════════════════════════════════════════
# # # #  STRATEGY 2 — OpenCV line detection
# # # # ══════════════════════════════════════════════
# # # def build_from_opencv():
# # #     img   = raw_images[0]
# # #     gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
# # #     kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# # #     gray  = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
# # #     edges = cv2.Canny(gray,40,120,apertureSize=3)
# # #     lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,
# # #                              minLineLength=img_w//12,maxLineGap=25)
# # #     h_lines,v_lines=[],[]
# # #     if lines is not None:
# # #         for line in lines:
# # #             x1,y1,x2,y2=line[0]
# # #             angle=abs(math.degrees(math.atan2(y2-y1,x2-x1)))
# # #             length=math.hypot(x2-x1,y2-y1)
# # #             if angle<15 or angle>165: h_lines.append((min(x1,x2),max(x1,x2),(y1+y2)//2,length))
# # #             elif 75<angle<105:        v_lines.append(((x1+x2)//2,min(y1,y2),max(y1,y2),length))
 
# # #     def cluster(lines,axis,gap=35):
# # #         if not lines: return []
# # #         lines=sorted(lines,key=lambda l:l[2] if axis=='h' else l[0])
# # #         out=[lines[0]]
# # #         for ln in lines[1:]:
# # #             coord=ln[2] if axis=='h' else ln[0]; prev=out[-1][2] if axis=='h' else out[-1][0]
# # #             if abs(coord-prev)<gap: out[-1]=ln if ln[3]>out[-1][3] else out[-1]
# # #             else: out.append(ln)
# # #         return out
 
# # #     ew_c=0; iw_c=0; bm=0.12
# # #     for (x1,x2,y,l) in cluster(h_lines,'h'):
# # #         lft=round(l/(img_w/30))
# # #         if y<img_h*bm or y>img_h*(1-bm):
# # #             ew_c+=1; eid=f"ew{ew_c}"; pos="north" if y<img_h/2 else "south"
# # #             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# # #             ew_anchors.append({"x":(x1+x2)//2,"y":y,"id":eid})
# # #         else:
# # #             iw_c+=1; iwid=f"iw{iw_c}"
# # #             internal_walls[iwid]={"id":iwid,"position":"horizontal","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
# # #             iw_anchors.append({"x":(x1+x2)//2,"y":y,"id":iwid})
# # #     for (x,y1,y2,l) in cluster(v_lines,'v'):
# # #         lft=round(l/(img_h/30))
# # #         if x<img_w*bm or x>img_w*(1-bm):
# # #             ew_c+=1; eid=f"ew{ew_c}"; pos="west" if x<img_w/2 else "east"
# # #             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# # #             ew_anchors.append({"x":x,"y":(y1+y2)//2,"id":eid})
# # #         else:
# # #             iw_c+=1; iwid=f"iw{iw_c}"
# # #             internal_walls[iwid]={"id":iwid,"position":"vertical","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
# # #             iw_anchors.append({"x":x,"y":(y1+y2)//2,"id":iwid})
# # #     for iwa in iw_anchors:
# # #         iwid=iwa["id"]
# # #         for ea in sorted(ew_anchors,key=lambda a:dist(iwa,a))[:2]:
# # #             if ea["id"] not in internal_walls[iwid]["connects_external"]: internal_walls[iwid]["connects_external"].append(ea["id"])
# # #             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]: external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
# # #     return ew_c>0
 
# # # # ══════════════════════════════════════════════
# # # #  STRATEGY 3 — Auto 4 boundary walls
# # # # ══════════════════════════════════════════════
# # # def build_fallback():
# # #     for pos,eid,x,y in [("north","ew1",img_w//2,5),("east","ew2",img_w-5,img_h//2),
# # #                          ("south","ew3",img_w//2,img_h-5),("west","ew4",5,img_h//2)]:
# # #         external_walls[eid]={"id":eid,"position":pos,"length_ft":None,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# # #         ew_anchors.append({"x":x,"y":y,"id":eid})
 
# # # # ══════════════════════════════════════════════
# # # #  RUN STRATEGY CHAIN
# # # # ══════════════════════════════════════════════
# # # strategy = "auto_boundary"
# # # if ew_anchors or iw_anchors:
# # #     build_from_text_labels(); strategy="text_labels"
# # # elif raw_images and build_from_opencv():
# # #     strategy="opencv_lines"
# # # else:
# # #     build_fallback()
 
# # # # ══════════════════════════════════════════════
# # # #  DOORS — universal detection
# # # # ══════════════════════════════════════════════
# # # door_list, door_seen = [], []
# # # for e in elements:
# # #     t=e["text"]
# # #     w,h=extract_dim_float(t,DOOR_PATTERNS)
# # #     if w and h:
# # #         if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
# # #         if not is_dup(e["x"],e["y"],door_seen):
# # #             door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
# # #             door_seen.append((e["x"],e["y"]))
# # #     # Numbered door labels: D1, D(Main) — look for nearby dimensions
# # #     elif re.match(r"^D\d+$|^DOOR\d+$|^DR\d+$|^PORTE\d*$|^D\(.*\)$|^DMAIN$", t) and not is_dup(e["x"],e["y"],door_seen):
# # #         dw, dh = find_nearby_dim(e, radius=250)
# # #         if dw and dh:
# # #             if drawing_unit=="m" and dw<5: dw=round(dw*3.281,1); dh=round(dh*3.281,1)
# # #         else:
# # #             dw, dh = (4.0, 7.0) if "MAIN" in t else (3.0, 7.0)
# # #         door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":dw,"height_ft":dh})
# # #         door_seen.append((e["x"],e["y"]))
 
# # # # ══════════════════════════════════════════════
# # # #  WINDOWS — universal detection
# # # # ══════════════════════════════════════════════
# # # window_list, win_seen = [], []
# # # for e in elements:
# # #     t=e["text"]
# # #     w,h=extract_dim_float(t,WINDOW_PATTERNS)
# # #     if w and h:
# # #         if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
# # #         if not is_dup(e["x"],e["y"],win_seen):
# # #             window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
# # #             win_seen.append((e["x"],e["y"]))
# # #     # Numbered window labels: W1, W2 — look for nearby dimensions
# # #     elif re.match(r"^W\d+$|^WIN\d+$|^WINDOW\d+$|^FENETRE\d*$", t) and not is_dup(e["x"],e["y"],win_seen):
# # #         wv, hv = find_nearby_dim(e, radius=250)
# # #         if wv and hv:
# # #             if drawing_unit=="m" and wv<5: wv=round(wv*3.281,1); hv=round(hv*3.281,1)
# # #         else:
# # #             wv, hv = 4.0, 4.0
# # #         window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":wv,"height_ft":hv})
# # #         win_seen.append((e["x"],e["y"]))
 
# # # # ══════════════════════════════════════════════
# # # #  ASSIGN on_wall + BACK-FILL
# # # # ══════════════════════════════════════════════
# # # def nearest_wall(pos):
# # #     best,bd=None,999999
# # #     for a in ew_anchors+iw_anchors:
# # #         d=dist(pos,a)
# # #         if d<bd: bd,best=d,a["id"]
# # #     return best
 
# # # doors_out={}
# # # for d in door_list:
# # #     ow=nearest_wall({"x":d["x"],"y":d["y"]})
# # #     doors_out[d["id"]]={"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],"on_wall":ow}
# # #     if ow:
# # #         wl=external_walls.get(ow) or internal_walls.get(ow)
# # #         if wl and d["id"] not in wl["connected"]["doors"]: wl["connected"]["doors"].append(d["id"])
 
# # # windows_out={}
# # # for w in window_list:
# # #     ow=nearest_wall({"x":w["x"],"y":w["y"]})
# # #     windows_out[w["id"]]={"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],"on_wall":ow}
# # #     if ow:
# # #         wl=external_walls.get(ow) or internal_walls.get(ow)
# # #         if wl and w["id"] not in wl["connected"]["windows"]: wl["connected"]["windows"].append(w["id"])
 
# # # # ══════════════════════════════════════════════
# # # #  ZONES
# # # # ══════════════════════════════════════════════
# # # ZONE_PAT=re.compile(r"^ZONE\s*\d+[A-Z]?$")
# # # zones_out={}
 
# # # def nearest_room(base,r=400):
# # #     for e in sorted(elements,key=lambda e:dist(base,e)):
# # #         if dist(base,e)>r: break
# # #         for kw in ROOM_KEYWORDS:
# # #             if kw in e["text"]: return e["text"]
# # #     return "UNKNOWN"
 
# # # for e in elements:
# # #     if ZONE_PAT.match(e["text"]):
# # #         zid=re.sub(r"\s+","",e["text"])
# # #         name=nearest_room(e)
# # #         wft,lft=zone_dims(e)
# # #         area=round(wft*lft,2) if wft and lft else None
# # #         px_ft=img_w/40.0
# # #         zp=math.sqrt(area)*px_ft*0.6 if area else img_w*0.25
 
# # #         conn_ew=list(dict.fromkeys(a["id"] for a in ew_anchors if dist(e,a)<zp*1.3))
# # #         conn_iw=list(dict.fromkeys(a["id"] for a in iw_anchors if dist(e,a)<zp*1.1))
# # #         conn_d =list(dict.fromkeys(d["id"] for d in door_list   if dist(e,{"x":d["x"],"y":d["y"]})<zp))
# # #         conn_w =list(dict.fromkeys(w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp))
 
# # #         zones_out[zid]={
# # #             "id":zid,"label":name,"width_ft":wft,"length_ft":lft,"area_sqft":area,
# # #             "connected_external_walls":conn_ew,
# # #             "connected_internal_walls":conn_iw,
# # #             "total_walls_connected":len(conn_ew)+len(conn_iw),
# # #             "doors":conn_d,"windows":conn_w
# # #         }
 
# # # # ══════════════════════════════════════════════
# # # #  DEDUP SUMMARY
# # # # ══════════════════════════════════════════════
# # # all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
# # # all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])
 
# # # for zid in zones_out:
# # #     zones_out[zid]["connected_external_walls"] = list(dict.fromkeys(zones_out[zid]["connected_external_walls"]))
# # #     zones_out[zid]["connected_internal_walls"] = list(dict.fromkeys(zones_out[zid]["connected_internal_walls"]))
# # #     zones_out[zid]["doors"]   = list(dict.fromkeys(zones_out[zid]["doors"]))
# # #     zones_out[zid]["windows"] = list(dict.fromkeys(zones_out[zid]["windows"]))
# # #     zones_out[zid]["total_walls_connected"] = (
# # #         len(zones_out[zid]["connected_external_walls"]) +
# # #         len(zones_out[zid]["connected_internal_walls"])
# # #     )
 
# # # total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]), 2)
 
# # # # ══════════════════════════════════════════════
# # # #  OUTPUT
# # # # ══════════════════════════════════════════════
# # # print(json.dumps({
# # #     "external_walls": external_walls,
# # #     "internal_walls": internal_walls,
# # #     "doors":    doors_out,
# # #     "windows":  windows_out,
# # #     "zones":    zones_out,
# # #     "summary": {
# # #         "total_external_walls":    len(external_walls),
# # #         "total_internal_walls":    len(internal_walls),
# # #         "total_doors":             len(doors_out),
# # #         "total_windows":           len(windows_out),
# # #         "total_zones":             len(zones_out),
# # #         "total_area_sqft":         total_area,
# # #         "unique_doors_in_zones":   len(all_zone_doors),
# # #         "unique_windows_in_zones": len(all_zone_windows),
# # #         "drawing_unit":            drawing_unit,
# # #         "detection_strategy":      strategy
# # #     }
# # # }))

# # # # import cv2, easyocr, re, json, numpy as np, math, sys
# # # # from pdf2image import convert_from_path

# # # # MIN_CONF   = 0.28
# # # # reader     = easyocr.Reader(['en'], gpu=False)
# # # # input_path = sys.argv[1]

# # # # # ══════════════════════════════════════════════
# # # # #  UNIVERSAL LABEL PATTERNS
# # # # # ══════════════════════════════════════════════
# # # # EW_PATTERNS = [
# # # #     r"^EW[\s,=T\d\.]*$", r"^EW\d+[A-Z]?$", r"^E\.W\.?\d*$",
# # # #     r"^E\d+$", r"^EXT[\s\-]?WALL\d*$", r"^EXTERNAL[\s\-]?WALL\d*$",
# # # #     r"^OUTER[\s\-]?WALL\d*$", r"^BOUNDARY[\s\-]?WALL\d*$",
# # # #     r"^MUR[\s\-]?EXT\d*$", r"^MUR[\s\-]?EXTERIEUR\d*$",
# # # # ]
# # # # IW_PATTERNS = [
# # # #     r"^IW[\s,=T\d\.]*$", r"^IW\d+[A-Z]?$", r"^I\.W\.?\d*$",
# # # #     r"^IW[\s\(].*$",
# # # #     r"^I\d+$", r"^INT[\s\-]?WALL\d*$", r"^INTERNAL[\s\-]?WALL\d*$",
# # # #     r"^INNER[\s\-]?WALL\d*$", r"^PARTITION[\s\-]?\d*$",
# # # #     r"^DIVIDER[\s\-]?\d*$", r"^PW\d*$",
# # # #     r"^MUR[\s\-]?INT\d*$", r"^CLOISON\d*$",
# # # # ]
# # # # DOOR_PATTERNS = [
# # # #     r"^D[=\-\s]?(\d+\.?\d*)[xX×X\xd7](\d+\.?\d*)$",
# # # #     r"^DOOR[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # # #     r"^DR[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # # #     r"^D[\s\(].*?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*).*$",
# # # #     r"^D\(MAIN\).*?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*).*$",
# # # #     r"^D\d+$", r"^DOOR\d*$", r"^DR\d*$", r"^PORTE\d*$",
# # # #     r"^D\(.*\)$", r"^DMAIN$", r"^D\(MAIN\)$",
# # # # ]
# # # # WINDOW_PATTERNS = [
# # # #     r"^W[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # # #     r"^W[=\-\s]?(\d+['']\d*[\"']?)\s*[xX×\xd7]\s*(\d+['']\d*[\"']?)$",
# # # #     r"^W[=\-]?(\d+)'?(\d*)[\"']?[xX×\xd7](\d+)[''\"']?$",
# # # #     r"^WIN[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # # #     r"^WINDOW[=\-\s]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # # #     r"^W\d+[\s\n]?(\d+\.?\d*)[xX×\xd7](\d+\.?\d*)$",
# # # #     r"^W\d+$", r"^WIN\d*$", r"^WINDOW\d*$", r"^FENETRE\d*$",
# # # # ]
# # # # ROOM_KEYWORDS = [
# # # #     "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL",
# # # #     "BALCONY","STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM",
# # # #     "C.BEDROOM","MASTER","DRAWING","UTILITY","GARAGE","PASSAGE",
# # # #     "WASH","BATH","FAMILY","PANTRY","LAUNDRY","TERRACE","VERANDAH",
# # # #     "ENTRANCE","FOYER","MBEDROOM","CBEDROOM","BEDROOM 2","BEDROOM 3",
# # # #     "LIVING ROOM","DINING ROOM","STORE ROOM","PRAYER","POWDER",
# # # #     "CHAMBRE","SALON","CUISINE","SALLE","BAIN","TOILETTE","WC",
# # # #     "COULOIR","ENTREE","BUREAU","DRESSING","BUANDERIE","SEJOUR",
# # # #     "TERRASSE","CELLIER","PLACARD","DEGAGEMENT","SALLE DE BAIN",
# # # #     "SALLE A MANGER","SALON PRIVE","CHAMBRE ENFANT","DOUCHE","MAGASIN",
# # # # ]

# # # # DIM_SEP = r"[xX×\xd7\*]"   # universal dimension separator

# # # # # ══════════════════════════════════════════════
# # # # #  LOAD
# # # # # ══════════════════════════════════════════════
# # # # if input_path.lower().endswith(".pdf"):
# # # #     pages = convert_from_path(input_path, dpi=300)
# # # # else:
# # # #     img   = cv2.imread(input_path)
# # # #     pages = [img] if img is not None else []

# # # # elements   = []
# # # # raw_images = []

# # # # def preprocess(img):
# # # #     if not isinstance(img, np.ndarray):
# # # #         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# # # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
# # # #     gray = cv2.bilateralFilter(gray, 9, 75, 75)
# # # #     _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # # #     return th

# # # # for page in pages:
# # # #     image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
# # # #     raw_images.append(image)
# # # #     raw_results = reader.readtext(preprocess(image))

# # # #     # Merge split detections on same line
# # # #     merged   = []
# # # #     used     = set()
# # # #     raw_sorted = sorted(raw_results, key=lambda r: (round(r[0][0][1]/20), r[0][0][0]))
# # # #     for i, (bbox, text, prob) in enumerate(raw_sorted):
# # # #         if i in used or prob < MIN_CONF: continue
# # # #         x = int(sum([p[0] for p in bbox]) / 4)
# # # #         y = int(sum([p[1] for p in bbox]) / 4)
# # # #         if i+1 < len(raw_sorted) and i+1 not in used:
# # # #             bbox2, text2, prob2 = raw_sorted[i+1]
# # # #             x2 = int(sum([p[0] for p in bbox2]) / 4)
# # # #             y2 = int(sum([p[1] for p in bbox2]) / 4)
# # # #             if abs(y2-y) < 25 and abs(x2-x) < 130 and prob2 >= MIN_CONF:
# # # #                 merged_text = (text + text2).upper().strip()
# # # #                 if re.search(r'[WD].*\d+.*[Xx×\xd7].*\d+', merged_text):
# # # #                     merged.append({"text": merged_text, "x": (x+x2)//2, "y": (y+y2)//2})
# # # #                     used.add(i); used.add(i+1)
# # # #                     continue
# # # #         merged.append({"text": text.upper().strip(), "x": x, "y": y})
# # # #         used.add(i)
# # # #     elements.extend(merged)

# # # # # ══════════════════════════════════════════════
# # # # #  CLEAN — normalize ALL OCR noise
# # # # # ══════════════════════════════════════════════
# # # # def clean(t):
# # # #     t = t.upper().strip()
# # # #     # Normalize Unicode × to x
# # # #     t = t.replace('\xd7','X').replace('×','X').replace('✕','X')
# # # #     t = re.sub(r"\bEW,\b","EW",t)
# # # #     t = t.replace("1W","IW").replace("lW","IW").replace("lw","IW").replace("|W","IW")
# # # #     # Dash→equals
# # # #     t = re.sub(r'\b([TH])-(\d)', r'\1=\2', t)
# # # #     t = re.sub(r'\b([DW])-(\d)', r'\1=\2', t)
# # # #     # Fix OCR garbling
# # # #     t = re.sub(r'\bTZ(\d)', r'T=\1', t)
# # # #     t = re.sub(r'H-L(\d)', r'H=1\1', t)
# # # #     t = t.replace("T-G","T=9").replace("T=G","T=9")
# # # #     t = re.sub(r'\bW-(\d)', r'W=\1', t)
# # # #     t = re.sub(r'\bD-(\d)', r'D=\1', t)
# # # #     # Fix D(MAIN), D (MAIN) → DMAIN for easier matching
# # # #     t = re.sub(r'D\s*\(\s*MAIN\s*\)', 'D(MAIN)', t)
# # # #     # Fix letter/digit confusion
# # # #     t = re.sub(r'\b([DW]=)([A-Z])X', lambda m: m.group(1)+'4X', t)
# # # #     t = re.sub(r'X([A-Z])\b', 'X4', t)
# # # #     t = t.replace("EW.","EW").replace("IW.","IW")
# # # #     t = re.sub(r"\s+"," ",t)
# # # #     return t.strip()

# # # # for e in elements: e["text"] = clean(e["text"])

# # # # img_h = raw_images[0].shape[0] if raw_images else 1000
# # # # img_w = raw_images[0].shape[1] if raw_images else 1000

# # # # # Remove legend/footer labels (bottom 20% = legend area)
# # # # legend_y = img_h * 0.80
# # # # LEGEND_KEYWORDS = [
# # # #     "= EXTERNAL WALL","= INTERNAL WALL","EXTERNAL WALL","EW =","IW =",
# # # #     "EW:","ROOM TINTS","BEDROOMS","BATHROOMS","KITCHEN/DINING",
# # # #     "W = WINDOW","D = DOOR","T = THICKNESS","H = HEIGHT",
# # # #     "BEDROOMS / STUDY","BATHROOMS / LIVING","KITCHEN / DINING",
# # # # ]
# # # # elements = [e for e in elements if not (
# # # #     e["y"] > legend_y and
# # # #     any(kw in e["text"] for kw in LEGEND_KEYWORDS)
# # # # )]

# # # # # ══════════════════════════════════════════════
# # # # #  HELPERS
# # # # # ══════════════════════════════════════════════
# # # # def dist(a,b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
# # # # def is_dup(x,y,pos,th=50): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)
# # # # def nearby(base,r=220): return [e for e in elements if dist(base,e)<r]

# # # # def matches_any(text, patterns):
# # # #     return any(re.match(p, text) for p in patterns)

# # # # def extract_dim_float(text, patterns):
# # # #     """Extract WxH dimensions — handles x, X, × all normalized to X after clean()."""
# # # #     for p in patterns:
# # # #         m = re.match(p, text)
# # # #         if m and m.lastindex and m.lastindex >= 2:
# # # #             try: return float(m.group(1)), float(m.group(2))
# # # #             except: pass
# # # #     return None, None

# # # # def find_nearby_dim(base, radius=250):
# # # #     """
# # # #     Search nearby text for dimension pattern like:
# # # #     1.35 X 1.20  or  3.50 X 4.25  or  12'0"X15'0"
# # # #     Returns (w, h) as floats or (None, None)
# # # #     """
# # # #     for e in sorted([e for e in elements if dist(base,e)<radius], key=lambda e: dist(base,e)):
# # # #         t = e["text"]
# # # #         # Metric decimal: 1.35 X 1.20
# # # #         m = re.search(r'(\d+\.\d+)\s*X\s*(\d+\.\d+)', t)
# # # #         if m:
# # # #             return float(m.group(1)), float(m.group(2))
# # # #         # Imperial feet: 12'0"X15'0"
# # # #         m = re.search(r"(\d+'\d*[\"']?)\s*X\s*(\d+'\d*[\"']?)", t)
# # # #         if m:
# # # #             ww = parse_feet_inches(m.group(1))
# # # #             ll = parse_feet_inches(m.group(2))
# # # #             if ww and ll: return float(ww), float(ll)
# # # #         # Simple integers: 3X7
# # # #         m = re.search(r'\b(\d+\.?\d*)\s*X\s*(\d+\.?\d*)\b', t)
# # # #         if m:
# # # #             ww, ll = float(m.group(1)), float(m.group(2))
# # # #             if 0.3 <= ww <= 80 and 0.3 <= ll <= 80:
# # # #                 return ww, ll
# # # #     return None, None

# # # # def find_val_float(elems, pat):
# # # #     for e in elems:
# # # #         txt = re.sub(r'\b([TH])-(\d)', r'\1=\2', e["text"])
# # # #         m = re.search(pat, txt)
# # # #         if m:
# # # #             try: return float(m.group(1))
# # # #             except: pass
# # # #     return None

# # # # def find_dir(elems):
# # # #     dirs = ["NORTH","SOUTH","EAST","WEST","MIDDLE","CENTER","TOP","BOTTOM","LEFT","RIGHT"]
# # # #     for e in elems:
# # # #         for d in dirs:
# # # #             if d in e["text"]: return d.lower()
# # # #     return None

# # # # def pos_by_location(x, y):
# # # #     m = 0.15
# # # #     if   y < img_h*m:      return "north"
# # # #     elif y > img_h*(1-m):  return "south"
# # # #     elif x < img_w*m:      return "west"
# # # #     elif x > img_w*(1-m):  return "east"
# # # #     return "inner"

# # # # def parse_feet_inches(text):
# # # #     if not text: return None
# # # #     text = str(text).replace(" ","")
# # # #     fm   = re.search(r"(\d+)'", text)
# # # #     if not fm: return None
# # # #     feet   = int(fm.group(1))
# # # #     im     = re.search(r"'(\d+)", text)
# # # #     fracm  = re.search(r"(\d+)/(\d+)", text)
# # # #     inches = int(im.group(1)) if im else 0
# # # #     frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
# # # #     if inches > 12: inches = int(str(inches)[0])
# # # #     return math.ceil(feet+(inches+frac)/12)

# # # # def to_ft(val, unit):
# # # #     """Convert value to feet if needed."""
# # # #     if val is None: return None
# # # #     if unit == "m": return round(val * 3.281, 1)
# # # #     return round(val, 1)

# # # # # ══════════════════════════════════════════════
# # # # #  DETECT DRAWING UNIT
# # # # # ══════════════════════════════════════════════
# # # # def detect_unit():
# # # #     metric_hits = 0; imperial_hits = 0
# # # #     for e in elements:
# # # #         t = e["text"]
# # # #         if re.search(r"\d+\.\d{2}", t): metric_hits += 1
# # # #         if re.search(r"\b\d+M\b|\d+\.?\d*\s*M\b", t): metric_hits += 1
# # # #         if re.search(r"\d+'", t): imperial_hits += 1
# # # #         if re.search(r"\d+X\d+", t) and not re.search(r"\d+\.\d+", t): imperial_hits += 1
# # # #     return "m" if metric_hits > imperial_hits else "ft"

# # # # drawing_unit = detect_unit()

# # # # # ══════════════════════════════════════════════
# # # # #  ZONE DIMENSION PARSER — all formats
# # # # # ══════════════════════════════════════════════
# # # # def zone_dims(base, r=500):
# # # #     for e in elements:
# # # #         if dist(base,e) > r: continue
# # # #         t = e["text"]

# # # #         # Metric decimal: 3.50 X 4.25
# # # #         m = re.search(r'(\d+\.\d+)\s*X\s*(\d+\.\d+)', t)
# # # #         if m:
# # # #             ww = to_ft(float(m.group(1)), drawing_unit)
# # # #             ll = to_ft(float(m.group(2)), drawing_unit)
# # # #             if ww and ll and 4<=ww<=150 and 4<=ll<=150:
# # # #                 return round(min(ww,ll),1), round(max(ww,ll),1)

# # # #         # Imperial: 12'0"X15'0"
# # # #         m = re.search(r"(\d+['']\d+[\"']?)\s*X\s*(\d+['']\d+[\"']?)", t)
# # # #         if m:
# # # #             ww = parse_feet_inches(m.group(1))
# # # #             ll = parse_feet_inches(m.group(2))
# # # #             if ww and ll and 4<=ww<=100 and 4<=ll<=100:
# # # #                 return min(ww,ll), max(ww,ll)

# # # #         # Imperial no separator: 12'0"15'0"
# # # #         m = re.search(r"(\d+['']\d+)[\"']\s*(\d+['']\d+)", t)
# # # #         if m:
# # # #             ww = parse_feet_inches(m.group(1))
# # # #             ll = parse_feet_inches(m.group(2))
# # # #             if ww and ll and 4<=ww<=100 and 4<=ll<=100:
# # # #                 return min(ww,ll), max(ww,ll)

# # # #         # Simple integers NxM
# # # #         m2 = re.search(r'(\d{1,2})\s*X\s*(\d{1,2})', t)
# # # #         if m2:
# # # #             ww,ll = int(m2.group(1)), int(m2.group(2))
# # # #             if 5<=ww<=80 and 5<=ll<=80 and not matches_any(t, WINDOW_PATTERNS+DOOR_PATTERNS):
# # # #                 return min(ww,ll), max(ww,ll)

# # # #     return None, None

# # # # # ══════════════════════════════════════════════
# # # # #  STRATEGY 1 — Text label detection
# # # # # ══════════════════════════════════════════════
# # # # ew_anchors, ew_seen = [], []
# # # # iw_anchors, iw_seen = [], []

# # # # for e in elements:
# # # #     t = e["text"]
# # # #     if matches_any(t, EW_PATTERNS):
# # # #         if not is_dup(e["x"],e["y"],ew_seen):
# # # #             ew_anchors.append(dict(e)); ew_seen.append((e["x"],e["y"]))
# # # #     elif matches_any(t, IW_PATTERNS):
# # # #         if not is_dup(e["x"],e["y"],iw_seen):
# # # #             iw_anchors.append(dict(e)); iw_seen.append((e["x"],e["y"]))

# # # # external_walls = {}
# # # # internal_walls = {}

# # # # def build_from_text_labels():
# # # #     for i, anchor in enumerate(ew_anchors):
# # # #         eid = f"ew{i+1}"
# # # #         nb  = nearby(anchor, 280)
# # # #         t_v = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
# # # #         h_v = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
# # # #         if t_v and drawing_unit == "m":
# # # #             t_display = round(t_v * 39.37)
# # # #             h_display = round(h_v * 3.281) if h_v else 10
# # # #         else:
# # # #             t_display = int(t_v) if t_v else 9
# # # #             h_display = int(h_v) if h_v else 10
# # # #         d = find_dir(nb) or pos_by_location(anchor["x"], anchor["y"])
# # # #         external_walls[eid] = {
# # # #             "id":eid, "position":d,
# # # #             "length_ft":None, "thickness_in":t_display, "height_ft":h_display,
# # # #             "connected":{"windows":[],"doors":[],"internal_walls":[]}
# # # #         }
# # # #         anchor["id"] = eid

# # # #     for i, anchor in enumerate(iw_anchors):
# # # #         iwid = f"iw{i+1}"
# # # #         nb   = nearby(anchor, 280)
# # # #         t_v  = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
# # # #         h_v  = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
# # # #         if t_v and drawing_unit == "m":
# # # #             t_display = round(t_v * 39.37)
# # # #             h_display = round(h_v * 3.281) if h_v else 10
# # # #         else:
# # # #             t_display = int(t_v) if t_v else 4
# # # #             h_display = int(h_v) if h_v else 10
# # # #         d    = find_dir(nb)
# # # #         conn = []
# # # #         for ea in sorted(ew_anchors, key=lambda a: dist(anchor,a))[:2]:
# # # #             conn.append(ea["id"])
# # # #             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
# # # #                 external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
# # # #         internal_walls[iwid] = {
# # # #             "id":iwid, "position":d,
# # # #             "thickness_in":t_display, "height_ft":h_display, "length_ft":None,
# # # #             "connects_external":conn,
# # # #             "connected":{"doors":[],"windows":[]}
# # # #         }
# # # #         anchor["id"] = iwid

# # # # # ══════════════════════════════════════════════
# # # # #  STRATEGY 2 — OpenCV line detection
# # # # # ══════════════════════════════════════════════
# # # # def build_from_opencv():
# # # #     img   = raw_images[0]
# # # #     gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
# # # #     kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# # # #     gray  = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
# # # #     edges = cv2.Canny(gray,40,120,apertureSize=3)
# # # #     lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,
# # # #                              minLineLength=img_w//12,maxLineGap=25)
# # # #     h_lines,v_lines=[],[]
# # # #     if lines is not None:
# # # #         for line in lines:
# # # #             x1,y1,x2,y2=line[0]
# # # #             angle=abs(math.degrees(math.atan2(y2-y1,x2-x1)))
# # # #             length=math.hypot(x2-x1,y2-y1)
# # # #             if angle<15 or angle>165: h_lines.append((min(x1,x2),max(x1,x2),(y1+y2)//2,length))
# # # #             elif 75<angle<105:        v_lines.append(((x1+x2)//2,min(y1,y2),max(y1,y2),length))

# # # #     def cluster(lines,axis,gap=35):
# # # #         if not lines: return []
# # # #         lines=sorted(lines,key=lambda l:l[2] if axis=='h' else l[0])
# # # #         out=[lines[0]]
# # # #         for ln in lines[1:]:
# # # #             coord=ln[2] if axis=='h' else ln[0]; prev=out[-1][2] if axis=='h' else out[-1][0]
# # # #             if abs(coord-prev)<gap: out[-1]=ln if ln[3]>out[-1][3] else out[-1]
# # # #             else: out.append(ln)
# # # #         return out

# # # #     ew_c=0; iw_c=0; bm=0.12
# # # #     for (x1,x2,y,l) in cluster(h_lines,'h'):
# # # #         lft=round(l/(img_w/30))
# # # #         if y<img_h*bm or y>img_h*(1-bm):
# # # #             ew_c+=1; eid=f"ew{ew_c}"; pos="north" if y<img_h/2 else "south"
# # # #             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# # # #             ew_anchors.append({"x":(x1+x2)//2,"y":y,"id":eid})
# # # #         else:
# # # #             iw_c+=1; iwid=f"iw{iw_c}"
# # # #             internal_walls[iwid]={"id":iwid,"position":"horizontal","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
# # # #             iw_anchors.append({"x":(x1+x2)//2,"y":y,"id":iwid})
# # # #     for (x,y1,y2,l) in cluster(v_lines,'v'):
# # # #         lft=round(l/(img_h/30))
# # # #         if x<img_w*bm or x>img_w*(1-bm):
# # # #             ew_c+=1; eid=f"ew{ew_c}"; pos="west" if x<img_w/2 else "east"
# # # #             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# # # #             ew_anchors.append({"x":x,"y":(y1+y2)//2,"id":eid})
# # # #         else:
# # # #             iw_c+=1; iwid=f"iw{iw_c}"
# # # #             internal_walls[iwid]={"id":iwid,"position":"vertical","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
# # # #             iw_anchors.append({"x":x,"y":(y1+y2)//2,"id":iwid})
# # # #     for iwa in iw_anchors:
# # # #         iwid=iwa["id"]
# # # #         for ea in sorted(ew_anchors,key=lambda a:dist(iwa,a))[:2]:
# # # #             if ea["id"] not in internal_walls[iwid]["connects_external"]: internal_walls[iwid]["connects_external"].append(ea["id"])
# # # #             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]: external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
# # # #     return ew_c>0

# # # # # ══════════════════════════════════════════════
# # # # #  STRATEGY 3 — Auto 4 boundary walls
# # # # # ══════════════════════════════════════════════
# # # # def build_fallback():
# # # #     for pos,eid,x,y in [("north","ew1",img_w//2,5),("east","ew2",img_w-5,img_h//2),
# # # #                          ("south","ew3",img_w//2,img_h-5),("west","ew4",5,img_h//2)]:
# # # #         external_walls[eid]={"id":eid,"position":pos,"length_ft":None,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
# # # #         ew_anchors.append({"x":x,"y":y,"id":eid})

# # # # # ══════════════════════════════════════════════
# # # # #  RUN STRATEGY CHAIN
# # # # # ══════════════════════════════════════════════
# # # # strategy = "auto_boundary"
# # # # if ew_anchors or iw_anchors:
# # # #     build_from_text_labels(); strategy="text_labels"
# # # # elif raw_images and build_from_opencv():
# # # #     strategy="opencv_lines"
# # # # else:
# # # #     build_fallback()

# # # # # ══════════════════════════════════════════════
# # # # #  DOORS — universal detection
# # # # # ══════════════════════════════════════════════
# # # # door_list, door_seen = [], []
# # # # for e in elements:
# # # #     t=e["text"]
# # # #     w,h=extract_dim_float(t,DOOR_PATTERNS)
# # # #     if w and h:
# # # #         if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
# # # #         if not is_dup(e["x"],e["y"],door_seen):
# # # #             door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
# # # #             door_seen.append((e["x"],e["y"]))
# # # #     # Numbered door labels: D1, D(Main) — look for nearby dimensions
# # # #     elif re.match(r"^D\d+$|^DOOR\d+$|^DR\d+$|^PORTE\d*$|^D\(.*\)$|^DMAIN$", t) and not is_dup(e["x"],e["y"],door_seen):
# # # #         dw, dh = find_nearby_dim(e, radius=250)
# # # #         if dw and dh:
# # # #             if drawing_unit=="m" and dw<5: dw=round(dw*3.281,1); dh=round(dh*3.281,1)
# # # #         else:
# # # #             dw, dh = (4.0, 7.0) if "MAIN" in t else (3.0, 7.0)
# # # #         door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":dw,"height_ft":dh})
# # # #         door_seen.append((e["x"],e["y"]))

# # # # # ══════════════════════════════════════════════
# # # # #  WINDOWS — universal detection
# # # # # ══════════════════════════════════════════════
# # # # window_list, win_seen = [], []
# # # # for e in elements:
# # # #     t=e["text"]
# # # #     w,h=extract_dim_float(t,WINDOW_PATTERNS)
# # # #     if w and h:
# # # #         if drawing_unit=="m" and w<5: w=round(w*3.281,1); h=round(h*3.281,1)
# # # #         if not is_dup(e["x"],e["y"],win_seen):
# # # #             window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
# # # #             win_seen.append((e["x"],e["y"]))
# # # #     # Numbered window labels: W1, W2 — look for nearby dimensions
# # # #     elif re.match(r"^W\d+$|^WIN\d+$|^WINDOW\d+$|^FENETRE\d*$", t) and not is_dup(e["x"],e["y"],win_seen):
# # # #         wv, hv = find_nearby_dim(e, radius=250)
# # # #         if wv and hv:
# # # #             if drawing_unit=="m" and wv<5: wv=round(wv*3.281,1); hv=round(hv*3.281,1)
# # # #         else:
# # # #             wv, hv = 4.0, 4.0
# # # #         window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":wv,"height_ft":hv})
# # # #         win_seen.append((e["x"],e["y"]))

# # # # # ══════════════════════════════════════════════
# # # # #  ASSIGN on_wall + BACK-FILL
# # # # # ══════════════════════════════════════════════
# # # # def nearest_wall(pos):
# # # #     best,bd=None,999999
# # # #     for a in ew_anchors+iw_anchors:
# # # #         d=dist(pos,a)
# # # #         if d<bd: bd,best=d,a["id"]
# # # #     return best

# # # # doors_out={}
# # # # for d in door_list:
# # # #     ow=nearest_wall({"x":d["x"],"y":d["y"]})
# # # #     doors_out[d["id"]]={"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],"on_wall":ow}
# # # #     if ow:
# # # #         wl=external_walls.get(ow) or internal_walls.get(ow)
# # # #         if wl and d["id"] not in wl["connected"]["doors"]: wl["connected"]["doors"].append(d["id"])

# # # # windows_out={}
# # # # for w in window_list:
# # # #     ow=nearest_wall({"x":w["x"],"y":w["y"]})
# # # #     windows_out[w["id"]]={"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],"on_wall":ow}
# # # #     if ow:
# # # #         wl=external_walls.get(ow) or internal_walls.get(ow)
# # # #         if wl and w["id"] not in wl["connected"]["windows"]: wl["connected"]["windows"].append(w["id"])

# # # # # ══════════════════════════════════════════════
# # # # #  ZONES
# # # # # ══════════════════════════════════════════════
# # # # ZONE_PAT=re.compile(r"^ZONE\s*\d+[A-Z]?$")
# # # # zones_out={}

# # # # def nearest_room(base,r=400):
# # # #     for e in sorted(elements,key=lambda e:dist(base,e)):
# # # #         if dist(base,e)>r: break
# # # #         for kw in ROOM_KEYWORDS:
# # # #             if kw in e["text"]: return e["text"]
# # # #     return "UNKNOWN"

# # # # for e in elements:
# # # #     if ZONE_PAT.match(e["text"]):
# # # #         zid=re.sub(r"\s+","",e["text"])
# # # #         name=nearest_room(e)
# # # #         wft,lft=zone_dims(e)
# # # #         area=round(wft*lft,2) if wft and lft else None
# # # #         px_ft=img_w/40.0
# # # #         zp=math.sqrt(area)*px_ft*0.6 if area else img_w*0.25

# # # #         conn_ew=list(dict.fromkeys(a["id"] for a in ew_anchors if dist(e,a)<zp*1.3))
# # # #         conn_iw=list(dict.fromkeys(a["id"] for a in iw_anchors if dist(e,a)<zp*1.1))
# # # #         conn_d =list(dict.fromkeys(d["id"] for d in door_list   if dist(e,{"x":d["x"],"y":d["y"]})<zp))
# # # #         conn_w =list(dict.fromkeys(w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp))

# # # #         zones_out[zid]={
# # # #             "id":zid,"label":name,"width_ft":wft,"length_ft":lft,"area_sqft":area,
# # # #             "connected_external_walls":conn_ew,
# # # #             "connected_internal_walls":conn_iw,
# # # #             "total_walls_connected":len(conn_ew)+len(conn_iw),
# # # #             "doors":conn_d,"windows":conn_w
# # # #         }

# # # # # ══════════════════════════════════════════════
# # # # #  DEDUP SUMMARY
# # # # # ══════════════════════════════════════════════
# # # # all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
# # # # all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])

# # # # for zid in zones_out:
# # # #     zones_out[zid]["connected_external_walls"] = list(dict.fromkeys(zones_out[zid]["connected_external_walls"]))
# # # #     zones_out[zid]["connected_internal_walls"] = list(dict.fromkeys(zones_out[zid]["connected_internal_walls"]))
# # # #     zones_out[zid]["doors"]   = list(dict.fromkeys(zones_out[zid]["doors"]))
# # # #     zones_out[zid]["windows"] = list(dict.fromkeys(zones_out[zid]["windows"]))
# # # #     zones_out[zid]["total_walls_connected"] = (
# # # #         len(zones_out[zid]["connected_external_walls"]) +
# # # #         len(zones_out[zid]["connected_internal_walls"])
# # # #     )

# # # # total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]), 2)

# # # # # ══════════════════════════════════════════════
# # # # #  OUTPUT
# # # # # ══════════════════════════════════════════════
# # # # print(json.dumps({
# # # #     "external_walls": external_walls,
# # # #     "internal_walls": internal_walls,
# # # #     "doors":    doors_out,
# # # #     "windows":  windows_out,
# # # #     "zones":    zones_out,
# # # #     "summary": {
# # # #         "total_external_walls":    len(external_walls),
# # # #         "total_internal_walls":    len(internal_walls),
# # # #         "total_doors":             len(doors_out),
# # # #         "total_windows":           len(windows_out),
# # # #         "total_zones":             len(zones_out),
# # # #         "total_area_sqft":         total_area,
# # # #         "unique_doors_in_zones":   len(all_zone_doors),
# # # #         "unique_windows_in_zones": len(all_zone_windows),
# # # #         "drawing_unit":            drawing_unit,
# # # #         "detection_strategy":      strategy
# # # #     }
# # # # }))
