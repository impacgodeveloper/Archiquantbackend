import cv2, easyocr, re, json, numpy as np, math, sys
from pdf2image import convert_from_path
 
MIN_CONF   = 0.28
reader     = easyocr.Reader(['en'], gpu=False)
input_path = sys.argv[1]
 
# ══════════════════════════════════════════════
#  UNIVERSAL LABEL PATTERNS
# ══════════════════════════════════════════════
EW_PATTERNS = [
    r"^EW[\s,=T\d\.]*$", r"^EW\d+[A-Z]?$", r"^E\.W\.?\d*$",
    r"^E\d+$", r"^EXT[\s\-]?WALL\d*$", r"^EXTERNAL[\s\-]?WALL\d*$",
    r"^OUTER[\s\-]?WALL\d*$", r"^BOUNDARY[\s\-]?WALL\d*$",
    r"^MUR[\s\-]?EXT\d*$", r"^MUR[\s\-]?EXTERIEUR\d*$",   # French
]
IW_PATTERNS = [
    r"^IW[\s,=T\d\.]*$", r"^IW\d+[A-Z]?$", r"^I\.W\.?\d*$",
    r"^IW[\s\(].*$",                                         # IW (bath)
    r"^I\d+$", r"^INT[\s\-]?WALL\d*$", r"^INTERNAL[\s\-]?WALL\d*$",
    r"^INNER[\s\-]?WALL\d*$", r"^PARTITION[\s\-]?\d*$",
    r"^DIVIDER[\s\-]?\d*$", r"^PW\d*$",
    r"^MUR[\s\-]?INT\d*$", r"^CLOISON\d*$",                 # French
]
DOOR_PATTERNS = [
    r"^D[=\-\s]?(\d+\.?\d*)[xX×](\d+\.?\d*)$",
    r"^DOOR[=\-\s]?(\d+\.?\d*)[xX×](\d+\.?\d*)$",
    r"^DR[=\-\s]?(\d+\.?\d*)[xX×](\d+\.?\d*)$",
    r"^D[\s\(].*(\d+\.?\d*)[xX×](\d+\.?\d*).*$",            # D (Main) 1.20x2.10
    r"^D\d+$", r"^DOOR\d*$", r"^DR\d*$", r"^PORTE\d*$",
]
WINDOW_PATTERNS = [
    r"^W[=\-\s]?(\d+\.?\d*)[xX×](\d+\.?\d*)$",
    r"^WIN[=\-\s]?(\d+\.?\d*)[xX×](\d+\.?\d*)$",
    r"^WINDOW[=\-\s]?(\d+\.?\d*)[xX×](\d+\.?\d*)$",
    r"^W\d+[\s\n]?(\d+\.?\d*)[xX×](\d+\.?\d*)$",            # W1 1.35x1.20
    r"^W\d+$", r"^WIN\d*$", r"^WINDOW\d*$", r"^FENETRE\d*$",
]
 
# ══════════════════════════════════════════════
#  ROOM KEYWORDS — English + French + others
# ══════════════════════════════════════════════
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
    "TERRASSE","GARAGE","CELLIER","PLACARD","DEGAGEMENT","HALL",
    "SALLE DE BAIN","SALLE A MANGER","SALON PRIVE","CHAMBRE ENFANT",
    # Arabic transliteration
    "DOUCHE","MAGASIN","MAGASIN",
]
 
# ══════════════════════════════════════════════
#  LOAD
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
    for (bbox, text, prob) in reader.readtext(preprocess(image)):
        if prob < MIN_CONF: continue
        text = text.upper().strip()
        x = int(sum([p[0] for p in bbox]) / 4)
        y = int(sum([p[1] for p in bbox]) / 4)
        elements.append({"text": text, "x": x, "y": y})
 
# ══════════════════════════════════════════════
#  CLEAN — normalize ALL OCR noise
# ══════════════════════════════════════════════
def clean(t):
    t = t.upper().strip()
    t = re.sub(r"\bEW,\b","EW",t)
    t = t.replace("1W","IW").replace("lW","IW").replace("lw","IW").replace("|W","IW")
    # Dash→equals for dimension markers
    t = re.sub(r'\b([TH])-(\d)', r'\1=\2', t)
    t = re.sub(r'\b([DW])-(\d)', r'\1=\2', t)
    # Fix OCR letter/digit confusion in dimensions
    t = re.sub(r'\b([DW]=)([A-Z])X', lambda m: m.group(1)+'4X', t)
    t = re.sub(r'X([A-Z])\b', 'X4', t)
    t = t.replace("EW.","EW").replace("IW.","IW")
    t = re.sub(r"\s+"," ",t)
    return t.strip()
 
for e in elements: e["text"] = clean(e["text"])
 
img_h = raw_images[0].shape[0] if raw_images else 1000
img_w = raw_images[0].shape[1] if raw_images else 1000
 
# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def dist(a,b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
def is_dup(x,y,pos,th=50): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)
def nearby(base,r=220): return [e for e in elements if dist(base,e)<r]
 
def matches_any(text, patterns):
    return any(re.match(p, text) for p in patterns)
 
def extract_dim_float(text, patterns):
    """Extract WxH as floats — handles both 3x7 and 1.20x2.10."""
    for p in patterns:
        m = re.match(p, text)
        if m and m.lastindex and m.lastindex >= 2:
            try:
                return float(m.group(1)), float(m.group(2))
            except: pass
    return None, None
 
def find_val_float(elems, pat):
    """Find numeric value — accepts integers AND decimals."""
    for e in elems:
        txt = re.sub(r'\b([TH])-(\d)', r'\1=\2', e["text"])
        m = re.search(pat, txt)
        if m:
            try: return float(m.group(1))
            except: pass
    return None
 
def find_dir(elems):
    dirs = ["NORTH","SOUTH","EAST","WEST","MIDDLE","CENTER","TOP","BOTTOM","LEFT","RIGHT"]
    for e in elems:
        for d in dirs:
            if d in e["text"]: return d.lower()
    return None
 
def pos_by_location(x, y):
    m = 0.15
    if   y < img_h*m:      return "north"
    elif y > img_h*(1-m):  return "south"
    elif x < img_w*m:      return "west"
    elif x > img_w*(1-m):  return "east"
    return "inner"
 
def is_metric(val):
    """Guess if a value is in meters (< 15) vs feet (>= 4)."""
    return val is not None and val < 15
 
def to_feet(val, unit):
    """Convert to feet if needed."""
    if val is None: return None
    if unit == "m": return round(val * 3.281, 1)
    return val
 
# ══════════════════════════════════════════════
#  DIMENSION UNIT DETECTION
#  Scan drawing to decide if it's metric or imperial
# ══════════════════════════════════════════════
def detect_unit():
    """Return 'm' for metric or 'ft' for imperial."""
    metric_hits   = 0
    imperial_hits = 0
    for e in elements:
        t = e["text"]
        # Metric indicators: decimals like 3.50, 0.23, or explicit 'm'
        if re.search(r"\d+\.\d{2}", t): metric_hits += 1
        if re.search(r"\d+M$|\d+\.?\d*\s*M\b", t): metric_hits += 1
        # Imperial indicators: feet marks
        if re.search(r"\d+'", t): imperial_hits += 1
        if re.search(r"\d+[xX]\d+", t) and not re.search(r"\d+\.\d+", t):
            imperial_hits += 1
    return "m" if metric_hits > imperial_hits else "ft"
 
drawing_unit = detect_unit()
 
def parse_feet_inches(text):
    """Parse imperial: 12'0" → 12, 6'6" → 7."""
    if not text: return None
    text = str(text).replace(" ","")
    fm   = re.search(r"(\d+)'", text)
    if not fm: return None
    feet   = int(fm.group(1))
    im     = re.search(r"'(\d+)", text)
    fracm  = re.search(r"(\d+)/(\d+)", text)
    inches = int(im.group(1)) if im else 0
    frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
    if inches > 12: inches = int(str(inches)[0])
    return math.ceil(feet+(inches+frac)/12)
 
def parse_metric(val_str):
    """Parse metric: 3.50 → 11.5ft, 0.23 → thickness."""
    try:
        v = float(val_str)
        return round(v * 3.281, 1)
    except: return None
 
# ══════════════════════════════════════════════
#  ZONE DIMENSION PARSER — all formats
# ══════════════════════════════════════════════
def zone_dims(base, r=450):
    for e in elements:
        if dist(base,e) > r: continue
        t = e["text"]
 
        # Format 1: 12'0"X15'0" imperial with X
        m = re.search(r"(\d+['']\d+[\"']?)\s*[xX×]\s*(\d+['']\d+[\"']?)", t)
        if m:
            ww = parse_feet_inches(m.group(1))
            ll = parse_feet_inches(m.group(2))
            if ww and ll and 4<=ww<=100 and 4<=ll<=100:
                return min(ww,ll), max(ww,ll)
 
        # Format 2: 12'0"15'0" imperial no separator
        m = re.search(r"(\d+['']\d+)[\"']\s*(\d+['']\d+)", t)
        if m:
            ww = parse_feet_inches(m.group(1))
            ll = parse_feet_inches(m.group(2))
            if ww and ll and 4<=ww<=100 and 4<=ll<=100:
                return min(ww,ll), max(ww,ll)
 
        # Format 3: 3.50 × 4.25 metric decimals
        m = re.search(r"(\d+\.\d+)\s*[×xX]\s*(\d+\.\d+)", t)
        if m:
            ww = parse_metric(m.group(1))
            ll = parse_metric(m.group(2))
            if ww and ll and 4<=ww<=150 and 4<=ll<=150:
                return min(ww,ll), max(ww,ll)
 
        # Format 4: simple NxM integers (feet) — skip if looks like window/door
        m2 = re.search(r"(\d{1,2})\s*[xX]\s*(\d{1,2})", t)
        if m2:
            ww,ll = int(m2.group(1)), int(m2.group(2))
            if 5<=ww<=80 and 5<=ll<=80 and not matches_any(t, WINDOW_PATTERNS+DOOR_PATTERNS):
                return min(ww,ll), max(ww,ll)
 
    return None, None
 
# ══════════════════════════════════════════════
#  STRATEGY 1 — Text label detection
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
 
def build_from_text_labels():
    for i, anchor in enumerate(ew_anchors):
        eid = f"ew{i+1}"
        nb  = nearby(anchor, 280)
        # Accept both integer and decimal T/H values
        t_v = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
        h_v = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
        # Convert metric thickness/height to display units
        if t_v and drawing_unit == "m":
            t_display = round(t_v * 39.37)   # meters → inches
            h_display = round(h_v * 3.281) if h_v else 10
        else:
            t_display = int(t_v) if t_v else 9
            h_display = int(h_v) if h_v else 10
        d = find_dir(nb) or pos_by_location(anchor["x"], anchor["y"])
        external_walls[eid] = {
            "id":eid, "position":d,
            "length_ft":None, "thickness_in":t_display, "height_ft":h_display,
            "connected":{"windows":[],"doors":[],"internal_walls":[]}
        }
        anchor["id"] = eid
 
    for i, anchor in enumerate(iw_anchors):
        iwid = f"iw{i+1}"
        nb   = nearby(anchor, 280)
        t_v  = find_val_float(nb, r"T[=\s](\d+\.?\d*)")
        h_v  = find_val_float(nb, r"H[=\s](\d+\.?\d*)")
        if t_v and drawing_unit == "m":
            t_display = round(t_v * 39.37)
            h_display = round(h_v * 3.281) if h_v else 10
        else:
            t_display = int(t_v) if t_v else 4
            h_display = int(h_v) if h_v else 10
        d    = find_dir(nb)
        conn = []
        for ea in sorted(ew_anchors, key=lambda a: dist(anchor,a))[:2]:
            conn.append(ea["id"])
            if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
                external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
        internal_walls[iwid] = {
            "id":iwid, "position":d,
            "thickness_in":t_display, "height_ft":h_display, "length_ft":None,
            "connects_external":conn,
            "connected":{"doors":[],"windows":[]}
        }
        anchor["id"] = iwid
 
# ══════════════════════════════════════════════
#  STRATEGY 2 — OpenCV line detection
# ══════════════════════════════════════════════
def build_from_opencv():
    img   = raw_images[0]
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    gray  = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
    edges = cv2.Canny(gray,40,120,apertureSize=3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,
                             minLineLength=img_w//12,maxLineGap=25)
    h_lines,v_lines=[],[]
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            angle=abs(math.degrees(math.atan2(y2-y1,x2-x1)))
            length=math.hypot(x2-x1,y2-y1)
            if angle<15 or angle>165: h_lines.append((min(x1,x2),max(x1,x2),(y1+y2)//2,length))
            elif 75<angle<105:        v_lines.append(((x1+x2)//2,min(y1,y2),max(y1,y2),length))
 
    def cluster(lines,axis,gap=35):
        if not lines: return []
        lines=sorted(lines,key=lambda l:l[2] if axis=='h' else l[0])
        out=[lines[0]]
        for ln in lines[1:]:
            coord=ln[2] if axis=='h' else ln[0]; prev=out[-1][2] if axis=='h' else out[-1][0]
            if abs(coord-prev)<gap: out[-1]=ln if ln[3]>out[-1][3] else out[-1]
            else: out.append(ln)
        return out
 
    ew_c=0; iw_c=0; m=0.12
    for (x1,x2,y,l) in cluster(h_lines,'h'):
        lft=round(l/(img_w/30))
        if y<img_h*m or y>img_h*(1-m):
            ew_c+=1; eid=f"ew{ew_c}"; pos="north" if y<img_h/2 else "south"
            external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
            ew_anchors.append({"x":(x1+x2)//2,"y":y,"id":eid})
        else:
            iw_c+=1; iwid=f"iw{iw_c}"
            internal_walls[iwid]={"id":iwid,"position":"horizontal","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
            iw_anchors.append({"x":(x1+x2)//2,"y":y,"id":iwid})
 
    for (x,y1,y2,l) in cluster(v_lines,'v'):
        lft=round(l/(img_h/30))
        if x<img_w*m or x>img_w*(1-m):
            ew_c+=1; eid=f"ew{ew_c}"; pos="west" if x<img_w/2 else "east"
            external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
            ew_anchors.append({"x":x,"y":(y1+y2)//2,"id":eid})
        else:
            iw_c+=1; iwid=f"iw{iw_c}"
            internal_walls[iwid]={"id":iwid,"position":"vertical","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
            iw_anchors.append({"x":x,"y":(y1+y2)//2,"id":iwid})
 
    for iwa in iw_anchors:
        iwid=iwa["id"]
        for ea in sorted(ew_anchors,key=lambda a:dist(iwa,a))[:2]:
            if ea["id"] not in internal_walls[iwid]["connects_external"]: internal_walls[iwid]["connects_external"].append(ea["id"])
            if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]: external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
    return ew_c>0
 
# ══════════════════════════════════════════════
#  STRATEGY 3 — Auto 4 boundary walls
# ══════════════════════════════════════════════
def build_fallback():
    for pos,eid,x,y in [("north","ew1",img_w//2,5),("east","ew2",img_w-5,img_h//2),
                         ("south","ew3",img_w//2,img_h-5),("west","ew4",5,img_h//2)]:
        external_walls[eid]={"id":eid,"position":pos,"length_ft":None,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
        ew_anchors.append({"x":x,"y":y,"id":eid})
 
# ══════════════════════════════════════════════
#  RUN STRATEGY CHAIN
# ══════════════════════════════════════════════
strategy = "auto_boundary"
if ew_anchors or iw_anchors:
    build_from_text_labels(); strategy="text_labels"
elif raw_images and build_from_opencv():
    strategy="opencv_lines"
else:
    build_fallback()
 
# ══════════════════════════════════════════════
#  DOORS — universal detection
# ══════════════════════════════════════════════
door_list, door_seen = [], []
for e in elements:
    t=e["text"]
    w,h=extract_dim_float(t,DOOR_PATTERNS)
    if w and h:
        # Convert metric door dims to feet
        if drawing_unit=="m" and w<5:
            w=round(w*3.281,1); h=round(h*3.281,1)
        if not is_dup(e["x"],e["y"],door_seen):
            door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
            door_seen.append((e["x"],e["y"]))
    elif matches_any(t,[r"^D\d+$",r"^DOOR\d+$",r"^DR\d+$",r"^PORTE\d*$"]) and not is_dup(e["x"],e["y"],door_seen):
        nb=nearby(e,150)
        wv=find_val_float(nb,r"(\d+\.?\d*)[xX]\d+\.?\d*") or 3
        hv=find_val_float(nb,r"\d+\.?\d*[xX](\d+\.?\d*)") or 7
        if drawing_unit=="m" and wv<5: wv=round(wv*3.281,1); hv=round(hv*3.281,1)
        door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":wv,"height_ft":hv})
        door_seen.append((e["x"],e["y"]))
 
# ══════════════════════════════════════════════
#  WINDOWS — universal detection
# ══════════════════════════════════════════════
window_list, win_seen = [], []
for e in elements:
    t=e["text"]
    w,h=extract_dim_float(t,WINDOW_PATTERNS)
    if w and h:
        if drawing_unit=="m" and w<5:
            w=round(w*3.281,1); h=round(h*3.281,1)
        if not is_dup(e["x"],e["y"],win_seen):
            window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
            win_seen.append((e["x"],e["y"]))
    elif matches_any(t,[r"^W\d+$",r"^WIN\d+$",r"^WINDOW\d+$",r"^FENETRE\d*$"]) and not is_dup(e["x"],e["y"],win_seen):
        nb=nearby(e,150)
        wv=find_val_float(nb,r"(\d+\.?\d*)[xX]\d+\.?\d*") or 4
        hv=find_val_float(nb,r"\d+\.?\d*[xX](\d+\.?\d*)") or 4
        if drawing_unit=="m" and wv<5: wv=round(wv*3.281,1); hv=round(hv*3.281,1)
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
ZONE_PAT=re.compile(r"^ZONE\s*\d+[A-Z]?$")
zones_out={}
 
def nearest_room(base,r=400):
    for e in sorted(elements,key=lambda e:dist(base,e)):
        if dist(base,e)>r: break
        for kw in ROOM_KEYWORDS:
            if kw in e["text"]: return e["text"]
    return "UNKNOWN"
 
for e in elements:
    if ZONE_PAT.match(e["text"]):
        zid=re.sub(r"\s+","",e["text"])
        name=nearest_room(e)
        wft,lft=zone_dims(e)
        area=round(wft*lft,2) if wft and lft else None
        px_ft=img_w/40.0
        zp=math.sqrt(area)*px_ft*0.6 if area else img_w*0.25
 
        conn_ew=list(dict.fromkeys(a["id"] for a in ew_anchors if dist(e,a)<zp*1.3))
        conn_iw=list(dict.fromkeys(a["id"] for a in iw_anchors if dist(e,a)<zp*1.1))
        conn_d =list(dict.fromkeys(d["id"] for d in door_list   if dist(e,{"x":d["x"],"y":d["y"]})<zp))
        conn_w =list(dict.fromkeys(w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp))
 
        zones_out[zid]={
            "id":zid,"label":name,"width_ft":wft,"length_ft":lft,"area_sqft":area,
            "connected_external_walls":conn_ew,
            "connected_internal_walls":conn_iw,
            "total_walls_connected":len(conn_ew)+len(conn_iw),
            "doors":conn_d,"windows":conn_w
        }
 
# ══════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
#  DEDUP SUMMARY — no double counting
# ══════════════════════════════════════════════
 
# Area: sum each unique zone once only
# Bathrooms are sub-zones — already included inside parent zone area
# so we only sum zones that are NOT fully contained inside another zone
# Simple rule: if a zone's area < 100 sqft AND parent zone exists nearby, skip it
# Better rule: just sum ALL unique zone areas as-is (each zone is distinct space)
# The real dedup needed is for doors/windows assigned to multiple zones
 
# Deduplicate doors across zones — each door counted once
all_zone_doors    = set(d for z in zones_out.values() for d in z["doors"])
all_zone_windows  = set(w for z in zones_out.values() for w in z["windows"])
all_zone_ew       = set(e for z in zones_out.values() for e in z["connected_external_walls"])
all_zone_iw       = set(i for z in zones_out.values() for i in z["connected_internal_walls"])
 
# Total area: sum unique zones (no zone appears twice)
total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]), 2)
 
# Net wall area calculation (deduped)
# Each wall counted once regardless of how many zones reference it
unique_walls = {}
for wid, wdata in {**external_walls, **internal_walls}.items():
    l = wdata.get("length_ft") or 0
    h = wdata.get("height_ft") or 10
    unique_walls[wid] = round(l * h, 2) if l else 0
 
# Dedup doors/windows in zone lists — remove duplicates within each zone
for zid in zones_out:
    zones_out[zid]["doors"]   = list(dict.fromkeys(zones_out[zid]["doors"]))
    zones_out[zid]["windows"] = list(dict.fromkeys(zones_out[zid]["windows"]))
    zones_out[zid]["connected_external_walls"] = list(dict.fromkeys(zones_out[zid]["connected_external_walls"]))
    zones_out[zid]["connected_internal_walls"] = list(dict.fromkeys(zones_out[zid]["connected_internal_walls"]))
    zones_out[zid]["total_walls_connected"] = (
        len(zones_out[zid]["connected_external_walls"]) +
        len(zones_out[zid]["connected_internal_walls"])
    )
 
print(json.dumps({
    "external_walls":external_walls,
    "internal_walls":internal_walls,
    "doors":doors_out,
    "windows":windows_out,
    "zones":zones_out,
    "summary":{
        "total_external_walls": len(external_walls),
        "total_internal_walls": len(internal_walls),
        "total_doors":          len(doors_out),          # unique doors in drawing
        "total_windows":        len(windows_out),        # unique windows in drawing
        "total_zones":          len(zones_out),
        "total_area_sqft":      total_area,              # sum of all zone areas
        "unique_doors_in_zones":    len(all_zone_doors),     # deduped across zones
        "unique_windows_in_zones":  len(all_zone_windows),   # deduped across zones
        "drawing_unit":         drawing_unit,
        "detection_strategy":   strategy
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
#     r"^EW[\s,=T\d]*$", r"^EW\d+[A-Z]?$", r"^E\.W\.?\d*$",
#     r"^E\d+$", r"^EXT[\s\-]?WALL\d*$", r"^EXTERNAL[\s\-]?WALL\d*$",
#     r"^OUTER[\s\-]?WALL\d*$", r"^BOUNDARY[\s\-]?WALL\d*$",
# ]
# IW_PATTERNS = [
#     r"^IW[\s,=T\d]*$", r"^IW\d+[A-Z]?$", r"^I\.W\.?\d*$",
#     r"^I\d+$", r"^INT[\s\-]?WALL\d*$", r"^INTERNAL[\s\-]?WALL\d*$",
#     r"^INNER[\s\-]?WALL\d*$", r"^PARTITION[\s\-]?\d*$",
#     r"^DIVIDER[\s\-]?\d*$", r"^PW\d*$",
# ]
# DOOR_PATTERNS = [
#     r"^D[=\-\s]?(\d+)[xX×](\d+)$",
#     r"^DOOR[=\-\s]?(\d+)[xX×](\d+)$",
#     r"^DR[=\-\s]?(\d+)[xX×](\d+)$",
#     r"^D\d+$", r"^DOOR\d*$", r"^DR\d*$",
# ]
# WINDOW_PATTERNS = [
#     r"^W[=\-\s]?(\d+)[xX×](\d+)$",
#     r"^WIN[=\-\s]?(\d+)[xX×](\d+)$",
#     r"^WINDOW[=\-\s]?(\d+)[xX×](\d+)$",
#     r"^W\d+$", r"^WIN\d*$", r"^WINDOW\d*$",
# ]
# ROOM_KEYWORDS = [
#     "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL",
#     "BALCONY","STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM",
#     "C.BEDROOM","MASTER","DRAWING","UTILITY","GARAGE","PASSAGE",
#     "WASH","BATH","BED","FAMILY","PANTRY","LAUNDRY","TERRACE",
#     "VERANDAH","COURTYARD","PORCH","ENTRANCE","FOYER","MBEDROOM","CBEDROOM",
# ]
 
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
#     for (bbox, text, prob) in reader.readtext(preprocess(image)):
#         if prob < MIN_CONF: continue
#         text = text.upper().strip()
#         x = int(sum([p[0] for p in bbox]) / 4)
#         y = int(sum([p[1] for p in bbox]) / 4)
#         elements.append({"text": text, "x": x, "y": y})
 
# # ══════════════════════════════════════════════
# #  CLEAN — normalize ALL OCR noise
# # ══════════════════════════════════════════════
# def clean(t):
#     t = t.upper().strip()
#     # Fix IW misreads
#     t = re.sub(r"\bEW,\b","EW",t)
#     t = t.replace("1W","IW").replace("lW","IW").replace("lw","IW").replace("|W","IW")
#     # KEY FIX: normalize dash to equals for dimension markers
#     # T-4 → T=4,  H-10 → H=10,  D-3X7 → D=3X7,  W-6X8 → W=6X8
#     t = re.sub(r'\b([TH])-(\d)', r'\1=\2', t)
#     t = re.sub(r'\b([DW])-(\d)', r'\1=\2', t)
#     # Fix OCR letter/digit confusion: A→4, O→0 in dimension context
#     # W=AX4 → W=4X4,  D=AX7 → D=4X7
#     t = re.sub(r'\b([DW]=)([A-Z])X', lambda m: m.group(1)+'4X', t)
#     t = re.sub(r'X([A-Z])\b', lambda m: 'X4', t)
#     # Clean dots and extra spaces
#     t = t.replace("EW.","EW").replace("IW.","IW")
#     t = re.sub(r"\s+"," ",t)
#     return t.strip()
 
# for e in elements: e["text"] = clean(e["text"])
 
# img_h = raw_images[0].shape[0] if raw_images else 1000
# img_w = raw_images[0].shape[1] if raw_images else 1000
 
# # ══════════════════════════════════════════════
# #  HELPERS
# # ══════════════════════════════════════════════
# def dist(a,b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
# def is_dup(x,y,pos,th=50): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)
# def nearby(base,r=220): return [e for e in elements if dist(base,e)<r]
 
# def matches_any(text, patterns):
#     return any(re.match(p, text) for p in patterns)
 
# def extract_dim(text, patterns):
#     for p in patterns:
#         m = re.match(p, text)
#         if m and m.lastindex and m.lastindex >= 2:
#             return int(m.group(1)), int(m.group(2))
#     return None, None
 
# def find_val(elems, pat):
#     """Accept both = and - as separators."""
#     for e in elems:
#         # normalize before matching
#         txt = re.sub(r'\b([TH])-(\d)', r'\1=\2', e["text"])
#         m = re.search(pat, txt)
#         if m: return int(m.group(1))
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
 
# # ══════════════════════════════════════════════
# #  ZONE DIMENSION PARSER — handles all formats
# # ══════════════════════════════════════════════
# def zone_dims(base, r=450):
#     for e in elements:
#         if dist(base,e) > r: continue
#         t = e["text"]
 
#         # Format 1: 12'0"X15'0" or 12'0" X 15'0" (standard with X)
#         m = re.search(r"(\d+['']\d+[\"']?)\s*[xX×]\s*(\d+['']\d+[\"']?)", t)
#         if m:
#             ww = parse_feet_inches(m.group(1))
#             ll = parse_feet_inches(m.group(2))
#             if ww and ll and 4<=ww<=60 and 4<=ll<=60:
#                 return min(ww,ll), max(ww,ll)
 
#         # Format 2: 12'0"15'0" (no separator — KEY FIX for this drawing)
#         m = re.search(r"(\d+['']\d+)[\"']\s*(\d+['']\d+)", t)
#         if m:
#             ww = parse_feet_inches(m.group(1))
#             ll = parse_feet_inches(m.group(2))
#             if ww and ll and 4<=ww<=60 and 4<=ll<=60:
#                 return min(ww,ll), max(ww,ll)
 
#         # Format 3: 12x15 or 12X15 (simple numbers, only if no feet found nearby)
#         m2 = re.search(r"(\d{1,2})\s*[xX]\s*(\d{1,2})", t)
#         if m2:
#             ww,ll = int(m2.group(1)), int(m2.group(2))
#             # Only use if reasonable room size AND not a window/door label
#             if 5<=ww<=50 and 5<=ll<=50 and not matches_any(t, WINDOW_PATTERNS+DOOR_PATTERNS):
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
#         nb  = nearby(anchor, 260)
#         t_v = find_val(nb, r"T[=\s](\d+)")
#         h_v = find_val(nb, r"H[=\s](\d+)")
#         d   = find_dir(nb) or pos_by_location(anchor["x"], anchor["y"])
#         external_walls[eid] = {
#             "id":eid, "position":d,
#             "length_ft":None, "thickness_in":t_v or 9, "height_ft":h_v or 10,
#             "connected":{"windows":[],"doors":[],"internal_walls":[]}
#         }
#         anchor["id"] = eid
 
#     for i, anchor in enumerate(iw_anchors):
#         iwid = f"iw{i+1}"
#         nb   = nearby(anchor, 260)
#         t_v  = find_val(nb, r"T[=\s](\d+)")
#         h_v  = find_val(nb, r"H[=\s](\d+)")
#         d    = find_dir(nb)
#         conn = []
#         for ea in sorted(ew_anchors, key=lambda a: dist(anchor,a))[:2]:
#             conn.append(ea["id"])
#             if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
#                 external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
#         internal_walls[iwid] = {
#             "id":iwid, "position":d,
#             "thickness_in":t_v or 4, "height_ft":h_v or 10, "length_ft":None,
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
 
#     ew_c=0; iw_c=0; m=0.12
#     for (x1,x2,y,l) in cluster(h_lines,'h'):
#         lft=round(l/(img_w/30))
#         if y<img_h*m or y>img_h*(1-m):
#             ew_c+=1; eid=f"ew{ew_c}"; pos="north" if y<img_h/2 else "south"
#             external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,"thickness_in":9,"height_ft":10,"connected":{"windows":[],"doors":[],"internal_walls":[]}}
#             ew_anchors.append({"x":(x1+x2)//2,"y":y,"id":eid})
#         else:
#             iw_c+=1; iwid=f"iw{iw_c}"
#             internal_walls[iwid]={"id":iwid,"position":"horizontal","length_ft":lft,"thickness_in":4,"height_ft":10,"connects_external":[],"connected":{"doors":[],"windows":[]}}
#             iw_anchors.append({"x":(x1+x2)//2,"y":y,"id":iwid})
 
#     for (x,y1,y2,l) in cluster(v_lines,'v'):
#         lft=round(l/(img_h/30))
#         if x<img_w*m or x>img_w*(1-m):
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
# #  DOORS
# # ══════════════════════════════════════════════
# door_list, door_seen = [], []
# for e in elements:
#     t=e["text"]
#     w,h=extract_dim(t,DOOR_PATTERNS)
#     if w and h and not is_dup(e["x"],e["y"],door_seen):
#         door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h}); door_seen.append((e["x"],e["y"]))
#     elif matches_any(t,[r"^D\d+$",r"^DOOR\d+$",r"^DR\d+$"]) and not is_dup(e["x"],e["y"],door_seen):
#         nb=nearby(e,150)
#         door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],
#             "width_ft":find_val(nb,r"(\d+)[xX]\d+") or 3,"height_ft":find_val(nb,r"\d+[xX](\d+)") or 7})
#         door_seen.append((e["x"],e["y"]))
 
# # ══════════════════════════════════════════════
# #  WINDOWS
# # ══════════════════════════════════════════════
# window_list, win_seen = [], []
# for e in elements:
#     t=e["text"]
#     w,h=extract_dim(t,WINDOW_PATTERNS)
#     if w and h and not is_dup(e["x"],e["y"],win_seen):
#         window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h}); win_seen.append((e["x"],e["y"]))
#     elif matches_any(t,[r"^W\d+$",r"^WIN\d+$",r"^WINDOW\d+$"]) and not is_dup(e["x"],e["y"],win_seen):
#         nb=nearby(e,150)
#         window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],
#             "width_ft":find_val(nb,r"(\d+)[xX]\d+") or 4,"height_ft":find_val(nb,r"\d+[xX](\d+)") or 4})
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
# #  ZONES
# # ══════════════════════════════════════════════
# ZONE_PAT=re.compile(r"^ZONE\s*\d+[A-Z]?$")
# zones_out={}
 
# def nearest_room(base,r=350):
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
#         px_ft=img_w/40.0
#         zp=math.sqrt(area)*px_ft*0.6 if area else img_w*0.25
 
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
# #  OUTPUT
# # ══════════════════════════════════════════════
# print(json.dumps({
#     "external_walls":external_walls,
#     "internal_walls":internal_walls,
#     "doors":doors_out,
#     "windows":windows_out,
#     "zones":zones_out,
#     "summary":{
#         "total_external_walls":len(external_walls),
#         "total_internal_walls":len(internal_walls),
#         "total_doors":len(doors_out),
#         "total_windows":len(windows_out),
#         "total_zones":len(zones_out),
#         "total_area_sqft":round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]),2),
#         "detection_strategy":strategy
#     }
# }))
