import cv2, easyocr, re, json, numpy as np, math, sys
from pdf2image import convert_from_path
 
MIN_CONF   = 0.28          # slightly lower for hand drawings
reader     = easyocr.Reader(['en'], gpu=False)
input_path = sys.argv[1]
 
# ══════════════════════════════════════════════
#  UNIVERSAL LABEL PATTERNS
#  Covers every notation style architects use
# ══════════════════════════════════════════════
EW_PATTERNS = [
    r"^EW[\s,=T\d]*$",          # EW, EW1, EW T=9, EW,T=9
    r"^EW\d+[A-Z]?$",           # EW1, EW2A
    r"^E\.W\.?\d*$",            # E.W., E.W.1
    r"^E\d+$",                  # E1, E2, E3
    r"^EXT[\s\-]?WALL\d*$",     # EXT WALL, EXT-WALL1
    r"^EXTERNAL[\s\-]?WALL\d*$",# EXTERNAL WALL
    r"^OUTER[\s\-]?WALL\d*$",   # OUTER WALL
    r"^BOUNDARY[\s\-]?WALL\d*$",# BOUNDARY WALL
    r"^PERIMETER[\s\-]?WALL\d*$",
]
 
IW_PATTERNS = [
    r"^IW[\s,=T\d]*$",          # IW, IW1, IW T=4, IW,T=4
    r"^IW\d+[A-Z]?$",           # IW1, IW2A
    r"^I\.W\.?\d*$",            # I.W., I.W.1
    r"^I\d+$",                  # I1, I2, I3
    r"^INT[\s\-]?WALL\d*$",     # INT WALL, INT-WALL
    r"^INTERNAL[\s\-]?WALL\d*$",# INTERNAL WALL
    r"^INNER[\s\-]?WALL\d*$",   # INNER WALL
    r"^PARTITION[\s\-]?\d*$",   # PARTITION, PARTITION1
    r"^DIVIDER[\s\-]?\d*$",     # DIVIDER
    r"^PW\d*$",                 # PW, PW1 (partition wall)
]
 
DOOR_PATTERNS = [
    r"^D[=\s]?(\d+)[xX×](\d+)$",    # D=3x7, D3x7, D3X7
    r"^DOOR[=\s]?(\d+)[xX×](\d+)$", # DOOR3x7
    r"^DR[=\s]?(\d+)[xX×](\d+)$",   # DR3x7
    r"^D\d+$",                        # D1, D2
    r"^DOOR\d*$",                     # DOOR, DOOR1
    r"^DR\d*$",                       # DR, DR1
]
 
WINDOW_PATTERNS = [
    r"^W[=\s]?(\d+)[xX×](\d+)$",      # W=4x4, W4x4
    r"^WIN[=\s]?(\d+)[xX×](\d+)$",    # WIN4x4
    r"^WINDOW[=\s]?(\d+)[xX×](\d+)$", # WINDOW4x4
    r"^W\d+$",                          # W1, W2
    r"^WIN\d*$",                        # WIN, WIN1
    r"^WINDOW\d*$",                     # WINDOW, WINDOW1
]
 
ROOM_KEYWORDS = [
    "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL",
    "BALCONY","STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM",
    "C.BEDROOM","MASTER","DRAWING","UTILITY","GARAGE","PASSAGE",
    "WASH","BATH","BED","LIVING ROOM","FAMILY","PANTRY","LAUNDRY",
    "TERRACE","VERANDAH","COURTYARD","PORCH","ENTRANCE","FOYER",
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
    # Enhance for hand drawings
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
 
# Clean OCR noise common in hand drawings
def clean(t):
    t = t.upper().strip()
    # Fix common OCR misreads
    t = re.sub(r"\bEW,\b","EW",t)
    t = t.replace("1W","IW").replace("lW","IW").replace("lw","IW")
    t = t.replace("0W","DW").replace("|W","IW")
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
 
def extract_dim(text, patterns):
    """Extract WxH from text using door/window patterns."""
    for p in patterns:
        m = re.match(p, text)
        if m and m.lastindex and m.lastindex >= 2:
            return int(m.group(1)), int(m.group(2))
    return None, None
 
def find_val(elems, pat):
    for e in elems:
        m = re.search(pat, e["text"])
        if m: return int(m.group(1))
    return None
 
def find_dir(elems):
    dirs = ["NORTH","SOUTH","EAST","WEST","MIDDLE","CENTER","TOP","BOTTOM",
            "LEFT","RIGHT","FRONT","BACK","REAR"]
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
 
def parse_feet_inches(text):
    text = text.replace(" ","")
    fm   = re.search(r"(\d+)'", text)
    if not fm: return None
    feet   = int(fm.group(1))
    im     = re.search(r"'(\d+)", text)
    fracm  = re.search(r"(\d+)/(\d+)", text)
    inches = int(im.group(1)) if im else 0
    frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
    if inches > 12: inches = int(str(inches)[0])
    return math.ceil(feet+(inches+frac)/12)
 
# ══════════════════════════════════════════════
#  STRATEGY 1 — Universal text label matching
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
        nb  = nearby(anchor, 260)
        t   = find_val(nb, r"T[=\s](\d+)")
        h   = find_val(nb, r"H[=\s](\d+)")
        d   = find_dir(nb) or pos_by_location(anchor["x"],anchor["y"])
        external_walls[eid] = {
            "id":eid,"position":d,
            "length_ft":None,"thickness_in":t or 9,"height_ft":h or 10,
            "connected":{"windows":[],"doors":[],"internal_walls":[]}
        }
        anchor["id"] = eid
 
    for i, anchor in enumerate(iw_anchors):
        iwid = f"iw{i+1}"
        nb   = nearby(anchor, 260)
        t    = find_val(nb, r"T[=\s](\d+)")
        h    = find_val(nb, r"H[=\s](\d+)")
        d    = find_dir(nb)
        conn = []
        for ea in sorted(ew_anchors, key=lambda a: dist(anchor,a))[:2]:
            conn.append(ea["id"])
            if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
                external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
        internal_walls[iwid] = {
            "id":iwid,"position":d,
            "thickness_in":t or 4,"height_ft":h or 10,"length_ft":None,
            "connects_external":conn,
            "connected":{"doors":[],"windows":[]}
        }
        anchor["id"] = iwid
 
# ══════════════════════════════════════════════
#  STRATEGY 2 — OpenCV line detection
#  Used when drawing has NO text wall labels
# ══════════════════════════════════════════════
def build_from_opencv():
    img   = raw_images[0]
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    # Use morphological ops to enhance thin lines (hand drawings)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    gray   = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    edges  = cv2.Canny(gray,40,120,apertureSize=3)
    lines  = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,
                              minLineLength=img_w//12,maxLineGap=25)
    h_lines, v_lines = [], []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle  = abs(math.degrees(math.atan2(y2-y1,x2-x1)))
            length = math.hypot(x2-x1,y2-y1)
            if angle<15 or angle>165:
                h_lines.append((min(x1,x2),max(x1,x2),(y1+y2)//2,length))
            elif 75<angle<105:
                v_lines.append(((x1+x2)//2,min(y1,y2),max(y1,y2),length))
 
    def cluster(lines,axis,gap=35):
        if not lines: return []
        lines = sorted(lines,key=lambda l:l[2] if axis=='h' else l[0])
        out   = [lines[0]]
        for ln in lines[1:]:
            coord = ln[2] if axis=='h' else ln[0]
            prev  = out[-1][2] if axis=='h' else out[-1][0]
            out[-1] = ln if abs(coord-prev)<gap and ln[3]>out[-1][3] else out[-1]
            if abs(coord-prev)>=gap: out.append(ln)
        return out
 
    ew_c=0; iw_c=0; m=0.12
    for (x1,x2,y,length) in cluster(h_lines,'h'):
        lft = round(length/(img_w/30))
        if y<img_h*m or y>img_h*(1-m):
            ew_c+=1; eid=f"ew{ew_c}"
            pos="north" if y<img_h/2 else "south"
            external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,
                "thickness_in":9,"height_ft":10,
                "connected":{"windows":[],"doors":[],"internal_walls":[]}}
            ew_anchors.append({"x":(x1+x2)//2,"y":y,"id":eid})
        else:
            iw_c+=1; iwid=f"iw{iw_c}"
            internal_walls[iwid]={"id":iwid,"position":"horizontal","length_ft":lft,
                "thickness_in":4,"height_ft":10,"connects_external":[],
                "connected":{"doors":[],"windows":[]}}
            iw_anchors.append({"x":(x1+x2)//2,"y":y,"id":iwid})
 
    for (x,y1,y2,length) in cluster(v_lines,'v'):
        lft = round(length/(img_h/30))
        if x<img_w*m or x>img_w*(1-m):
            ew_c+=1; eid=f"ew{ew_c}"
            pos="west" if x<img_w/2 else "east"
            external_walls[eid]={"id":eid,"position":pos,"length_ft":lft,
                "thickness_in":9,"height_ft":10,
                "connected":{"windows":[],"doors":[],"internal_walls":[]}}
            ew_anchors.append({"x":x,"y":(y1+y2)//2,"id":eid})
        else:
            iw_c+=1; iwid=f"iw{iw_c}"
            internal_walls[iwid]={"id":iwid,"position":"vertical","length_ft":lft,
                "thickness_in":4,"height_ft":10,"connects_external":[],
                "connected":{"doors":[],"windows":[]}}
            iw_anchors.append({"x":x,"y":(y1+y2)//2,"id":iwid})
 
    for iwa in iw_anchors:
        iwid=iwa["id"]
        for ea in sorted(ew_anchors,key=lambda a:dist(iwa,a))[:2]:
            if ea["id"] not in internal_walls[iwid]["connects_external"]:
                internal_walls[iwid]["connects_external"].append(ea["id"])
            if iwid not in external_walls[ea["id"]]["connected"]["internal_walls"]:
                external_walls[ea["id"]]["connected"]["internal_walls"].append(iwid)
    return ew_c>0
 
# ══════════════════════════════════════════════
#  STRATEGY 3 — Auto 4 boundary walls (last resort)
# ══════════════════════════════════════════════
def build_fallback():
    for pos,eid,x,y in [("north","ew1",img_w//2,5),("east","ew2",img_w-5,img_h//2),
                         ("south","ew3",img_w//2,img_h-5),("west","ew4",5,img_h//2)]:
        external_walls[eid]={"id":eid,"position":pos,"length_ft":None,
            "thickness_in":9,"height_ft":10,
            "connected":{"windows":[],"doors":[],"internal_walls":[]}}
        ew_anchors.append({"x":x,"y":y,"id":eid})
 
# ══════════════════════════════════════════════
#  RUN STRATEGY CHAIN
# ══════════════════════════════════════════════
if ew_anchors or iw_anchors:
    build_from_text_labels()                     # Strategy 1
elif raw_images and build_from_opencv():
    pass                                          # Strategy 2
else:
    build_fallback()                              # Strategy 3
 
# ══════════════════════════════════════════════
#  DOORS — universal detection
# ══════════════════════════════════════════════
door_list, door_seen = [], []
for e in elements:
    t = e["text"]
    w,h = extract_dim(t, DOOR_PATTERNS)
    if w and h:
        if not is_dup(e["x"],e["y"],door_seen):
            door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
            door_seen.append((e["x"],e["y"]))
    elif matches_any(t,[r"^D\d+$",r"^DOOR\d+$",r"^DR\d+$"]):
        nb=nearby(e,150)
        w2=find_val(nb,r"(\d+)[xX]\d+") or 3
        h2=find_val(nb,r"\d+[xX](\d+)") or 7
        if not is_dup(e["x"],e["y"],door_seen):
            door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w2,"height_ft":h2})
            door_seen.append((e["x"],e["y"]))
 
# ══════════════════════════════════════════════
#  WINDOWS — universal detection
# ══════════════════════════════════════════════
window_list, win_seen = [], []
for e in elements:
    t = e["text"]
    w,h = extract_dim(t, WINDOW_PATTERNS)
    if w and h:
        if not is_dup(e["x"],e["y"],win_seen):
            window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
            win_seen.append((e["x"],e["y"]))
    elif matches_any(t,[r"^W\d+$",r"^WIN\d+$",r"^WINDOW\d+$"]):
        nb=nearby(e,150)
        w2=find_val(nb,r"(\d+)[xX]\d+") or 4
        h2=find_val(nb,r"\d+[xX](\d+)") or 4
        if not is_dup(e["x"],e["y"],win_seen):
            window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w2,"height_ft":h2})
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
 
def nearest_room(base,r=350):
    for e in sorted(elements,key=lambda e:dist(base,e)):
        if dist(base,e)>r: break
        for kw in ROOM_KEYWORDS:
            if kw in e["text"]: return e["text"]
    return "UNKNOWN"
 
def zone_dims(base,r=450):
    for e in elements:
        if dist(base,e)<r:
            m=re.search(r"(\d+['\"]\d*(?:\s*\d+/\d+)?[\"']?)\s*[xX\"]\s*(\d+['\"]\d*(?:\s*\d+/\d+)?[\"']?)",e["text"])
            if m:
                ww=parse_feet_inches(m.group(1)); ll=parse_feet_inches(m.group(2))
                if ww and ll and 4<=ww<=60 and 4<=ll<=60: return min(ww,ll),max(ww,ll)
            m2=re.search(r"(\d{1,2})\s*[xX]\s*(\d{1,2})",e["text"])
            if m2:
                ww,ll=int(m2.group(1)),int(m2.group(2))
                if 4<=ww<=50 and 4<=ll<=50: return min(ww,ll),max(ww,ll)
    return None,None
 
for e in elements:
    if ZONE_PAT.match(e["text"]):
        zid=re.sub(r"\s+","",e["text"])
        name=nearest_room(e); wft,lft=zone_dims(e)
        area=round(wft*lft,2) if wft and lft else None
        px_ft=img_w/40.0
        zp=math.sqrt(area)*px_ft*0.6 if area else img_w*0.25
 
        zones_out[zid]={
            "id":zid,"label":name,"width_ft":wft,"length_ft":lft,"area_sqft":area,
            "connected_external_walls":list(dict.fromkeys(a["id"] for a in ew_anchors if dist(e,a)<zp*1.3)),
            "connected_internal_walls":list(dict.fromkeys(a["id"] for a in iw_anchors if dist(e,a)<zp*1.1)),
            "total_walls_connected":0,
            "doors":list(dict.fromkeys(d["id"] for d in door_list if dist(e,{"x":d["x"],"y":d["y"]})<zp)),
            "windows":list(dict.fromkeys(w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp)),
        }
        zones_out[zid]["total_walls_connected"]=(
            len(zones_out[zid]["connected_external_walls"])+
            len(zones_out[zid]["connected_internal_walls"])
        )
 
# ══════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════
print(json.dumps({
    "external_walls":external_walls,
    "internal_walls":internal_walls,
    "doors":doors_out,
    "windows":windows_out,
    "zones":zones_out,
    "summary":{
        "total_external_walls":len(external_walls),
        "total_internal_walls":len(internal_walls),
        "total_doors":len(doors_out),
        "total_windows":len(windows_out),
        "total_zones":len(zones_out),
        "total_area_sqft":round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]),2),
        "detection_strategy":(
            "text_labels" if (ew_seen or iw_seen) else
            "opencv_lines" if len(external_walls)>0 else
            "auto_boundary"
        )
    }
}))
