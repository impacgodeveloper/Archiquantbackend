import cv2, easyocr, re, json, numpy as np, math, sys
from collections import Counter
from pdf2image import convert_from_path

# ── Settings ──────────────────────────────────────────────────────────────────
MIN_CONF   = 0.30
reader     = easyocr.Reader(['en'], gpu=False)
input_path = sys.argv[1]

# ── Load ──────────────────────────────────────────────────────────────────────
if input_path.lower().endswith(".pdf"):
    pages = convert_from_path(input_path, dpi=500)
else:
    pages = [cv2.imread(input_path)]

elements = []

def preprocess(img):
    if not isinstance(img, np.ndarray):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return th

for page in pages:
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
    for (bbox, text, prob) in reader.readtext(preprocess(image)):
        if prob < MIN_CONF: continue
        x = int(sum([p[0] for p in bbox]) / 4)
        y = int(sum([p[1] for p in bbox]) / 4)
        elements.append({"text": text.upper().strip(), "x": x, "y": y})

img_h = max((e["y"] for e in elements), default=1000) * 1.1
img_w = max((e["x"] for e in elements), default=1000) * 1.1

# ── Clean ─────────────────────────────────────────────────────────────────────
def clean(t):
    t = t.upper().strip()
    # Fix wall labels
    t = t.replace("1W","IW").replace("lW","IW").replace("|W","IW")
    t = re.sub(r"\bEW,\b","EW",t)
    # EWA→EW4 (digit/letter confusion)
    _ld = {'A':'4','B':'2','C':'3','S':'5','G':'6','I':'1','O':'0'}
    t = re.sub(r'^EW([A-Z])$', lambda m: 'EW'+_ld.get(m.group(1),m.group(1)), t)
    # Dash→equals in specs: T-9→T=9, H-10→H=10
    t = re.sub(r'\b([TH])-(\d)', r'\1=\2', t)
    t = re.sub(r'\b([DW])-(\d)', r'\1=\2', t)
    # Normalize dimension separator → X
    t = t.replace('×','X').replace('*','X').replace('x','X')
    # Remove spaces around X in dims: "10' X 10'" → "10'X10'"
    t = re.sub(r"(\d)\s+X\s+(\d)", r"\1X\2", t)
    # Remove trailing apostrophe after dim: "4X4'" → "4X4"
    t = re.sub(r"(\d+X\d+)'$", r"\1", t)
    # Fix garbled specs
    t = re.sub(r'\bTZ(\d)', r'T=\1', t)
    t = re.sub(r'H-L(\d)', r'H=1\1', t)
    t = t.replace("T-G","T=9").replace("T=G","T=9")
    t = re.sub(r'\s+', ' ', t).strip()
    return t

for e in elements:
    e["text"] = clean(e["text"])

# ── Merge split ZONE tokens ───────────────────────────────────────────────────
# e.g. "ZONE" + "1" → "ZONE 1", "ZONE" + "4" → "ZONE 4"
sorted_by_pos = sorted(range(len(elements)), key=lambda i:(round(elements[i]["y"]/30), elements[i]["x"]))
used = set()
for ii, i in enumerate(sorted_by_pos):
    if i in used: continue
    if elements[i]["text"] == "ZONE":
        for j in sorted_by_pos[ii+1:ii+8]:
            if j in used: continue
            ne = elements[j]
            dx = abs(ne["x"] - elements[i]["x"])
            dy = abs(ne["y"] - elements[i]["y"])
            if dy < 50 and dx < 300 and re.match(r'^\d[A-Z]?$', ne["text"]):
                elements[i]["text"] = "ZONE " + ne["text"]
                used.add(j)
                break
elements = [e for i,e in enumerate(elements) if i not in used]

# Remove legend text (bottom 15% of image)
elements = [e for e in elements if e["y"] < img_h * 0.85]

# ── Helpers ───────────────────────────────────────────────────────────────────
def dist(a,b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
def is_dup(x,y,pos,th=25): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)

def parse_feet_inches(text):
    text = str(text).replace(" ","")
    fm = re.search(r"(\d+)'", text)
    if not fm: return None
    feet = int(fm.group(1))
    im   = re.search(r"'(\d+)", text)
    fracm= re.search(r"(\d+)/(\d+)", text)
    inches = int(im.group(1)) if im else 0
    frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
    if inches > 12: inches = int(str(inches)[0])
    return math.ceil(feet + (inches+frac)/12)

def ft_to_m(v):  return round(v*0.3048,2) if v else None
def sqft_to_sqm(v): return round(v*0.0929,2) if v else None

# ── Zone dimension parser ─────────────────────────────────────────────────────
def parse_zone_dims(base):
    for e in elements:
        if dist(base,e) > 700: continue
        t = e["text"]
        # Format: 12'0"X15'0"  or  12'0"15'0" (no X)
        m = re.search(r"(\d+['\`]\d+[\"']?)\s*X?\s*(\d+['\`]\d+[\"']?)", t)
        if m:
            w = parse_feet_inches(m.group(1))
            l = parse_feet_inches(m.group(2))
            if w and l and 4<=w<=100 and 4<=l<=100 and w!=l or (w and l and 4<=w<=100 and 4<=l<=100):
                return min(w,l), max(w,l)
        # Simple NxM
        m = re.search(r'(\d{1,2})\s*X\s*(\d{1,2})', t)
        if m:
            w,l = int(m.group(1)),int(m.group(2))
            if 4<=w<=80 and 4<=l<=80 and not re.match(r'^[DW]=',t):
                return min(w,l),max(w,l)
    return None,None

# ── Room name finder ──────────────────────────────────────────────────────────
ROOM_KW = [
    "BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA","HALL","BALCONY",
    "STORE","TOILET","LOBBY","STUDY","MBR","M.BEDROOM","C.BEDROOM","MASTER",
    "DRAWING","UTILITY","GARAGE","PASSAGE","WASH","BATH","FAMILY","PANTRY",
    "LAUNDRY","TERRACE","VERANDAH","ENTRANCE","FOYER","LIVING ROOM","DINING ROOM",
    "PRAYER","POWDER","KITTFEN","KTICHEN","KICHEN","KITCEN",  # OCR variants
    "CBEDROOM","MBEDROOM",
]

def nearest_room(base):
    ignore = ["ZONE","T=","H=","D=","W=","NORTH","SOUTH","EAST","WEST","IW","EW"]
    best,bd = None,999999
    for e in elements:
        txt = e["text"]
        if any(i in txt for i in ignore): continue
        for kw in ROOM_KW:
            if kw in txt:
                d = dist(base,e)
                if d < bd: best,bd = txt,d
    return best or "UNKNOWN"

# ── EW Anchors ────────────────────────────────────────────────────────────────
EW_RE = re.compile(r'^EW[\s,=T\d]*$|^EW\d+[A-Z]?$|^EW$')
ew_anchors, ew_seen = [], []
for e in elements:
    if EW_RE.match(e["text"]):
        if not is_dup(e["x"],e["y"],ew_seen,40):
            ew_anchors.append(dict(e)); ew_seen.append((e["x"],e["y"]))

# ── IW Anchors ────────────────────────────────────────────────────────────────
IW_RE = re.compile(r'^IW[\s,=T\d]*$|^IW\d+[A-Z]?$|^IW$|^IW[\s\(]')
iw_anchors, iw_seen = [], []
for e in elements:
    if IW_RE.match(e["text"]):
        if not is_dup(e["x"],e["y"],iw_seen,40):
            iw_anchors.append(dict(e)); iw_seen.append((e["x"],e["y"]))

def find_specs(base, r=250):
    t=h=None
    for e in elements:
        if dist(base,e)<r:
            txt=e["text"]
            if t is None:
                m=re.search(r'T=\s*(\d+\.?\d*)',txt)
                if m: t=int(float(m.group(1)))
            if h is None:
                m=re.search(r'H=\s*(\d+\.?\d*)',txt)
                if m: h=int(float(m.group(1)))
    return t,h

def find_dir(base, r=250):
    for e in elements:
        if dist(base,e)<r:
            for d in ["NORTH","SOUTH","EAST","WEST","MIDDLE","CENTER"]:
                if d in e["text"]: return d.lower()
    return None

def pos_by_location(x,y):
    m=0.15
    if y<img_h*m: return "north"
    elif y>img_h*(1-m): return "south"
    elif x<img_w*m: return "west"
    elif x>img_w*(1-m): return "east"
    return "inner"

# ── Build EW/IW maps ──────────────────────────────────────────────────────────
external_walls_map = {}
for i,anchor in enumerate(ew_anchors):
    eid = f"ew{i+1}"
    tv,hv = find_specs(anchor)
    pos   = find_dir(anchor) or pos_by_location(anchor["x"],anchor["y"])
    external_walls_map[eid] = {
        "id":eid,"position":pos,
        "thickness_in":tv or 9,"height_ft":hv or 10,
        "thickness_m":ft_to_m((tv or 9)/12),"height_m":ft_to_m(hv or 10),
        "connected":{"windows":[],"doors":[],"internal_walls":[]}
    }
    anchor["id"] = eid

internal_walls_map = {}
for i,anchor in enumerate(iw_anchors):
    iwid = f"iw{i+1}"
    tv,hv = find_specs(anchor)
    pos   = find_dir(anchor)
    conn  = []
    for ea in sorted(ew_anchors,key=lambda a:dist(anchor,a))[:2]:
        conn.append(ea["id"])
        if iwid not in external_walls_map[ea["id"]]["connected"]["internal_walls"]:
            external_walls_map[ea["id"]]["connected"]["internal_walls"].append(iwid)
    internal_walls_map[iwid] = {
        "id":iwid,"position":pos,
        "thickness_in":tv or 4,"height_ft":hv or 10,
        "thickness_m":ft_to_m((tv or 4)/12),"height_m":ft_to_m(hv or 10),
        "connects_external":conn,
        "connected":{"doors":[],"windows":[]}
    }
    anchor["id"] = iwid

# ── Doors ─────────────────────────────────────────────────────────────────────
door_list, door_seen = [], []
door_sizes = []
for e in elements:
    t=e["text"]
    m=re.search(r'D=(\d{1,2})X(\d{1,2})',t)
    if m:
        w,h=int(m.group(1)),int(m.group(2))
        if 2<=w<=6 and 6<=h<=9:
            if not is_dup(e["x"],e["y"],door_seen):
                door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
                door_seen.append((e["x"],e["y"])); door_sizes.append((w,h))

# ── Windows ───────────────────────────────────────────────────────────────────
window_list, win_seen = [], []
window_sizes = []
for e in elements:
    t=e["text"]
    m=re.search(r'W=(\d{1,2})X(\d{1,2})',t)
    if m:
        w,h=int(m.group(1)),int(m.group(2))
        if 2<=w<=10 and 2<=h<=10:
            if not is_dup(e["x"],e["y"],win_seen):
                window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})
                win_seen.append((e["x"],e["y"])); window_sizes.append((w,h))

# ── Assign on_wall ────────────────────────────────────────────────────────────
def nearest_wall(pos):
    best,bd=None,999999
    for a in ew_anchors+iw_anchors:
        d=dist(pos,a)
        if d<bd: bd,best=d,a["id"]
    return best

doors_out={}
for d in door_list:
    ow=nearest_wall({"x":d["x"],"y":d["y"]})
    doors_out[d["id"]]={"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],
        "width_m":ft_to_m(d["width_ft"]),"height_m":ft_to_m(d["height_ft"]),"on_wall":ow}
    if ow:
        wl=external_walls_map.get(ow) or internal_walls_map.get(ow)
        if wl and d["id"] not in wl["connected"]["doors"]: wl["connected"]["doors"].append(d["id"])

windows_out={}
for w in window_list:
    ow=nearest_wall({"x":w["x"],"y":w["y"]})
    windows_out[w["id"]]={"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],
        "width_m":ft_to_m(w["width_ft"]),"height_m":ft_to_m(w["height_ft"]),"on_wall":ow}
    if ow:
        wl=external_walls_map.get(ow) or internal_walls_map.get(ow)
        if wl and w["id"] not in wl["connected"]["windows"]: wl["connected"]["windows"].append(w["id"])

# ── Zones ─────────────────────────────────────────────────────────────────────
ZONE_RE = re.compile(r'^ZONE\s*\d+[A-Z]?$')
zones_out = {}
for e in elements:
    if not ZONE_RE.match(e["text"]): continue
    zid  = re.sub(r'\s+','',e["text"])
    name = nearest_room(e)
    wft,lft = parse_zone_dims(e)
    area = round(wft*lft,2) if wft and lft else None

    # Zone proximity radius — based on zone dimensions
    if wft and lft:
        px_ft = img_w / 40.0
        zp = min(wft,lft) * px_ft * 0.55
    else:
        zp = img_w * 0.20

    conn_ew = list(dict.fromkeys(a["id"] for a in ew_anchors if dist(e,a)<zp*1.3))
    conn_iw = list(dict.fromkeys(a["id"] for a in iw_anchors if dist(e,a)<zp*1.1))
    conn_d  = list(dict.fromkeys(d["id"] for d in door_list   if dist(e,{"x":d["x"],"y":d["y"]})<zp))
    conn_w  = list(dict.fromkeys(w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp))

    zones_out[zid] = {
        "id":zid,"label":name,"zone_id":e["text"],
        "size":{"length_ft":lft,"width_ft":wft} if wft else None,
        "area_sqft":area,"width_ft":wft,"length_ft":lft,
        "area_sqm":sqft_to_sqm(area),"width_m":ft_to_m(wft),"length_m":ft_to_m(lft),
        "connected_external_walls":conn_ew,"connected_internal_walls":conn_iw,
        "total_walls_connected":len(conn_ew)+len(conn_iw),
        "doors":conn_d,"windows":conn_w,
    }

# ── Legacy aggregation ────────────────────────────────────────────────────────
doors_agg   = [{"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c} for (w,h),c in Counter(door_sizes).items()]
windows_agg = [{"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c} for (w,h),c in Counter(window_sizes).items()]

iw_s=[]; ew_s=[]; vis=[]
for e in elements:
    if any(dist(e,v)<120 for v in vis): continue
    if "IW" in e["text"]:
        tv,hv=find_specs(e)
        if tv and hv and 3<=tv<=12 and 6<=hv<=12: iw_s.append((tv,hv)); vis.append(e)
    elif "EW" in e["text"]:
        tv,hv=find_specs(e)
        if tv and hv and 6<=tv<=15 and 8<=hv<=15: ew_s.append((tv,hv)); vis.append(e)

iw_agg=[{"type":"internal","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(iw_s).items()]
ew_agg=[{"type":"external","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(ew_s).items()]

all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])
total_area       = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]),2)

# ── Output ────────────────────────────────────────────────────────────────────
print(json.dumps({
    "zones":          zones_out,
    "external_walls": external_walls_map,
    "internal_walls": internal_walls_map,
    "doors":          doors_out,
    "windows":        windows_out,
    "openings":       {"doors":doors_agg,"windows":windows_agg},
    "walls":          {"internal":iw_agg,"external":ew_agg},
    "summary":{
        "total_area_sqft":         total_area,
        "total_area_sqm":          sqft_to_sqm(total_area),
        "total_zones":             len(zones_out),
        "total_doors":             len(doors_out),
        "total_windows":           len(windows_out),
        "total_external_walls":    len(external_walls_map),
        "total_internal_walls":    len(internal_walls_map),
        "total_door_area_sqft":    sum(d["total_area_sqft"] for d in doors_agg),
        "total_window_area_sqft":  sum(w["total_area_sqft"] for w in windows_agg),
        "total_opening_area_sqft": sum(d["total_area_sqft"] for d in doors_agg)+sum(w["total_area_sqft"] for w in windows_agg),
        "total_door_area_sqm":     sqft_to_sqm(sum(d["total_area_sqft"] for d in doors_agg)),
        "total_window_area_sqm":   sqft_to_sqm(sum(w["total_area_sqft"] for w in windows_agg)),
        "unique_doors_in_zones":   len(all_zone_doors),
        "unique_windows_in_zones": len(all_zone_windows),
        "detection_strategy":      "archiquant_standard"
    }
}))


# import cv2
# import easyocr
# import re
# import json
# import numpy as np
# import math
# import sys
# from collections import Counter
# from pdf2image import convert_from_path

# MIN_CONF = 0.30
# reader = easyocr.Reader(['en'], gpu=False)
# input_path = sys.argv[1]

# # ── Load ──────────────────────────────────────────────────────────────────────
# if input_path.lower().endswith(".pdf"):
#     pages = convert_from_path(input_path, dpi=500)
# else:
#     pages = [cv2.imread(input_path)]

# elements = []

# def preprocess(img):
#     if not isinstance(img, np.ndarray):
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
#     _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#     return th

# for page in pages:
#     image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
#     for (bbox, text, prob) in reader.readtext(preprocess(image)):
#         if prob < MIN_CONF: continue
#         text = text.upper()
#         x = int(sum([p[0] for p in bbox]) / 4)
#         y = int(sum([p[1] for p in bbox]) / 4)
#         elements.append({"text": text, "x": x, "y": y})

# # ── Clean ─────────────────────────────────────────────────────────────────────
# def clean(t):
#     t = t.upper()
#     t = t.replace("-", "=").replace("_", "")
#     t = t.replace("H-", "H=").replace("T-", "T=")
#     t = t.replace("D-", "D=").replace("W-", "W=")
#     t = t.replace("1W", "IW").replace("lW", "IW").replace("|W", "IW")
#     # Fix EWA→EW4, EWB→EW2 (digit/letter confusion in EW numbered labels)
#     letter_to_digit = {'A':'4','B':'2','C':'3','S':'5','G':'6','I':'1','O':'0'}
#     t = re.sub(r'^EW([A-Za-z])$', lambda m: 'EW' + letter_to_digit.get(m.group(1).upper(), m.group(1)), t)
#     return t.strip()

# for e in elements:
#     e["text"] = clean(e["text"])

# # ── Merge "ZONE" + "1" split tokens ─────────────────────────────────
# import math as _math
# merged_zones = []
# used_idx = set()
# sorted_elems = sorted(range(len(elements)), key=lambda i: (round(elements[i]["y"]/30), elements[i]["x"]))
# for ii, i in enumerate(sorted_elems):
#     if i in used_idx: continue
#     e = elements[i]
#     if e["text"] == "ZONE":
#         # Look for a nearby digit token
#         for jj, j in enumerate(sorted_elems[ii+1:ii+6], 1):
#             if j in used_idx: continue
#             ne = elements[j]
#             dx = abs(ne["x"] - e["x"])
#             dy = abs(ne["y"] - e["y"])
#             if dy < 40 and dx < 250 and re.match(r"^\d[A-Z]?$", ne["text"]):
#                 elements[i]["text"] = "ZONE " + ne["text"]
#                 used_idx.add(j)
#                 break
# # Remove merged duplicates
# elements = [e for i,e in enumerate(elements) if i not in used_idx]

# # ── Helpers ───────────────────────────────────────────────────────────────────
# def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

# def ft_to_m(v):
#     """Convert feet to meters, rounded to 2 decimal places."""
#     if v is None: return None
#     return round(v * 0.3048, 2)

# def sqft_to_sqm(v):
#     """Convert square feet to square meters."""
#     if v is None: return None
#     return round(v * 0.0929, 2)

# def is_duplicate(x, y, positions, th=25):
#     return any(abs(x-px)<th and abs(y-py)<th for px,py in positions)

# def parse_feet_inches_fraction(text):
#     text = text.replace(" ", "")
#     fm = re.search(r"(\d+)'", text)
#     if not fm: return None
#     feet = int(fm.group(1))
#     im   = re.search(r"'(\d+)", text)
#     fracm= re.search(r"(\d+)/(\d+)", text)
#     inches = int(im.group(1)) if im else 0
#     frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
#     if inches > 12: inches = int(str(inches)[0])
#     return math.ceil(feet + (inches+frac)/12)

# def parse_size_from_elements(base):
#     best_match, best_dist = None, 999999
#     for e in elements:
#         d = dist(base, e)
#         if d < 700:
#             txt = e["text"]
#             m = re.search(r"(\d+'\d+(?:\s*\d+/\d+)?\"?)\s*[xX]\s*(\d+'\d+(?:\s*\d+/\d+)?\"?)", txt)
#             if m:
#                 w = parse_feet_inches_fraction(m.group(1))
#                 l = parse_feet_inches_fraction(m.group(2))
#                 if w and l and 5<=w<=50 and 5<=l<=50 and d<best_dist:
#                     best_match=(w,l); best_dist=d
#     if best_match:
#         w,l = best_match
#         if w>l: w,l=l,w
#         return {"length_ft":l,"width_ft":w}

#     best_match, best_area = None, 0
#     for e in elements:
#         d = dist(base, e)
#         if d < 700:
#             txt = e["text"]
#             if "D=" in txt or "W=" in txt: continue
#             m = re.search(r"(\d{1,2})\s*[xX]\s*(\d{1,2})", txt)
#             if m:
#                 w,l = int(m.group(1)), int(m.group(2))
#                 if 5<=w<=40 and 5<=l<=40 and w*l>best_area:
#                     best_match=(w,l); best_area=w*l
#     if best_match:
#         w,l = best_match
#         if w>l: w,l=l,w
#         return {"length_ft":l,"width_ft":w}
#     return None

# def nearest_room(base):
#     ignore = ["ZONE","T=","H=","D=","W=","NORTH","SOUTH","EAST","WEST"]
#     best, best_d = None, 999999
#     for e in elements:
#         txt = e["text"]
#         if any(i in txt for i in ignore): continue
#         d = dist(base, e)
#         if d < best_d: best=txt; best_d=d
#     return best

# def find_wall_specs(base):
#     t=h=None
#     for e in elements:
#         if dist(base,e)<200:
#             txt=e["text"]
#             if t is None:
#                 m=re.search(r"T=?\s*(\d{1,2})",txt)
#                 if m: t=int(m.group(1))
#             if h is None:
#                 m=re.search(r"H=?\s*(\d{1,2})",txt)
#                 if m: h=int(m.group(1))
#     return t, h

# img_h = 1000; img_w = 1000
# if elements:
#     img_w = max(e["x"] for e in elements) * 1.1
#     img_h = max(e["y"] for e in elements) * 1.1

# # ── EW Anchors ────────────────────────────────────────────────────────────────
# ew_anchors, ew_positions = [], []
# for e in elements:
#     txt = e["text"]
#     if re.match(r"^EW[\s,=T\d\.]*$|^EW\d+[A-Z]?$|^EW$", txt):
#         if not is_duplicate(e["x"],e["y"],ew_positions,40):
#             ew_anchors.append(dict(e))
#             ew_positions.append((e["x"],e["y"]))

# # ── IW Anchors ────────────────────────────────────────────────────────────────
# iw_anchors, iw_positions = [], []
# for e in elements:
#     txt = e["text"]
#     if re.match(r"^IW[\s,=T\d\.]*$|^IW\d+[A-Z]?$|^IW$|^IW[\s\(].*$", txt):
#         if not is_duplicate(e["x"],e["y"],iw_positions,40):
#             iw_anchors.append(dict(e))
#             iw_positions.append((e["x"],e["y"]))

# # ── Build EW/IW maps ──────────────────────────────────────────────────────────
# def pos_by_location(x, y):
#     m=0.15
#     if y<img_h*m: return "north"
#     elif y>img_h*(1-m): return "south"
#     elif x<img_w*m: return "west"
#     elif x>img_w*(1-m): return "east"
#     return "inner"

# def find_dir(base, r=220):
#     for e in elements:
#         if dist(base,e)<r:
#             for d in ["NORTH","SOUTH","EAST","WEST","MIDDLE"]:
#                 if d in e["text"]: return d.lower()
#     return None

# external_walls_map = {}
# for i, anchor in enumerate(ew_anchors):
#     eid = f"ew{i+1}"
#     tv, hv = find_wall_specs(anchor)
#     pos = find_dir(anchor) or pos_by_location(anchor["x"], anchor["y"])
#     external_walls_map[eid] = {
#         "id":eid,"position":pos,
#         "thickness_in":tv or 9,"height_ft":hv or 10,
#         "thickness_m": ft_to_m((tv or 9)/12),
#         "height_m":    ft_to_m(hv or 10),
#         "connected":{"windows":[],"doors":[],"internal_walls":[]}
#     }
#     anchor["id"] = eid

# internal_walls_map = {}
# for i, anchor in enumerate(iw_anchors):
#     iwid = f"iw{i+1}"
#     tv, hv = find_wall_specs(anchor)
#     pos = find_dir(anchor)
#     conn = []
#     for ea in sorted(ew_anchors, key=lambda a:dist(anchor,a))[:2]:
#         conn.append(ea["id"])
#         if iwid not in external_walls_map[ea["id"]]["connected"]["internal_walls"]:
#             external_walls_map[ea["id"]]["connected"]["internal_walls"].append(iwid)
#     internal_walls_map[iwid] = {
#         "id":iwid,"position":pos,
#         "thickness_in":tv or 4,"height_ft":hv or 10,
#         "thickness_m": ft_to_m((tv or 4)/12),
#         "height_m":    ft_to_m(hv or 10),
#         "connects_external":conn,
#         "connected":{"doors":[],"windows":[]}
#     }
#     anchor["id"] = iwid

# # ── Doors ─────────────────────────────────────────────────────────────────────
# door_sizes, door_positions_list = [], []
# door_list = []
# for e in elements:
#     txt=e["text"]
#     m=re.search(r"D=?(\d{1,2})x(\d{1,2})",txt)
#     if m:
#         w,h=int(m.group(1)),int(m.group(2))
#         if 2<=w<=5 and 6<=h<=8:
#             if not is_duplicate(e["x"],e["y"],door_positions_list):
#                 door_positions_list.append((e["x"],e["y"]))
#                 door_sizes.append((w,h))
#                 door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})

# # ── Windows ───────────────────────────────────────────────────────────────────
# window_sizes, window_positions_list = [], []
# window_list = []
# for e in elements:
#     txt=e["text"]
#     m=re.search(r"W=?(\d{1,2})x(\d{1,2})",txt)
#     if m:
#         w,h=int(m.group(1)),int(m.group(2))
#         if 2<=w<=8 and 2<=h<=8:
#             if not is_duplicate(e["x"],e["y"],window_positions_list):
#                 window_positions_list.append((e["x"],e["y"]))
#                 window_sizes.append((w,h))
#                 window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})

# # ── Assign on_wall + back-fill ────────────────────────────────────────────────
# def nearest_wall(pos):
#     best,bd=None,999999
#     for a in ew_anchors+iw_anchors:
#         d=dist(pos,a)
#         if d<bd: bd,best=d,a["id"]
#     return best

# doors_out={}
# for d in door_list:
#     ow=nearest_wall({"x":d["x"],"y":d["y"]})
#     doors_out[d["id"]]={"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],"width_m":ft_to_m(d["width_ft"]),"height_m":ft_to_m(d["height_ft"]),"on_wall":ow}
#     if ow:
#         wl=external_walls_map.get(ow) or internal_walls_map.get(ow)
#         if wl and d["id"] not in wl["connected"]["doors"]: wl["connected"]["doors"].append(d["id"])

# windows_out={}
# for w in window_list:
#     ow=nearest_wall({"x":w["x"],"y":w["y"]})
#     windows_out[w["id"]]={"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],"width_m":ft_to_m(w["width_ft"]),"height_m":ft_to_m(w["height_ft"]),"on_wall":ow}
#     if ow:
#         wl=external_walls_map.get(ow) or internal_walls_map.get(ow)
#         if wl and w["id"] not in wl["connected"]["windows"]: wl["connected"]["windows"].append(w["id"])

# # ── Zones ─────────────────────────────────────────────────────────────────────
# zone_pattern = r"ZONE\s*\d+[A-Z]?"
# zones_out = {}

# for e in elements:
#     if not re.match(zone_pattern, e["text"]): continue
#     zid  = re.sub(r"\s+","",e["text"])
#     name = nearest_room(e) or "UNKNOWN"
#     size = parse_size_from_elements(e)
#     area = round(size["length_ft"]*size["width_ft"],2) if size else None

#     # Zone radius — based on zone size
#     if size:
#         shorter = min(size["width_ft"],size["length_ft"])
#         px_ft   = img_w / 40.0
#         zp      = shorter * px_ft * 0.55
#     else:
#         zp = img_w * 0.20

#     conn_ew = list(dict.fromkeys(
#         a["id"] for a in ew_anchors if dist(e,a)<zp*1.3
#     ))
#     conn_iw = list(dict.fromkeys(
#         a["id"] for a in iw_anchors if dist(e,a)<zp*1.1
#     ))
#     conn_d  = list(dict.fromkeys(
#         d["id"] for d in door_list if dist(e,{"x":d["x"],"y":d["y"]})<zp
#     ))
#     conn_w  = list(dict.fromkeys(
#         w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp
#     ))

#     wft = size["width_ft"]  if size else None
#     lft = size["length_ft"] if size else None
#     zones_out[zid] = {
#         "id":    zid,
#         "label": name,
#         "zone_id": e["text"],
#         "size":  size,
#         # Imperial
#         "area_sqft": area,
#         "width_ft":  wft,
#         "length_ft": lft,
#         # Metric
#         "area_sqm":  sqft_to_sqm(area),
#         "width_m":   ft_to_m(wft),
#         "length_m":  ft_to_m(lft),
#         "connected_external_walls": conn_ew,
#         "connected_internal_walls": conn_iw,
#         "total_walls_connected":    len(conn_ew)+len(conn_iw),
#         "doors":   conn_d,
#         "windows": conn_w,
#     }

# # ── Legacy aggregation (kept for backward compat) ─────────────────────────────
# doors_agg = []
# for (w,h),c in Counter(door_sizes).items():
#     doors_agg.append({"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c})

# windows_agg = []
# for (w,h),c in Counter(window_sizes).items():
#     windows_agg.append({"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c})

# visited=[]
# iw_sizes_agg=[]; ew_sizes_agg=[]
# for e in elements:
#     txt=e["text"]
#     if any(dist(e,v)<120 for v in visited): continue
#     if "IW" in txt:
#         t,h=find_wall_specs(e)
#         if t and h and 3<=t<=12 and 6<=h<=12: iw_sizes_agg.append((t,h)); visited.append(e)
#     elif "EW" in txt:
#         t,h=find_wall_specs(e)
#         if t and h and 6<=t<=15 and 8<=h<=15: ew_sizes_agg.append((t,h)); visited.append(e)

# internal_walls_agg=[{"type":"internal","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(iw_sizes_agg).items()]
# external_walls_agg=[{"type":"external","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(ew_sizes_agg).items()]

# total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]),2)
# all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
# all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])

# # ── Output ────────────────────────────────────────────────────────────────────
# print(json.dumps({
#     "zones":    zones_out,
#     "external_walls": external_walls_map,
#     "internal_walls": internal_walls_map,
#     "doors":    doors_out,
#     "windows":  windows_out,
#     "openings": {"doors":doors_agg,"windows":windows_agg},
#     "walls":    {"internal":internal_walls_agg,"external":external_walls_agg},
#     "summary": {
#         "total_area_sqft":         total_area,
#         "total_zones":             len(zones_out),
#         "total_doors":             len(doors_out),
#         "total_windows":           len(windows_out),
#         "total_external_walls":    len(external_walls_map),
#         "total_internal_walls":    len(internal_walls_map),
#         "total_door_area_sqft":    sum(d["total_area_sqft"] for d in doors_agg),
#         "total_window_area_sqft":  sum(w["total_area_sqft"] for w in windows_agg),
#         "total_opening_area_sqft": sum(d["total_area_sqft"] for d in doors_agg)+sum(w["total_area_sqft"] for w in windows_agg),
#         "total_area_sqm":          sqft_to_sqm(total_area),
#         "total_door_area_sqm":     sqft_to_sqm(sum(d["total_area_sqft"] for d in doors_agg)),
#         "total_window_area_sqm":   sqft_to_sqm(sum(w["total_area_sqft"] for w in windows_agg)),
#         "unique_doors_in_zones":   len(all_zone_doors),
#         "unique_windows_in_zones": len(all_zone_windows),
#         "detection_strategy":      "easyocr_standard"
#     }
# }))




# # import cv2
# # import easyocr
# # import re
# # import json
# # import numpy as np
# # import math
# # import sys
# # from collections import Counter
# # from pdf2image import convert_from_path

# # MIN_CONF = 0.30
# # reader = easyocr.Reader(['en'], gpu=False)
# # input_path = sys.argv[1]

# # # ── Load ──────────────────────────────────────────────────────────────────────
# # if input_path.lower().endswith(".pdf"):
# #     pages = convert_from_path(input_path, dpi=500)
# # else:
# #     pages = [cv2.imread(input_path)]

# # elements = []

# # def preprocess(img):
# #     if not isinstance(img, np.ndarray):
# #         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.bilateralFilter(gray, 9, 75, 75)
# #     _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# #     return th

# # for page in pages:
# #     image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
# #     for (bbox, text, prob) in reader.readtext(preprocess(image)):
# #         if prob < MIN_CONF: continue
# #         text = text.upper()
# #         x = int(sum([p[0] for p in bbox]) / 4)
# #         y = int(sum([p[1] for p in bbox]) / 4)
# #         elements.append({"text": text, "x": x, "y": y})

# # # ── Clean ─────────────────────────────────────────────────────────────────────
# # def clean(t):
# #     t = t.upper()
# #     t = t.replace("-", "=").replace("_", "")
# #     t = t.replace("H-", "H=").replace("T-", "T=")
# #     t = t.replace("D-", "D=").replace("W-", "W=")
# #     t = t.replace("1W", "IW").replace("lW", "IW").replace("|W", "IW")
# #     return t.strip()

# # for e in elements:
# #     e["text"] = clean(e["text"])

# # # ── Helpers ───────────────────────────────────────────────────────────────────
# # def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

# # def ft_to_m(v):
# #     """Convert feet to meters, rounded to 2 decimal places."""
# #     if v is None: return None
# #     return round(v * 0.3048, 2)

# # def sqft_to_sqm(v):
# #     """Convert square feet to square meters."""
# #     if v is None: return None
# #     return round(v * 0.0929, 2)

# # def is_duplicate(x, y, positions, th=25):
# #     return any(abs(x-px)<th and abs(y-py)<th for px,py in positions)

# # def parse_feet_inches_fraction(text):
# #     text = text.replace(" ", "")
# #     fm = re.search(r"(\d+)'", text)
# #     if not fm: return None
# #     feet = int(fm.group(1))
# #     im   = re.search(r"'(\d+)", text)
# #     fracm= re.search(r"(\d+)/(\d+)", text)
# #     inches = int(im.group(1)) if im else 0
# #     frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
# #     if inches > 12: inches = int(str(inches)[0])
# #     return math.ceil(feet + (inches+frac)/12)

# # def parse_size_from_elements(base):
# #     best_match, best_dist = None, 999999
# #     for e in elements:
# #         d = dist(base, e)
# #         if d < 700:
# #             txt = e["text"]
# #             m = re.search(r"(\d+'\d+(?:\s*\d+/\d+)?\"?)\s*[xX]\s*(\d+'\d+(?:\s*\d+/\d+)?\"?)", txt)
# #             if m:
# #                 w = parse_feet_inches_fraction(m.group(1))
# #                 l = parse_feet_inches_fraction(m.group(2))
# #                 if w and l and 5<=w<=50 and 5<=l<=50 and d<best_dist:
# #                     best_match=(w,l); best_dist=d
# #     if best_match:
# #         w,l = best_match
# #         if w>l: w,l=l,w
# #         return {"length_ft":l,"width_ft":w}

# #     best_match, best_area = None, 0
# #     for e in elements:
# #         d = dist(base, e)
# #         if d < 700:
# #             txt = e["text"]
# #             if "D=" in txt or "W=" in txt: continue
# #             m = re.search(r"(\d{1,2})\s*[xX]\s*(\d{1,2})", txt)
# #             if m:
# #                 w,l = int(m.group(1)), int(m.group(2))
# #                 if 5<=w<=40 and 5<=l<=40 and w*l>best_area:
# #                     best_match=(w,l); best_area=w*l
# #     if best_match:
# #         w,l = best_match
# #         if w>l: w,l=l,w
# #         return {"length_ft":l,"width_ft":w}
# #     return None

# # def nearest_room(base):
# #     ignore = ["ZONE","T=","H=","D=","W=","NORTH","SOUTH","EAST","WEST"]
# #     best, best_d = None, 999999
# #     for e in elements:
# #         txt = e["text"]
# #         if any(i in txt for i in ignore): continue
# #         d = dist(base, e)
# #         if d < best_d: best=txt; best_d=d
# #     return best

# # def find_wall_specs(base):
# #     t=h=None
# #     for e in elements:
# #         if dist(base,e)<200:
# #             txt=e["text"]
# #             if t is None:
# #                 m=re.search(r"T=?\s*(\d{1,2})",txt)
# #                 if m: t=int(m.group(1))
# #             if h is None:
# #                 m=re.search(r"H=?\s*(\d{1,2})",txt)
# #                 if m: h=int(m.group(1))
# #     return t, h

# # img_h = 1000; img_w = 1000
# # if elements:
# #     img_w = max(e["x"] for e in elements) * 1.1
# #     img_h = max(e["y"] for e in elements) * 1.1

# # # ── EW Anchors ────────────────────────────────────────────────────────────────
# # ew_anchors, ew_positions = [], []
# # for e in elements:
# #     txt = e["text"]
# #     if re.match(r"^EW[\s,=T\d\.]*$|^EW\d+[A-Z]?$|^EW$", txt):
# #         if not is_duplicate(e["x"],e["y"],ew_positions,40):
# #             ew_anchors.append(dict(e))
# #             ew_positions.append((e["x"],e["y"]))

# # # ── IW Anchors ────────────────────────────────────────────────────────────────
# # iw_anchors, iw_positions = [], []
# # for e in elements:
# #     txt = e["text"]
# #     if re.match(r"^IW[\s,=T\d\.]*$|^IW\d+[A-Z]?$|^IW$|^IW[\s\(].*$", txt):
# #         if not is_duplicate(e["x"],e["y"],iw_positions,40):
# #             iw_anchors.append(dict(e))
# #             iw_positions.append((e["x"],e["y"]))

# # # ── Build EW/IW maps ──────────────────────────────────────────────────────────
# # def pos_by_location(x, y):
# #     m=0.15
# #     if y<img_h*m: return "north"
# #     elif y>img_h*(1-m): return "south"
# #     elif x<img_w*m: return "west"
# #     elif x>img_w*(1-m): return "east"
# #     return "inner"

# # def find_dir(base, r=220):
# #     for e in elements:
# #         if dist(base,e)<r:
# #             for d in ["NORTH","SOUTH","EAST","WEST","MIDDLE"]:
# #                 if d in e["text"]: return d.lower()
# #     return None

# # external_walls_map = {}
# # for i, anchor in enumerate(ew_anchors):
# #     eid = f"ew{i+1}"
# #     tv, hv = find_wall_specs(anchor)
# #     pos = find_dir(anchor) or pos_by_location(anchor["x"], anchor["y"])
# #     external_walls_map[eid] = {
# #         "id":eid,"position":pos,
# #         "thickness_in":tv or 9,"height_ft":hv or 10,
# #         "thickness_m": ft_to_m((tv or 9)/12),
# #         "height_m":    ft_to_m(hv or 10),
# #         "connected":{"windows":[],"doors":[],"internal_walls":[]}
# #     }
# #     anchor["id"] = eid

# # internal_walls_map = {}
# # for i, anchor in enumerate(iw_anchors):
# #     iwid = f"iw{i+1}"
# #     tv, hv = find_wall_specs(anchor)
# #     pos = find_dir(anchor)
# #     conn = []
# #     for ea in sorted(ew_anchors, key=lambda a:dist(anchor,a))[:2]:
# #         conn.append(ea["id"])
# #         if iwid not in external_walls_map[ea["id"]]["connected"]["internal_walls"]:
# #             external_walls_map[ea["id"]]["connected"]["internal_walls"].append(iwid)
# #     internal_walls_map[iwid] = {
# #         "id":iwid,"position":pos,
# #         "thickness_in":tv or 4,"height_ft":hv or 10,
# #         "thickness_m": ft_to_m((tv or 4)/12),
# #         "height_m":    ft_to_m(hv or 10),
# #         "connects_external":conn,
# #         "connected":{"doors":[],"windows":[]}
# #     }
# #     anchor["id"] = iwid

# # # ── Doors ─────────────────────────────────────────────────────────────────────
# # door_sizes, door_positions_list = [], []
# # door_list = []
# # for e in elements:
# #     txt=e["text"]
# #     m=re.search(r"D=?(\d{1,2})x(\d{1,2})",txt)
# #     if m:
# #         w,h=int(m.group(1)),int(m.group(2))
# #         if 2<=w<=5 and 6<=h<=8:
# #             if not is_duplicate(e["x"],e["y"],door_positions_list):
# #                 door_positions_list.append((e["x"],e["y"]))
# #                 door_sizes.append((w,h))
# #                 door_list.append({"id":f"d{len(door_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})

# # # ── Windows ───────────────────────────────────────────────────────────────────
# # window_sizes, window_positions_list = [], []
# # window_list = []
# # for e in elements:
# #     txt=e["text"]
# #     m=re.search(r"W=?(\d{1,2})x(\d{1,2})",txt)
# #     if m:
# #         w,h=int(m.group(1)),int(m.group(2))
# #         if 2<=w<=8 and 2<=h<=8:
# #             if not is_duplicate(e["x"],e["y"],window_positions_list):
# #                 window_positions_list.append((e["x"],e["y"]))
# #                 window_sizes.append((w,h))
# #                 window_list.append({"id":f"w{len(window_list)+1}","x":e["x"],"y":e["y"],"width_ft":w,"height_ft":h})

# # # ── Assign on_wall + back-fill ────────────────────────────────────────────────
# # def nearest_wall(pos):
# #     best,bd=None,999999
# #     for a in ew_anchors+iw_anchors:
# #         d=dist(pos,a)
# #         if d<bd: bd,best=d,a["id"]
# #     return best

# # doors_out={}
# # for d in door_list:
# #     ow=nearest_wall({"x":d["x"],"y":d["y"]})
# #     doors_out[d["id"]]={"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],"width_m":ft_to_m(d["width_ft"]),"height_m":ft_to_m(d["height_ft"]),"on_wall":ow}
# #     if ow:
# #         wl=external_walls_map.get(ow) or internal_walls_map.get(ow)
# #         if wl and d["id"] not in wl["connected"]["doors"]: wl["connected"]["doors"].append(d["id"])

# # windows_out={}
# # for w in window_list:
# #     ow=nearest_wall({"x":w["x"],"y":w["y"]})
# #     windows_out[w["id"]]={"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],"width_m":ft_to_m(w["width_ft"]),"height_m":ft_to_m(w["height_ft"]),"on_wall":ow}
# #     if ow:
# #         wl=external_walls_map.get(ow) or internal_walls_map.get(ow)
# #         if wl and w["id"] not in wl["connected"]["windows"]: wl["connected"]["windows"].append(w["id"])

# # # ── Zones ─────────────────────────────────────────────────────────────────────
# # zone_pattern = r"ZONE\s*\d+[A-Z]?"
# # zones_out = {}

# # for e in elements:
# #     if not re.match(zone_pattern, e["text"]): continue
# #     zid  = re.sub(r"\s+","",e["text"])
# #     name = nearest_room(e) or "UNKNOWN"
# #     size = parse_size_from_elements(e)
# #     area = round(size["length_ft"]*size["width_ft"],2) if size else None

# #     # Zone radius — based on zone size
# #     if size:
# #         shorter = min(size["width_ft"],size["length_ft"])
# #         px_ft   = img_w / 40.0
# #         zp      = shorter * px_ft * 0.55
# #     else:
# #         zp = img_w * 0.20

# #     conn_ew = list(dict.fromkeys(
# #         a["id"] for a in ew_anchors if dist(e,a)<zp*1.3
# #     ))
# #     conn_iw = list(dict.fromkeys(
# #         a["id"] for a in iw_anchors if dist(e,a)<zp*1.1
# #     ))
# #     conn_d  = list(dict.fromkeys(
# #         d["id"] for d in door_list if dist(e,{"x":d["x"],"y":d["y"]})<zp
# #     ))
# #     conn_w  = list(dict.fromkeys(
# #         w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<zp
# #     ))

# #     wft = size["width_ft"]  if size else None
# #     lft = size["length_ft"] if size else None
# #     zones_out[zid] = {
# #         "id":    zid,
# #         "label": name,
# #         "zone_id": e["text"],
# #         "size":  size,
# #         # Imperial
# #         "area_sqft": area,
# #         "width_ft":  wft,
# #         "length_ft": lft,
# #         # Metric
# #         "area_sqm":  sqft_to_sqm(area),
# #         "width_m":   ft_to_m(wft),
# #         "length_m":  ft_to_m(lft),
# #         "connected_external_walls": conn_ew,
# #         "connected_internal_walls": conn_iw,
# #         "total_walls_connected":    len(conn_ew)+len(conn_iw),
# #         "doors":   conn_d,
# #         "windows": conn_w,
# #     }

# # # ── Legacy aggregation (kept for backward compat) ─────────────────────────────
# # doors_agg = []
# # for (w,h),c in Counter(door_sizes).items():
# #     doors_agg.append({"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c})

# # windows_agg = []
# # for (w,h),c in Counter(window_sizes).items():
# #     windows_agg.append({"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c})

# # visited=[]
# # iw_sizes_agg=[]; ew_sizes_agg=[]
# # for e in elements:
# #     txt=e["text"]
# #     if any(dist(e,v)<120 for v in visited): continue
# #     if "IW" in txt:
# #         t,h=find_wall_specs(e)
# #         if t and h and 3<=t<=12 and 6<=h<=12: iw_sizes_agg.append((t,h)); visited.append(e)
# #     elif "EW" in txt:
# #         t,h=find_wall_specs(e)
# #         if t and h and 6<=t<=15 and 8<=h<=15: ew_sizes_agg.append((t,h)); visited.append(e)

# # internal_walls_agg=[{"type":"internal","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(iw_sizes_agg).items()]
# # external_walls_agg=[{"type":"external","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(ew_sizes_agg).items()]

# # total_area = round(sum(z["area_sqft"] for z in zones_out.values() if z["area_sqft"]),2)
# # all_zone_doors   = set(d for z in zones_out.values() for d in z["doors"])
# # all_zone_windows = set(w for z in zones_out.values() for w in z["windows"])

# # # ── Output ────────────────────────────────────────────────────────────────────
# # print(json.dumps({
# #     "zones":    zones_out,
# #     "external_walls": external_walls_map,
# #     "internal_walls": internal_walls_map,
# #     "doors":    doors_out,
# #     "windows":  windows_out,
# #     "openings": {"doors":doors_agg,"windows":windows_agg},
# #     "walls":    {"internal":internal_walls_agg,"external":external_walls_agg},
# #     "summary": {
# #         "total_area_sqft":         total_area,
# #         "total_zones":             len(zones_out),
# #         "total_doors":             len(doors_out),
# #         "total_windows":           len(windows_out),
# #         "total_external_walls":    len(external_walls_map),
# #         "total_internal_walls":    len(internal_walls_map),
# #         "total_door_area_sqft":    sum(d["total_area_sqft"] for d in doors_agg),
# #         "total_window_area_sqft":  sum(w["total_area_sqft"] for w in windows_agg),
# #         "total_opening_area_sqft": sum(d["total_area_sqft"] for d in doors_agg)+sum(w["total_area_sqft"] for w in windows_agg),
# #         "total_area_sqm":          sqft_to_sqm(total_area),
# #         "total_door_area_sqm":     sqft_to_sqm(sum(d["total_area_sqft"] for d in doors_agg)),
# #         "total_window_area_sqm":   sqft_to_sqm(sum(w["total_area_sqft"] for w in windows_agg)),
# #         "unique_doors_in_zones":   len(all_zone_doors),
# #         "unique_windows_in_zones": len(all_zone_windows),
# #         "detection_strategy":      "easyocr_standard"
# #     }
# # }))
