import cv2
import easyocr
import re
import json
import numpy as np
import math
import sys
from pdf2image import convert_from_path

MIN_CONF = 0.30
reader   = easyocr.Reader(['en'], gpu=False)
input_path = sys.argv[1]

# ── Load image ────────────────────────────────────────────────────────────────
if input_path.lower().endswith(".pdf"):
    pages = convert_from_path(input_path, dpi=300)
else:
    img   = cv2.imread(input_path)
    pages = [img] if img is not None else []

elements = []

def preprocess(img):
    if isinstance(img, np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    else:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return th

raw_images = []
for page in pages:
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
    raw_images.append(image)
    processed = preprocess(image)
    results   = reader.readtext(processed)
    for (bbox, text, prob) in results:
        if prob < MIN_CONF: continue
        text = text.upper().strip()
        x = int(sum([p[0] for p in bbox]) / 4)
        y = int(sum([p[1] for p in bbox]) / 4)
        elements.append({"text": text, "x": x, "y": y})

def clean(t):
    t = t.upper().replace("-","=").replace("_","")
    t = t.replace("1W","IW").replace("lW","IW").replace("lw","IW")
    return t.strip()

for e in elements: e["text"] = clean(e["text"])

def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
def is_dup(x, y, pos, th=40): return any(abs(x-px)<th and abs(y-py)<th for px,py in pos)
def get_nearby(base, r=200):  return [e for e in elements if dist(base,e)<r]

def find_val(nearby, pat):
    for e in nearby:
        m = re.search(pat, e["text"])
        if m: return int(m.group(1))
    return None

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
    return math.ceil(feet + (inches+frac)/12)

# ── STEP 1: Detect walls via OpenCV line detection ────────────────────────────
def detect_walls_opencv(image):
    """Use HoughLinesP to detect horizontal/vertical lines = walls."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                             minLineLength=image.shape[1]//10, maxLineGap=20)
    h_lines, v_lines = [], []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            length = math.hypot(x2-x1, y2-y1)
            if angle < 15 or angle > 165:          # horizontal
                h_lines.append((min(x1,x2), max(x1,x2), (y1+y2)//2, length))
            elif 75 < angle < 105:                 # vertical
                v_lines.append(((x1+x2)//2, min(y1,y2), max(y1,y2), length))
    return h_lines, v_lines

def cluster_lines(lines, axis='h', gap=30):
    """Merge nearby parallel lines into single wall lines."""
    if not lines: return []
    lines = sorted(lines, key=lambda l: l[2] if axis=='h' else l[0])
    clusters = [lines[0]]
    for line in lines[1:]:
        coord = line[2] if axis=='h' else line[0]
        prev  = clusters[-1][2] if axis=='h' else clusters[-1][0]
        if abs(coord - prev) < gap:
            # merge — keep the longer one
            clusters[-1] = line if line[3] > clusters[-1][3] else clusters[-1]
        else:
            clusters.append(line)
    return clusters

# ── STEP 2: Auto-generate EW/IW from detected lines ──────────────────────────
external_walls = {}
internal_walls = {}
ew_anchors     = []
iw_anchors     = []

if raw_images:
    img_h, img_w = raw_images[0].shape[:2]
    h_lines, v_lines = detect_walls_opencv(raw_images[0])
    h_clusters = cluster_lines(h_lines, 'h')
    v_clusters = cluster_lines(v_lines, 'v')

    # Sort: outermost lines → EW, inner lines → IW
    h_sorted = sorted(h_clusters, key=lambda l: l[2])         # top to bottom
    v_sorted = sorted(v_clusters, key=lambda l: l[0])         # left to right

    boundary_margin = img_h * 0.12   # within 12% of edge = external

    def is_boundary_h(y): return y < boundary_margin or y > img_h - boundary_margin
    def is_boundary_v(x): return x < img_w*0.12 or x > img_w*0.88

    ew_count = 0
    iw_count = 0

    # Horizontal lines
    for i, (x1, x2, y, length) in enumerate(h_sorted):
        length_ft = round(length / (img_w / 30))   # rough scale: image_width ≈ 30ft
        height_ft = 10
        if is_boundary_h(y):
            ew_count += 1
            eid = f"ew{ew_count}"
            pos = "north" if y < img_h/2 else "south"
            external_walls[eid] = {
                "id": eid, "position": pos,
                "length_ft": length_ft, "thickness_in": 9, "height_ft": height_ft,
                "connected": {"windows": [], "doors": [], "internal_walls": []}
            }
            ew_anchors.append({"x": (x1+x2)//2, "y": y, "id": eid})
        else:
            iw_count += 1
            iwid = f"iw{iw_count}"
            internal_walls[iwid] = {
                "id": iwid, "length_ft": length_ft, "thickness_in": 4, "height_ft": height_ft,
                "connects_external": [],
                "connected": {"doors": [], "windows": []}
            }
            iw_anchors.append({"x": (x1+x2)//2, "y": y, "id": iwid})

    # Vertical lines
    for i, (x, y1, y2, length) in enumerate(v_sorted):
        length_ft = round(length / (img_h / 30))
        height_ft = 10
        if is_boundary_v(x):
            ew_count += 1
            eid = f"ew{ew_count}"
            pos = "west" if x < img_w/2 else "east"
            external_walls[eid] = {
                "id": eid, "position": pos,
                "length_ft": length_ft, "thickness_in": 9, "height_ft": height_ft,
                "connected": {"windows": [], "doors": [], "internal_walls": []}
            }
            ew_anchors.append({"x": x, "y": (y1+y2)//2, "id": eid})
        else:
            iw_count += 1
            iwid = f"iw{iw_count}"
            internal_walls[iwid] = {
                "id": iwid, "length_ft": length_ft, "thickness_in": 4, "height_ft": height_ft,
                "connects_external": [],
                "connected": {"doors": [], "windows": []}
            }
            iw_anchors.append({"x": x, "y": (y1+y2)//2, "id": iwid})

    # If OpenCV found nothing (simple PDF), create 4 default EWs
    if ew_count == 0:
        for pos, eid in [("north","ew1"),("east","ew2"),("south","ew3"),("west","ew4")]:
            external_walls[eid] = {
                "id": eid, "position": pos,
                "length_ft": None, "thickness_in": 9, "height_ft": 10,
                "connected": {"windows": [], "doors": [], "internal_walls": []}
            }
        ew_anchors = [
            {"x": img_w//2, "y": 10,          "id": "ew1"},
            {"x": img_w-10, "y": img_h//2,    "id": "ew2"},
            {"x": img_w//2, "y": img_h-10,    "id": "ew3"},
            {"x": 10,       "y": img_h//2,    "id": "ew4"},
        ]

    # Back-fill IW connects_external
    for iw_anchor in iw_anchors:
        iwid = iw_anchor["id"]
        for ew_anchor in ew_anchors:
            if dist(iw_anchor, ew_anchor) < max(img_w, img_h) * 0.6:
                if ew_anchor["id"] not in internal_walls[iwid]["connects_external"]:
                    internal_walls[iwid]["connects_external"].append(ew_anchor["id"])
            # Back-fill ew connected.internal_walls
            if dist(iw_anchor, ew_anchor) < max(img_w, img_h) * 0.6:
                if iwid not in external_walls[ew_anchor["id"]]["connected"]["internal_walls"]:
                    external_walls[ew_anchor["id"]]["connected"]["internal_walls"].append(iwid)

# ── STEP 3: Doors ─────────────────────────────────────────────────────────────
door_list, door_positions = [], []
for e in elements:
    txt = e["text"]
    m = re.match(r"D[=\s]?(\d+)[xX](\d+)", txt)
    if m:
        if not is_dup(e["x"], e["y"], door_positions):
            door_list.append({"id":f"d{len(door_list)+1}", "x":e["x"], "y":e["y"],
                               "width_ft":int(m.group(1)), "height_ft":int(m.group(2))})
            door_positions.append((e["x"], e["y"]))
    elif re.match(r"^D\d+$", txt):
        nb = get_nearby(e, 150)
        w  = find_val(nb, r"(\d+)[xX]\d+") or 3
        h  = find_val(nb, r"\d+[xX](\d+)") or 7
        if not is_dup(e["x"], e["y"], door_positions):
            door_list.append({"id":f"d{len(door_list)+1}", "x":e["x"], "y":e["y"],
                               "width_ft":w, "height_ft":h})
            door_positions.append((e["x"], e["y"]))

# ── STEP 4: Windows ───────────────────────────────────────────────────────────
window_list, window_positions = [], []
for e in elements:
    txt = e["text"]
    m = re.match(r"W[=\s]?(\d+)[xX](\d+)", txt)
    if m:
        if not is_dup(e["x"], e["y"], window_positions):
            window_list.append({"id":f"w{len(window_list)+1}", "x":e["x"], "y":e["y"],
                                 "width_ft":int(m.group(1)), "height_ft":int(m.group(2))})
            window_positions.append((e["x"], e["y"]))
    elif re.match(r"^W\d+$", txt):
        nb = get_nearby(e, 150)
        w  = find_val(nb, r"(\d+)[xX]\d+") or 4
        h  = find_val(nb, r"\d+[xX](\d+)") or 4
        if not is_dup(e["x"], e["y"], window_positions):
            window_list.append({"id":f"w{len(window_list)+1}", "x":e["x"], "y":e["y"],
                                 "width_ft":w, "height_ft":h})
            window_positions.append((e["x"], e["y"]))

# ── STEP 5: Assign on_wall to doors/windows ───────────────────────────────────
def find_nearest_wall(pos):
    best, best_d = None, 999999
    for a in ew_anchors:
        d = dist(pos, a)
        if d < best_d: best_d, best = d, a["id"]
    for a in iw_anchors:
        d = dist(pos, a)
        if d < best_d: best_d, best = d, a["id"]
    return best

doors_output = {}
for d in door_list:
    on_wall = find_nearest_wall({"x":d["x"],"y":d["y"]})
    doors_output[d["id"]] = {"id":d["id"],"width_ft":d["width_ft"],"height_ft":d["height_ft"],"on_wall":on_wall}
    if on_wall:
        wall = external_walls.get(on_wall) or internal_walls.get(on_wall)
        if wall and d["id"] not in wall["connected"]["doors"]:
            wall["connected"]["doors"].append(d["id"])

windows_output = {}
for w in window_list:
    on_wall = find_nearest_wall({"x":w["x"],"y":w["y"]})
    windows_output[w["id"]] = {"id":w["id"],"width_ft":w["width_ft"],"height_ft":w["height_ft"],"on_wall":on_wall}
    if on_wall:
        wall = external_walls.get(on_wall) or internal_walls.get(on_wall)
        if wall and w["id"] not in wall["connected"]["windows"]:
            wall["connected"]["windows"].append(w["id"])

# ── STEP 6: Zones ─────────────────────────────────────────────────────────────
zone_pattern  = r"^ZONE\s*\d+[A-Z]?$"
room_keywords = ["BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA",
                 "HALL","BALCONY","STORE","TOILET","LOBBY","STUDY","MBR",
                 "M.BEDROOM","C.BEDROOM","MASTER"]
zones_output  = {}

def nearest_room_name(base, radius=300):
    for e in sorted(elements, key=lambda e: dist(base,e)):
        if dist(base,e) > radius: break
        for kw in room_keywords:
            if kw in e["text"]: return e["text"]
    return "UNKNOWN"

def parse_zone_size(base, radius=400):
    for e in elements:
        if dist(base,e) < radius:
            m = re.search(r"(\d+'\d+(?:\s*\d+/\d+)?\"?)\s*[xX]\s*(\d+'\d+(?:\s*\d+/\d+)?\"?)", e["text"])
            if m:
                ww = parse_feet_inches(m.group(1))
                ll = parse_feet_inches(m.group(2))
                if ww and ll and 5<=ww<=50 and 5<=ll<=50: return min(ww,ll), max(ww,ll)
            m2 = re.search(r"(\d{1,2})\s*[xX]\s*(\d{1,2})", e["text"])
            if m2:
                ww, ll = int(m2.group(1)), int(m2.group(2))
                if 5<=ww<=40 and 5<=ll<=40: return min(ww,ll), max(ww,ll)
    return None, None

for e in elements:
    if re.match(zone_pattern, e["text"]):
        zid       = re.sub(r"\s+","", e["text"])
        room_name = nearest_room_name(e)
        wft, lft  = parse_zone_size(e)
        area      = round(wft*lft, 2) if wft and lft else None

        conn_ew = list(dict.fromkeys(
            a["id"] for a in ew_anchors if dist(e, a) < (raw_images[0].shape[1]*0.6 if raw_images else 500)
        ))
        conn_iw = list(dict.fromkeys(
            a["id"] for a in iw_anchors if dist(e, a) < (raw_images[0].shape[1]*0.4 if raw_images else 400)
        ))
        conn_doors = list(dict.fromkeys(
            d["id"] for d in door_list if dist(e,{"x":d["x"],"y":d["y"]})<400
        ))
        conn_wins = list(dict.fromkeys(
            w["id"] for w in window_list if dist(e,{"x":w["x"],"y":w["y"]})<400
        ))

        zones_output[zid] = {
            "id": zid, "label": room_name,
            "width_ft": wft, "length_ft": lft, "area_sqft": area,
            "connected_external_walls": conn_ew,
            "connected_internal_walls": conn_iw,
            "total_walls_connected": len(conn_ew)+len(conn_iw),
            "doors": conn_doors,
            "windows": conn_wins
        }

# ── FINAL OUTPUT ──────────────────────────────────────────────────────────────
final = {
    "external_walls": external_walls,
    "internal_walls": internal_walls,
    "doors":    doors_output,
    "windows":  windows_output,
    "zones":    zones_output,
    "summary": {
        "total_external_walls": len(external_walls),
        "total_internal_walls": len(internal_walls),
        "total_doors":          len(doors_output),
        "total_windows":        len(windows_output),
        "total_zones":          len(zones_output),
        "total_area_sqft":      round(sum(z["area_sqft"] for z in zones_output.values() if z["area_sqft"]), 2)
    }
}

print(json.dumps(final))
