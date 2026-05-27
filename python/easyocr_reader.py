import cv2
import easyocr
import re
import json
import numpy as np
import math
import sys
from collections import defaultdict
from pdf2image import convert_from_path

# ================= SETTINGS =================
MIN_CONF = 0.30
reader = easyocr.Reader(['en'], gpu=False)
input_path = sys.argv[1]

# ================= LOAD IMAGE / PDF =================
if input_path.lower().endswith(".pdf"):
    pages = convert_from_path(input_path, dpi=500)
else:
    img = cv2.imread(input_path)
    pages = [img] if img is not None else []

elements = []

# ================= PREPROCESS =================
def preprocess(img):
    if isinstance(img, np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    else:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return th

# ================= OCR =================
for page in pages:
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) if not isinstance(page, np.ndarray) else page
    processed = preprocess(image)
    results = reader.readtext(processed)
    for (bbox, text, prob) in results:
        if prob < MIN_CONF:
            continue
        text = text.upper().strip()
        x = int(sum([p[0] for p in bbox]) / 4)
        y = int(sum([p[1] for p in bbox]) / 4)
        elements.append({"text": text, "x": x, "y": y})

# ================= CLEAN TEXT =================
def clean(t):
    t = t.upper()
    t = t.replace("-", "=").replace("_", "")
    t = t.replace("H-", "H=").replace("T-", "T=")
    t = t.replace("D-", "D=").replace("W-", "W=")
    t = t.replace("1W", "IW").replace("lW", "IW").replace("lw", "IW")
    return t.strip()

for e in elements:
    e["text"] = clean(e["text"])

# ================= HELPERS =================
def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

def is_duplicate(x, y, positions, th=40):
    return any(abs(x - px) < th and abs(y - py) < th for px, py in positions)

def get_nearby(base, radius=200):
    return [e for e in elements if dist(base, e) < radius]

def find_val(nearby, pattern):
    for e in nearby:
        m = re.search(pattern, e["text"])
        if m:
            return int(m.group(1))
    return None

def parse_feet_inches(text):
    text = text.replace(" ", "")
    feet_m = re.search(r"(\d+)'", text)
    if not feet_m:
        return None
    feet = int(feet_m.group(1))
    inch_m = re.search(r"'(\d+)", text)
    frac_m = re.search(r"(\d+)/(\d+)", text)
    inches = int(inch_m.group(1)) if inch_m else 0
    frac = int(frac_m.group(1)) / int(frac_m.group(2)) if frac_m else 0
    if inches > 12:
        inches = int(str(inches)[0])
    return math.ceil(feet + (inches + frac) / 12)

# ================= STEP 1: EXTERNAL WALLS =================
ew_anchors, ew_positions = [], []
for e in elements:
    txt = e["text"]
    if re.match(r"^EW[\s,.<>:=\-]?\d+$", txt) or re.match(r"^EW\d+$", txt):
        if not is_duplicate(e["x"], e["y"], ew_positions):
            ew_anchors.append(e)
            ew_positions.append((e["x"], e["y"]))

external_walls = {}
for i, anchor in enumerate(ew_anchors):
    eid = f"ew{i+1}"
    nearby = get_nearby(anchor, 250)
    t_val = find_val(nearby, r"T[=\s](\d+)")
    h_val = find_val(nearby, r"H[=\s](\d+)")
    l_val = find_val(nearby, r"L[=\s](\d+)")
    external_walls[eid] = {
        "id": eid,
        "thickness_in": t_val,
        "height_ft": h_val,
        "length_ft": l_val,
        "connected": {
            "windows": [],
            "doors": [],
            "internal_walls": []
        }
    }

# ================= STEP 2: INTERNAL WALLS =================
iw_anchors, iw_positions = [], []
for e in elements:
    txt = e["text"]
    if re.match(r"^IW[\s,.<>:=\-]?\d+$", txt) or re.match(r"^IW\d+$", txt):
        if not is_duplicate(e["x"], e["y"], iw_positions):
            iw_anchors.append(e)
            iw_positions.append((e["x"], e["y"]))

internal_walls = {}
for i, anchor in enumerate(iw_anchors):
    iwid = f"iw{i+1}"
    nearby = get_nearby(anchor, 250)
    t_val = find_val(nearby, r"T[=\s](\d+)")
    h_val = find_val(nearby, r"H[=\s](\d+)")
    l_val = find_val(nearby, r"L[=\s](\d+)")
    connects_external = []
    for k, ew_anchor in enumerate(ew_anchors):
        if dist(anchor, ew_anchor) < 600:
            connects_external.append(f"ew{k+1}")
    internal_walls[iwid] = {
        "id": iwid,
        "thickness_in": t_val,
        "height_ft": h_val,
        "length_ft": l_val,
        "connects_external": connects_external,
        "connected": {
            "doors": [],
            "windows": []
        }
    }

# ================= STEP 3: DOORS =================
door_list = []
door_positions = []
for e in elements:
    txt = e["text"]
    m = re.match(r"D[=\s]?(\d+)[xX](\d+)", txt)
    if m:
        if not is_duplicate(e["x"], e["y"], door_positions):
            door_list.append({
                "id": f"d{len(door_list)+1}",
                "x": e["x"], "y": e["y"],
                "width_ft": int(m.group(1)),
                "height_ft": int(m.group(2))
            })
            door_positions.append((e["x"], e["y"]))
    elif re.match(r"^D\d+$", txt):
        nearby = get_nearby(e, 150)
        w = find_val(nearby, r"(\d+)[xX]\d+") or 3
        h = find_val(nearby, r"\d+[xX](\d+)") or 7
        if not is_duplicate(e["x"], e["y"], door_positions):
            door_list.append({
                "id": f"d{len(door_list)+1}",
                "x": e["x"], "y": e["y"],
                "width_ft": w,
                "height_ft": h
            })
            door_positions.append((e["x"], e["y"]))

# ================= STEP 4: WINDOWS =================
window_list = []
window_positions = []
for e in elements:
    txt = e["text"]
    m = re.match(r"W[=\s]?(\d+)[xX](\d+)", txt)
    if m:
        if not is_duplicate(e["x"], e["y"], window_positions):
            window_list.append({
                "id": f"w{len(window_list)+1}",
                "x": e["x"], "y": e["y"],
                "width_ft": int(m.group(1)),
                "height_ft": int(m.group(2))
            })
            window_positions.append((e["x"], e["y"]))
    elif re.match(r"^W\d+$", txt):
        nearby = get_nearby(e, 150)
        w = find_val(nearby, r"(\d+)[xX]\d+") or 4
        h = find_val(nearby, r"\d+[xX](\d+)") or 4
        if not is_duplicate(e["x"], e["y"], window_positions):
            window_list.append({
                "id": f"w{len(window_list)+1}",
                "x": e["x"], "y": e["y"],
                "width_ft": w,
                "height_ft": h
            })
            window_positions.append((e["x"], e["y"]))

# ================= STEP 5: ASSIGN on_wall + BACK-FILL =================
def find_nearest_wall(pos):
    best_wall = None
    best_d = 999999
    for k, ew_anchor in enumerate(ew_anchors):
        dd = dist(pos, ew_anchor)
        if dd < best_d:
            best_d = dd
            best_wall = f"ew{k+1}"
    for j, iw_anchor in enumerate(iw_anchors):
        dd = dist(pos, iw_anchor)
        if dd < best_d:
            best_d = dd
            best_wall = f"iw{j+1}"
    return best_wall

doors_output = {}
for d in door_list:
    on_wall = find_nearest_wall({"x": d["x"], "y": d["y"]})
    doors_output[d["id"]] = {
        "id": d["id"],
        "width_ft": d["width_ft"],
        "height_ft": d["height_ft"],
        "on_wall": on_wall
    }
    if on_wall:
        wall = external_walls.get(on_wall) or internal_walls.get(on_wall)
        if wall and d["id"] not in wall["connected"]["doors"]:
            wall["connected"]["doors"].append(d["id"])

windows_output = {}
for w in window_list:
    on_wall = find_nearest_wall({"x": w["x"], "y": w["y"]})
    windows_output[w["id"]] = {
        "id": w["id"],
        "width_ft": w["width_ft"],
        "height_ft": w["height_ft"],
        "on_wall": on_wall
    }
    if on_wall:
        wall = external_walls.get(on_wall) or internal_walls.get(on_wall)
        if wall and w["id"] not in wall["connected"]["windows"]:
            wall["connected"]["windows"].append(w["id"])

# Back-fill iw → ew connected.internal_walls
for j, iw_anchor in enumerate(iw_anchors):
    iwid = f"iw{j+1}"
    for k, ew_anchor in enumerate(ew_anchors):
        ewid = f"ew{k+1}"
        if dist(iw_anchor, ew_anchor) < 600:
            if iwid not in external_walls[ewid]["connected"]["internal_walls"]:
                external_walls[ewid]["connected"]["internal_walls"].append(iwid)

# ================= STEP 6: ZONES =================
zone_pattern = r"^ZONE\s*\d+[A-Z]?$"
room_keywords = ["BEDROOM","BATHROOM","KITCHEN","LIVING","DINING","POOJA",
                 "HALL","BALCONY","STORE","TOILET","LOBBY","STUDY","MBR",
                 "M.BEDROOM","C.BEDROOM","MASTER"]
zones_output = {}

def nearest_room_name(base, radius=300):
    for e in sorted(elements, key=lambda e: dist(base, e)):
        if dist(base, e) > radius:
            break
        for kw in room_keywords:
            if kw in e["text"]:
                return e["text"]
    return "UNKNOWN"

def parse_zone_size(base, radius=400):
    for e in elements:
        if dist(base, e) < radius:
            m = re.search(r"(\d+'\d+(?:\s*\d+/\d+)?\"?)\s*[xX]\s*(\d+'\d+(?:\s*\d+/\d+)?\"?)", e["text"])
            if m:
                ww = parse_feet_inches(m.group(1))
                ll = parse_feet_inches(m.group(2))
                if ww and ll and 5 <= ww <= 50 and 5 <= ll <= 50:
                    return min(ww, ll), max(ww, ll)
            m2 = re.search(r"(\d{1,2})\s*[xX]\s*(\d{1,2})", e["text"])
            if m2:
                ww, ll = int(m2.group(1)), int(m2.group(2))
                if 5 <= ww <= 40 and 5 <= ll <= 40:
                    return min(ww, ll), max(ww, ll)
    return None, None

for e in elements:
    if re.match(zone_pattern, e["text"]):
        zid = re.sub(r"\s+", "", e["text"])
        room_name = nearest_room_name(e)
        width_ft, length_ft = parse_zone_size(e)
        area = round(width_ft * length_ft, 2) if width_ft and length_ft else None

        connected_ew = list(dict.fromkeys(
            f"ew{k+1}" for k, ew_anchor in enumerate(ew_anchors)
            if dist(e, ew_anchor) < 500
        ))
        connected_iw = list(dict.fromkeys(
            f"iw{j+1}" for j, iw_anchor in enumerate(iw_anchors)
            if dist(e, iw_anchor) < 400
        ))
        connected_doors = list(dict.fromkeys(
            d["id"] for d in door_list
            if dist(e, {"x": d["x"], "y": d["y"]}) < 400
        ))
        connected_windows = list(dict.fromkeys(
            w["id"] for w in window_list
            if dist(e, {"x": w["x"], "y": w["y"]}) < 400
        ))

        zones_output[zid] = {
            "id": zid,
            "label": room_name,
            "width_ft": width_ft,
            "length_ft": length_ft,
            "area_sqft": area,
            "connected_external_walls": connected_ew,
            "connected_internal_walls": connected_iw,
            "total_walls_connected": len(connected_ew) + len(connected_iw),
            "doors": connected_doors,
            "windows": connected_windows
        }

# ================= FINAL OUTPUT =================
final = {
    "external_walls": external_walls,
    "internal_walls": internal_walls,
    "doors": doors_output,
    "windows": windows_output,
    "zones": zones_output,
    "summary": {
        "total_external_walls": len(external_walls),
        "total_internal_walls": len(internal_walls),
        "total_doors": len(doors_output),
        "total_windows": len(windows_output),
        "total_zones": len(zones_output),
        "total_area_sqft": round(
            sum(z["area_sqft"] for z in zones_output.values() if z["area_sqft"]), 2
        )
    }
}

print(json.dumps(final)) 
