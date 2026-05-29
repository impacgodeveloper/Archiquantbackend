import cv2
import easyocr
import re
import json
import numpy as np
import math
import sys
from collections import Counter
from pdf2image import convert_from_path

MIN_CONF = 0.30
reader = easyocr.Reader(['en'], gpu=False)
input_path = sys.argv[1]

if input_path.lower().endswith(".pdf"):
    pages = convert_from_path(input_path, dpi=500)
else:
    pages = [cv2.imread(input_path)]

elements = []

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return th

for page in pages:
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    for (bbox, text, prob) in reader.readtext(preprocess(image)):
        if prob < MIN_CONF: continue
        x = int(sum([p[0] for p in bbox]) / 4)
        y = int(sum([p[1] for p in bbox]) / 4)
        elements.append({"text": text.upper(), "x": x, "y": y})

def clean(t):
    t = t.upper()
    t = t.replace("_", "")
    t = t.replace("H-", "H=").replace("T-", "T=")
    t = t.replace("D-", "D=").replace("W-", "W=")
    # Normalize all x/X to lowercase x for consistent matching
    t = t.replace("×", "x").replace("*", "x")
    t = re.sub(r"(\d)\s*[xX]\s*(\d)", r"\1x\2", t)
    # Fix "xl" OCR misread: xl4 = X14 (l is digit 1)
    t = re.sub(r"xl(\d)", r"x1\1", t)
    # Fix H=lo / H-lo → H=10
    t = re.sub(r"H[=\-][Ll][oO0]", "H=10", t)
    # Fix EWA→EW4 etc
    _ld = {'A':'4','B':'2','C':'3','S':'5','G':'6','I':'1','O':'0'}
    t = re.sub(r'^EW([A-Z])$', lambda m: 'EW'+_ld.get(m.group(1),m.group(1)), t)
    # Normalize ZONE-1 and ZONE1 → ZONE 1
    t = re.sub(r'^ZONE[\s\-_]*(\d+[A-Z]?)$', r'ZONE \1', t)
    return t.strip()

for e in elements:
    e["text"] = clean(e["text"])

# Merge split "ZONE" + "1" tokens
sorted_idx = sorted(range(len(elements)), key=lambda i:(round(elements[i]["y"]/30), elements[i]["x"]))
used = set()
for ii, i in enumerate(sorted_idx):
    if i in used: continue
    if elements[i]["text"] == "ZONE":
        for j in sorted_idx[ii+1:ii+8]:
            if j in used: continue
            ne = elements[j]
            if abs(ne["y"]-elements[i]["y"]) < 50 and abs(ne["x"]-elements[i]["x"]) < 300:
                if re.match(r'^\d[A-Z]?$', ne["text"]):
                    elements[i]["text"] = "ZONE " + ne["text"]
                    used.add(j); break
elements = [e for i,e in enumerate(elements) if i not in used]

def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

def is_duplicate(x, y, positions, th=25):
    return any(abs(x-px)<th and abs(y-py)<th for px,py in positions)

def parse_feet_inches_fraction(text):
    text = text.replace(" ","")
    fm = re.search(r"(\d+)'", text)
    if not fm: return None
    feet = int(fm.group(1))
    im   = re.search(r"'(\d+)", text)
    fracm= re.search(r"(\d+)/(\d+)", text)
    inches = int(im.group(1)) if im else 0
    frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
    if inches > 12: inches = int(str(inches)[0])
    return math.ceil(feet + (inches+frac)/12)

def ft_to_m(v):     return round(v*0.3048,2) if v else None
def sqft_to_sqm(v): return round(v*0.0929,2) if v else None

def parse_size_from_elements(base):
    best_match, best_dist = None, 999999
    for e in elements:
        d = dist(base, e)
        if d < 700:
            txt = e["text"]
            # Feet format: 12'0"x15'0" or 12'0"15'0" (no separator)
            m = re.search(r"(\d+'\d+(?:\s*\d+/\d+)?\"?)\s*x?\s*(\d+'\d+(?:\s*\d+/\d+)?\"?)", txt)
            if m:
                w = parse_feet_inches_fraction(m.group(1))
                l = parse_feet_inches_fraction(m.group(2))
                if w and l and 5<=w<=50 and 5<=l<=50 and d<best_dist:
                    best_match=(w,l); best_dist=d
    if best_match:
        w,l = best_match
        if w>l: w,l=l,w
        return {"length_ft":l,"width_ft":w}

    # Simple NxM
    best_match, best_area = None, 0
    for e in elements:
        d = dist(base, e)
        if d < 700:
            txt = e["text"]
            if "d=" in txt or "w=" in txt: continue
            m = re.search(r"(\d{1,2})\s*x\s*(\d{1,2})", txt)
            if m:
                w,l = int(m.group(1)),int(m.group(2))
                if 5<=w<=40 and 5<=l<=40 and w*l>best_area:
                    best_match=(w,l); best_area=w*l
    if best_match:
        w,l = best_match
        if w>l: w,l=l,w
        return {"length_ft":l,"width_ft":w}
    return None

# ── Zones ─────────────────────────────────────────────────────────────────────
zone_pattern = r"^ZONE\s+\d+[A-Z]?$"

def nearest_room(base):
    ignore = ["ZONE","T=","H=","D=","W=","NORTH","SOUTH","EAST","WEST","IW","EW"]
    best,bd = None,999999
    for e in elements:
        txt = e["text"]
        if any(i in txt for i in ignore): continue
        d = dist(base,e)
        if d < bd: best,bd = txt,d
    return best

zones = []
for e in elements:
    if re.match(zone_pattern, e["text"]):
        size = parse_size_from_elements(e)
        area = round(size["length_ft"]*size["width_ft"],2) if size else None
        zones.append({
            "zone_id":   e["text"],
            "name":      nearest_room(e) or "UNKNOWN",
            "size":      size,
            "area_sqft": area,
            "area_sqm":  sqft_to_sqm(area),
            "width_ft":  size["width_ft"]  if size else None,
            "length_ft": size["length_ft"] if size else None,
            "width_m":   ft_to_m(size["width_ft"])  if size else None,
            "length_m":  ft_to_m(size["length_ft"]) if size else None,
        })

total_building_area = round(sum(z["area_sqft"] for z in zones if z["area_sqft"]),2)

# ── Doors ─────────────────────────────────────────────────────────────────────
door_sizes, door_positions = [], []
for e in elements:
    m = re.search(r"D=?(\d{1,2})x(\d{1,2})", e["text"])
    if m:
        w,h = int(m.group(1)),int(m.group(2))
        if 2<=w<=5 and 6<=h<=8:
            if not is_duplicate(e["x"],e["y"],door_positions):
                door_positions.append((e["x"],e["y"])); door_sizes.append((w,h))

# ── Windows ───────────────────────────────────────────────────────────────────
window_sizes, window_positions = [], []
for e in elements:
    m = re.search(r"W=?(\d{1,2})x(\d{1,2})", e["text"])
    if m:
        w,h = int(m.group(1)),int(m.group(2))
        if 2<=w<=8 and 2<=h<=8:
            if not is_duplicate(e["x"],e["y"],window_positions):
                window_positions.append((e["x"],e["y"])); window_sizes.append((w,h))

# ── Aggregation ───────────────────────────────────────────────────────────────
doors   = [{"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c} for (w,h),c in Counter(door_sizes).items()]
windows = [{"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c} for (w,h),c in Counter(window_sizes).items()]

total_door_area   = sum(d["total_area_sqft"] for d in doors)
total_window_area = sum(w["total_area_sqft"] for w in windows)

# ── Walls ─────────────────────────────────────────────────────────────────────
iw_sizes, ew_sizes, visited = [], [], []

def find_wall_specs(base):
    t=h=None
    for e in elements:
        if dist(base,e)<200:
            txt=e["text"]
            if t is None:
                m=re.search(r"T=?\s*(\d{1,2})",txt)
                if m: t=int(m.group(1))
            if h is None:
                m=re.search(r"H=?\s*(\d{1,2})",txt)
                if m: h=int(m.group(1))
    return t,h

for e in elements:
    txt = e["text"].replace("1W","IW").replace("lW","IW")
    if any(dist(e,v)<120 for v in visited): continue
    if "IW" in txt:
        t,h = find_wall_specs(e)
        if t and h and 3<=t<=12 and 6<=h<=12: iw_sizes.append((t,h)); visited.append(e)
    elif "EW" in txt:
        t,h = find_wall_specs(e)
        if t and h and 6<=t<=15 and 8<=h<=15: ew_sizes.append((t,h)); visited.append(e)

internal_walls = [{"type":"internal","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(iw_sizes).items()]
external_walls = [{"type":"external","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(ew_sizes).items()]

# ── Output ────────────────────────────────────────────────────────────────────
print(json.dumps({
    "zones": zones,
    "openings": {"doors": doors, "windows": windows},
    "walls":    {"internal": internal_walls, "external": external_walls},
    "summary": {
        "total_area_sqft":         total_building_area,
        "total_area_sqm":          sqft_to_sqm(total_building_area),
        "total_zones":             len(zones),
        "total_doors":             sum(d["count"] for d in doors),
        "total_windows":           sum(w["count"] for w in windows),
        "total_internal_walls":    sum(w["count"] for w in internal_walls),
        "total_external_walls":    sum(w["count"] for w in external_walls),
        "total_door_area_sqft":    total_door_area,
        "total_window_area_sqft":  total_window_area,
        "total_opening_area_sqft": total_door_area + total_window_area,
        "total_door_area_sqm":     sqft_to_sqm(total_door_area),
        "total_window_area_sqm":   sqft_to_sqm(total_window_area),
    }
}, indent=4))


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

# if input_path.lower().endswith(".pdf"):
#     pages = convert_from_path(input_path, dpi=500)
# else:
#     pages = [cv2.imread(input_path)]

# elements = []

# def preprocess(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
#     _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#     return th

# for page in pages:
#     image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
#     for (bbox, text, prob) in reader.readtext(preprocess(image)):
#         if prob < MIN_CONF: continue
#         x = int(sum([p[0] for p in bbox]) / 4)
#         y = int(sum([p[1] for p in bbox]) / 4)
#         elements.append({"text": text.upper(), "x": x, "y": y})

# def clean(t):
#     t = t.upper()
#     t = t.replace("_", "")
#     t = t.replace("H-", "H=").replace("T-", "T=")
#     t = t.replace("D-", "D=").replace("W-", "W=")
#     # Normalize all x/X to lowercase x for consistent matching
#     t = t.replace("×", "x").replace("*", "x")
#     t = re.sub(r"(\d)\s*[xX]\s*(\d)", r"\1x\2", t)
#     # Fix "xl" OCR misread: xl4 = X14 (l is digit 1)
#     t = re.sub(r"xl(\d)", r"x1\1", t)
#     # Fix H=lo / H-lo → H=10
#     t = re.sub(r"H[=\-][Ll][oO0]", "H=10", t)
#     # Fix EWA→EW4 etc
#     _ld = {'A':'4','B':'2','C':'3','S':'5','G':'6','I':'1','O':'0'}
#     t = re.sub(r'^EW([A-Z])$', lambda m: 'EW'+_ld.get(m.group(1),m.group(1)), t)
#     # Normalize ZONE-1 and ZONE1 → ZONE 1
#     t = re.sub(r'^ZONE[\s\-_]*(\d+[A-Z]?)$', r'ZONE \1', t)
#     return t.strip()

# for e in elements:
#     e["text"] = clean(e["text"])

# # Merge split "ZONE" + "1" tokens
# sorted_idx = sorted(range(len(elements)), key=lambda i:(round(elements[i]["y"]/30), elements[i]["x"]))
# used = set()
# for ii, i in enumerate(sorted_idx):
#     if i in used: continue
#     if elements[i]["text"] == "ZONE":
#         for j in sorted_idx[ii+1:ii+8]:
#             if j in used: continue
#             ne = elements[j]
#             if abs(ne["y"]-elements[i]["y"]) < 50 and abs(ne["x"]-elements[i]["x"]) < 300:
#                 if re.match(r'^\d[A-Z]?$', ne["text"]):
#                     elements[i]["text"] = "ZONE " + ne["text"]
#                     used.add(j); break
# elements = [e for i,e in enumerate(elements) if i not in used]

# def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

# def is_duplicate(x, y, positions, th=25):
#     return any(abs(x-px)<th and abs(y-py)<th for px,py in positions)

# def parse_feet_inches_fraction(text):
#     text = text.replace(" ","")
#     fm = re.search(r"(\d+)'", text)
#     if not fm: return None
#     feet = int(fm.group(1))
#     im   = re.search(r"'(\d+)", text)
#     fracm= re.search(r"(\d+)/(\d+)", text)
#     inches = int(im.group(1)) if im else 0
#     frac   = int(fracm.group(1))/int(fracm.group(2)) if fracm else 0
#     if inches > 12: inches = int(str(inches)[0])
#     return math.ceil(feet + (inches+frac)/12)

# def ft_to_m(v):     return round(v*0.3048,2) if v else None
# def sqft_to_sqm(v): return round(v*0.0929,2) if v else None

# def parse_size_from_elements(base):
#     best_match, best_dist = None, 999999
#     for e in elements:
#         d = dist(base, e)
#         if d < 700:
#             txt = e["text"]
#             # Feet format: 12'0"x15'0" or 12'0"15'0" (no separator)
#             m = re.search(r"(\d+'\d+(?:\s*\d+/\d+)?\"?)\s*x?\s*(\d+'\d+(?:\s*\d+/\d+)?\"?)", txt)
#             if m:
#                 w = parse_feet_inches_fraction(m.group(1))
#                 l = parse_feet_inches_fraction(m.group(2))
#                 if w and l and 5<=w<=50 and 5<=l<=50 and d<best_dist:
#                     best_match=(w,l); best_dist=d
#     if best_match:
#         w,l = best_match
#         if w>l: w,l=l,w
#         return {"length_ft":l,"width_ft":w}

#     # Simple NxM
#     best_match, best_area = None, 0
#     for e in elements:
#         d = dist(base, e)
#         if d < 700:
#             txt = e["text"]
#             if "d=" in txt or "w=" in txt: continue
#             m = re.search(r"(\d{1,2})\s*x\s*(\d{1,2})", txt)
#             if m:
#                 w,l = int(m.group(1)),int(m.group(2))
#                 if 5<=w<=40 and 5<=l<=40 and w*l>best_area:
#                     best_match=(w,l); best_area=w*l
#     if best_match:
#         w,l = best_match
#         if w>l: w,l=l,w
#         return {"length_ft":l,"width_ft":w}
#     return None

# # ── Zones ─────────────────────────────────────────────────────────────────────
# zone_pattern = r"^ZONE\s+\d+[A-Z]?$"

# def nearest_room(base):
#     ignore = ["ZONE","T=","H=","D=","W=","NORTH","SOUTH","EAST","WEST","IW","EW"]
#     best,bd = None,999999
#     for e in elements:
#         txt = e["text"]
#         if any(i in txt for i in ignore): continue
#         d = dist(base,e)
#         if d < bd: best,bd = txt,d
#     return best

# zones = []
# for e in elements:
#     if re.match(zone_pattern, e["text"]):
#         size = parse_size_from_elements(e)
#         area = round(size["length_ft"]*size["width_ft"],2) if size else None
#         zones.append({
#             "zone_id":   e["text"],
#             "name":      nearest_room(e) or "UNKNOWN",
#             "size":      size,
#             "area_sqft": area,
#             "area_sqm":  sqft_to_sqm(area),
#             "width_ft":  size["width_ft"]  if size else None,
#             "length_ft": size["length_ft"] if size else None,
#             "width_m":   ft_to_m(size["width_ft"])  if size else None,
#             "length_m":  ft_to_m(size["length_ft"]) if size else None,
#         })

# total_building_area = round(sum(z["area_sqft"] for z in zones if z["area_sqft"]),2)

# # ── Doors ─────────────────────────────────────────────────────────────────────
# door_sizes, door_positions = [], []
# for e in elements:
#     m = re.search(r"d=?(\d{1,2})x(\d{1,2})", e["text"])
#     if m:
#         w,h = int(m.group(1)),int(m.group(2))
#         if 2<=w<=5 and 6<=h<=8:
#             if not is_duplicate(e["x"],e["y"],door_positions):
#                 door_positions.append((e["x"],e["y"])); door_sizes.append((w,h))

# # ── Windows ───────────────────────────────────────────────────────────────────
# window_sizes, window_positions = [], []
# for e in elements:
#     m = re.search(r"w=?(\d{1,2})x(\d{1,2})", e["text"])
#     if m:
#         w,h = int(m.group(1)),int(m.group(2))
#         if 2<=w<=8 and 2<=h<=8:
#             if not is_duplicate(e["x"],e["y"],window_positions):
#                 window_positions.append((e["x"],e["y"])); window_sizes.append((w,h))

# # ── Aggregation ───────────────────────────────────────────────────────────────
# doors   = [{"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c} for (w,h),c in Counter(door_sizes).items()]
# windows = [{"size_ft":{"width":w,"height":h},"count":c,"area_each_sqft":w*h,"total_area_sqft":w*h*c} for (w,h),c in Counter(window_sizes).items()]

# total_door_area   = sum(d["total_area_sqft"] for d in doors)
# total_window_area = sum(w["total_area_sqft"] for w in windows)

# # ── Walls ─────────────────────────────────────────────────────────────────────
# iw_sizes, ew_sizes, visited = [], [], []

# def find_wall_specs(base):
#     t=h=None
#     for e in elements:
#         if dist(base,e)<200:
#             txt=e["text"]
#             if t is None:
#                 m=re.search(r"t=?\s*(\d{1,2})",txt)
#                 if m: t=int(m.group(1))
#             if h is None:
#                 m=re.search(r"h=?\s*(\d{1,2})",txt)
#                 if m: h=int(m.group(1))
#     return t,h

# for e in elements:
#     txt = e["text"].replace("1W","IW").replace("lW","IW")
#     if any(dist(e,v)<120 for v in visited): continue
#     if "IW" in txt:
#         t,h = find_wall_specs(e)
#         if t and h and 3<=t<=12 and 6<=h<=12: iw_sizes.append((t,h)); visited.append(e)
#     elif "EW" in txt:
#         t,h = find_wall_specs(e)
#         if t and h and 6<=t<=15 and 8<=h<=15: ew_sizes.append((t,h)); visited.append(e)

# internal_walls = [{"type":"internal","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(iw_sizes).items()]
# external_walls = [{"type":"external","thickness_inch":t,"height_ft":h,"count":c} for (t,h),c in Counter(ew_sizes).items()]

# # ── Output ────────────────────────────────────────────────────────────────────
# print(json.dumps({
#     "zones": zones,
#     "openings": {"doors": doors, "windows": windows},
#     "walls":    {"internal": internal_walls, "external": external_walls},
#     "summary": {
#         "total_area_sqft":         total_building_area,
#         "total_area_sqm":          sqft_to_sqm(total_building_area),
#         "total_zones":             len(zones),
#         "total_doors":             sum(d["count"] for d in doors),
#         "total_windows":           sum(w["count"] for w in windows),
#         "total_internal_walls":    sum(w["count"] for w in internal_walls),
#         "total_external_walls":    sum(w["count"] for w in external_walls),
#         "total_door_area_sqft":    total_door_area,
#         "total_window_area_sqft":  total_window_area,
#         "total_opening_area_sqft": total_door_area + total_window_area,
#         "total_door_area_sqm":     sqft_to_sqm(total_door_area),
#         "total_window_area_sqm":   sqft_to_sqm(total_window_area),
#     }
# }, indent=4))


# # import cv2
# # import easyocr
# # import re
# # import json
# # import numpy as np
# # import math
# # import sys
# # from collections import Counter
# # from pdf2image import convert_from_path

# # # ================= SETTINGS =================
# # MIN_CONF = 0.30
# # reader = easyocr.Reader(['en'], gpu=False)

# # input_path = sys.argv[1]

# # # ================= LOAD IMAGE / PDF =================
# # if input_path.lower().endswith(".pdf"):
# #     pages = convert_from_path(input_path, dpi=500)
# # else:
# #     pages = [cv2.imread(input_path)]

# # elements = []

# # # ================= PREPROCESS =================
# # def preprocess(img):
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.bilateralFilter(gray, 9, 75, 75)
# #     _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# #     return th

# # # ================= OCR =================
# # for page in pages:
# #     image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
# #     processed = preprocess(image)

# #     results = reader.readtext(processed)

# #     for (bbox, text, prob) in results:
# #         if prob < MIN_CONF:
# #             continue

# #         text = text.upper()

# #         x = int(sum([p[0] for p in bbox]) / 4)
# #         y = int(sum([p[1] for p in bbox]) / 4)

# #         elements.append({
# #             "text": text,
# #             "x": x,
# #             "y": y
# #         })

# # # ================= CLEAN =================
# # def clean(t):
# #     t=t.upper()
# #     t = t.replace("-", "=")
# #     t = t.replace("_", "")
# #     t = t.replace("H-", "H=").replace("T-", "T=")
# #     t = t.replace("D-", "D=").replace("W-", "W=")
# #     t = t.replace("X", "x")
# #     return t.strip()

# # for e in elements:
# #     e["text"] = clean(e["text"])

# # # ================= HELPERS =================
# # def dist(a, b):
# #     return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

# # def nearest(base, pattern, max_d=350):
# #     best, best_d = None, 999999
# #     for e in elements:
# #         if re.match(pattern, e["text"]):
# #             d = dist(base, e)
# #             if d < best_d and d < max_d:
# #                 best, best_d = e["text"], d
# #     return best

# # def nearest_dir(base):
# #     dirs = ["NORTH", "SOUTH", "EAST", "WEST"]
# #     best, best_d = None, 350
# #     for e in elements:
# #         if e["text"] in dirs:
# #             d = dist(base, e)
# #             if d < best_d:
# #                 best, best_d = e["text"], d
# #     return best

# # def nearest_number(base, min_val=6, max_val=8):
# #     for e in elements:
# #         if dist(base, e) < 120:
# #             nums = re.findall(r"\d{1,2}", e["text"])
# #             for n in nums:
# #                 val = int(n)
# #                 if min_val <= val <= max_val:
# #                     return val
# #     return None

# # def is_duplicate(x, y, positions, th=25):
# #     return any(abs(x - px) < th and abs(y - py) < th for px, py in positions)

# # def get_combined_text(base, radius=100):
# #     texts = []

# #     for e in elements:
# #         if dist(base, e) < radius:
# #             texts.append((e["y"], e["x"], e["text"]))

# #     texts = sorted(texts)
# #     return " ".join([t[2] for t in texts])

# # def parse_feet_inches_fraction(text):

# #     text = text.replace(" ", "")

# #     # Extract feet
# #     feet_match = re.search(r"(\d+)'", text)
# #     if not feet_match:
# #         return None
# #     feet = int(feet_match.group(1))

# #     # Extract inches + fraction separately
# #     inch_match = re.search(r"'(\d+)", text)
# #     frac_match = re.search(r"(\d+)/(\d+)", text)

# #     inches = int(inch_match.group(1)) if inch_match else 0
# #     frac = 0

# #     if frac_match:
# #         frac = int(frac_match.group(1)) / int(frac_match.group(2))

# #     # 🔥 FIX: prevent 41/2 issue
# #     if inches > 12:
# #         inches = int(str(inches)[0])  # take only first digit

# #     total = feet + (inches + frac) / 12

# #     return math.ceil(total)

# # def round_to_next_number(value):
# #     if value is None:
# #         return None
# #     return math.ceil(value)

# # # ---------------- SIZE PARSER ----------------
# # def parse_size_from_elements(base):

# #     best_match = None
# #     best_dist = 999999

# #     for e in elements:
# #         d = dist(base, e)

# #         if d < 700:
# #             txt = e["text"]

# #             match = re.search(r"(\d+'\d+(?:\s*\d+/\d+)?\"?)\s*[xX]\s*(\d+'\d+(?:\s*\d+/\d+)?\"?)", txt)

# #             if match:
# #                 w = parse_feet_inches_fraction(match.group(1))
# #                 l = parse_feet_inches_fraction(match.group(2))

# #                 if w and l and 5 <= w <= 50 and 5 <= l <= 50:
# #                     if d < best_dist:
# #                         best_match = (w, l)
# #                         best_dist = d

# #     if best_match:
# #         w, l = best_match
# #         if w > l:
# #             w, l = l, w

# #         return {
# #             "length_ft": round_to_next_number(l),
# #             "width_ft": round_to_next_number(w)
# #         }

# #     # =========================
# #     # STEP 2: NxM MATCH
# #     # =========================
# #     best_match = None
# #     best_area = 0

# #     for e in elements:
# #         d = dist(base, e)

# #         if d < 700:
# #             txt = e["text"]

# #             # skip doors/windows
# #             if "D=" in txt or "W=" in txt:
# #                 continue

# #             match = re.search(r"(\d{1,2})\s*[xX]\s*(\d{1,2})", txt)

# #             if match:
# #                 w = int(match.group(1))
# #                 l = int(match.group(2))

# #                 if w < 5 or l < 5:
# #                     continue

# #                 if w > 40 or l > 40:
# #                     continue

# #                 area = w * l

# #                 # choose largest → room not window
# #                 if area > best_area:
# #                     best_match = (w, l)
# #                     best_area = area

# #     if best_match:
# #         w, l = best_match
# #         if w > l:
# #             w, l = l, w

# #         return {
# #             "length_ft": round_to_next_number(l),
# #             "width_ft": round_to_next_number(w)
# #         }

# #     # =========================
# #     # STEP 3: LAST FALLBACK (NUMBERS)
# #     # =========================
# #     candidates = []

# #     for e in elements:
# #         d = dist(base, e)

# #         if d < 400:
# #             txt = e["text"]

# #             nums = re.findall(r"\d{1,2}", txt)

# #             for n in nums:
# #                 val = int(n)

# #                 # strict filter
# #                 if 5 <= val <= 40:
# #                     candidates.append((val, d))

# #     candidates = sorted(candidates, key=lambda x: x[1])

# #     values = [c[0] for c in candidates[:2]]

# #     if len(values) >= 2:
# #         w, l = values[0], values[1]

# #         if w > l:
# #             w, l = l, w

# #         return {
# #             "length_ft": round_to_next_number(l),
# #             "width_ft": round_to_next_number(w)
# #         }

# #     return None

# # # ---------------- ZONES ----------------
# # zones = []

# # zone_pattern = r"ZONE\s*\d+[A-Z]?"
# # size_pattern = r"\d+'\d+\"?x\d+'\d+\"?"

# # def nearest_room(base):
# #     ignore = ["ZONE","T=","H=","D=","W="]

# #     best = None
# #     best_d = 999999

# #     for e in elements:
# #         txt = e["text"]

# #         if any(i in txt for i in ignore):
# #             continue

# #         if txt in ["NORTH","SOUTH","EAST","WEST"]:
# #             continue

# #         if re.match(size_pattern, txt):
# #             continue

# #         d = dist(base, e)
# #         if d < best_d:
# #             best = txt
# #             best_d = d

# #     return best

# # for e in elements:
# #     if re.match(zone_pattern, e["text"]):

# #         size = parse_size_from_elements(e)

# #         area = None
# #         if size:
# #             area=round(size["length_ft"]*size["width_ft"],2)

# #         zones.append({
# #             "zone_id": e["text"],
# #             "name": nearest_room(e) or "UNKNOWN",
# #             "size": size,
# #             "area_sqft": area
# #         })

# # total_building_area = 0

# # for z in zones:
# #     if z.get("area_sqft"):
# #         total_building_area += z["area_sqft"]

# # total_building_area = round(total_building_area, 2)

# # # ================= DOORS =================
# # door_sizes = []
# # door_positions = []

# # for e in elements:
# #     txt = e["text"]
# #     x, y = e["x"], e["y"]

# #     m = re.search(r"D=?(\d{1,2})x(\d{1,2})", txt)
# #     if m:
# #         w, h = int(m.group(1)), int(m.group(2))

# #         if 2 <= w <= 5 and 6 <= h <= 8:
# #             if not is_duplicate(x, y, door_positions):
# #                 door_positions.append((x, y))
# #                 door_sizes.append((w, h))

# # # ================= WINDOWS =================
# # window_sizes = []
# # window_positions = []

# # for e in elements:
# #     txt = e["text"]
# #     x, y = e["x"], e["y"]

# #     m = re.search(r"W=?(\d{1,2})x(\d{1,2})", txt)
# #     if m:
# #         w, h = int(m.group(1)), int(m.group(2))

# #         if 2 <= w <= 8 and 2 <= h <= 8:
# #             if not is_duplicate(x, y, window_positions):
# #                 window_positions.append((x, y))
# #                 window_sizes.append((w, h))

# # # ================= AGGREGATION =================
# # doors = []

# # for (w, h), c in Counter(door_sizes).items():

# #     area_each = w * h
# #     door_total_area = area_each * c

# #     doors.append({
# #         "size_ft": {"width": w, "height": h},
# #         "count": c,
# #         "area_each_sqft": area_each,
# #         "total_area_sqft": door_total_area
# #     })

# # windows = []

# # for (w, h), c in Counter(window_sizes).items():

# #     area_each = w * h
# #     window_total_area = area_each * c

# #     windows.append({
# #         "size_ft": {"width": w, "height": h},
# #         "count": c,
# #         "area_each_sqft": area_each,
# #         "total_area_sqft": window_total_area
# #     })

# # total_door_area = sum(d["total_area_sqft"] for d in doors)
# # total_window_area = sum(w["total_area_sqft"] for w in windows)

# # # ================= WALLS (ROBUST FIX) =================
# # iw_sizes = []
# # ew_sizes = []

# # visited = []

# # def find_wall(base, wall_type):

# #     t = None
# #     h = None

# #     for e in elements:
# #         if dist(base, e) < 200:

# #             txt = e["text"]

# #             if t is None:
# #                 m = re.search(r"T=?\s*(\d{1,2})", txt)
# #                 if m:
# #                     t = int(m.group(1))

# #             if h is None:
# #                 m = re.search(r"H=?\s*(\d{1,2})", txt)
# #                 if m:
# #                     h = int(m.group(1))

# #     return t, h


# # for e in elements:

# #     txt = e["text"]
# #     txt = txt.replace("1W", "IW").replace("lW", "IW")

# #     # skip already used points
# #     if any(dist(e, v) < 120 for v in visited):
# #         continue

# #     # ---------------- INTERNAL WALL ----------------
# #     if "IW" in txt:

# #         t, h = find_wall(e, "IW")

# #         if t and h and (3 <= t <= 12) and (6 <= h <= 12):
# #             iw_sizes.append((t, h))
# #             visited.append(e)

# #     # ---------------- EXTERNAL WALL ----------------
# #     elif "EW" in txt:

# #         t, h = find_wall(e, "EW")

# #         if t and h and (6 <= t <= 15) and (8 <= h <= 15):
# #             ew_sizes.append((t, h))
# #             visited.append(e)
# # # ================= WALL SUMMARY =================
# # internal_walls = []
# # external_walls = []

# # iw_counter = Counter(iw_sizes)
# # ew_counter = Counter(ew_sizes)

# # for (t, h), c in iw_counter.items():
# #     internal_walls.append({
# #         "type": "internal",
# #         "thickness_inch": t,
# #         "height_ft": h,
# #         "count": c
# #     })

# # for (t, h), c in ew_counter.items():
# #     external_walls.append({
# #         "type": "external",
# #         "thickness_inch": t,
# #         "height_ft": h,
# #         "count": c
# #     })

# #     # ================= TOTAL COUNTS =================

# # total_zones = len(zones)

# # total_doors = sum([d["count"] for d in doors])

# # total_windows = sum([w["count"] for w in windows])

# # # ================= FINAL OUTPUT =================
# # final = {

# #     "zones": zones,

# #     "openings": {
# #         "doors": doors,
# #         "windows": windows
# #     },

# #     "walls": {
# #         "internal": internal_walls,
# #         "external": external_walls
# #     },

# #     "summary": {
# #         "total_area_sqft": total_building_area,
# #         "total_zones": total_zones,
# #         "total_doors": total_doors,
# #         "total_windows": total_windows,
# #         "total_internal_walls": sum([w["count"] for w in internal_walls]),
# #         "total_external_walls": sum([w["count"] for w in external_walls]),
# #         "total_door_area_sqft": total_door_area,
# #         "total_window_area_sqft": total_window_area,
# #         "total_opening_area_sqft": total_door_area + total_window_area
# #     }
# # }

# # print(json.dumps(final, indent=4))
