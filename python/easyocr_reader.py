import argparse
import base64
import json
import sys
import os
import re
import urllib.request
import urllib.error
 
# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-opus-4-5"
MAX_TOKENS        = 4096
 
# ─────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert architectural floor plan analyser.
Your job is to read a floor plan image and output ONLY a valid JSON object —
no explanation, no markdown fences, no extra text.
 
Identify and label:
  - External walls  → ew1, ew2, ew3 ... (north=ew1, east=ew2, south=ew3, west=ew4; ew5+ for extra outer walls)
  - Internal walls  → iw1, iw2, iw3 ...
  - Doors           → d1, d2, d3 ...
  - Windows         → w1, w2, w3 ...
  - Zones/Rooms     → use the visible room label as the key (e.g. "MBR", "KITCHEN", "LIVING")
 
For EACH external wall list every element physically attached to or opening through it:
  - windows cut into it
  - doors cut into it
  - internal walls that meet/intersect it
 
For EACH internal wall list:
  - which external walls it connects (starts/ends at)
  - doors or windows cut through it
 
For EACH zone/room list:
  - connected_external_walls: all external walls that form its boundary
  - connected_internal_walls: all internal walls that form its boundary
  - total_walls_connected: count of external + internal walls bounding this zone
  - windows: windows that open into this zone
  - doors: doors that open into this zone
 
Output schema (strict JSON, no trailing commas, use null for unknowns):
{
  "external_walls": {
    "ew1": {
      "id": "ew1",
      "position": "north",
      "length": null,
      "thickness": null,
      "height": null,
      "connected": {
        "windows": [],
        "doors": [],
        "internal_walls": []
      }
    }
  },
  "internal_walls": {
    "iw1": {
      "id": "iw1",
      "length": null,
      "thickness": null,
      "height": null,
      "connects_external": [],
      "connected": {
        "doors": [],
        "windows": []
      }
    }
  },
  "doors": {
    "d1": {
      "id": "d1",
      "width": null,
      "height": null,
      "swing": null,
      "on_wall": "<wall_id>"
    }
  },
  "windows": {
    "w1": {
      "id": "w1",
      "width": null,
      "height": null,
      "on_wall": "<wall_id>"
    }
  },
  "zones": {
    "MBR": {
      "id": "MBR",
      "label": "<full room label visible in plan>",
      "connected_external_walls": [],
      "connected_internal_walls": [],
      "total_walls_connected": 0,
      "windows": [],
      "doors": []
    }
  },
  "summary": {
    "total_external_walls": 0,
    "total_internal_walls": 0,
    "total_doors": 0,
    "total_windows": 0,
    "total_zones": 0
  }
}
 
Rules:
- Use null (not "null") for unknown values.
- If a dimension label is visible (e.g. T=9, H=10, L=20, D=3x7, W=4x4), extract it.
- zones keys must be short room codes (e.g. MBR, KIT, LIV); put the full label in the "label" field.
- total_walls_connected = len(connected_external_walls) + len(connected_internal_walls).
- Never add keys not in the schema above.
- Output raw JSON only.
"""
 
USER_PROMPT = "Analyse this floor plan and output the JSON as instructed."
 
 
# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
 
def encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type)."""
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type
 
 
def call_claude_vision(image_path: str, api_key: str) -> dict:
    """Send image to Claude Vision and return parsed JSON dict."""
    img_data, media_type = encode_image(image_path)
 
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data,
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
    }).encode("utf-8")
 
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
 
    try:
        with urllib.request.urlopen(req) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"API error {e.code}: {body}") from e
 
    response_text = raw["content"][0]["text"]
 
    # Strip accidental markdown fences
    response_text = re.sub(r"^```(json)?\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)
    response_text = response_text.strip()
 
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Claude returned non-JSON output.\n"
            f"Parse error: {e}\n"
            f"Raw response:\n{response_text[:800]}"
        ) from e
 
    return result
 
 
def post_process(data: dict) -> dict:
    """
    Ensure cross-references are consistent and compute zone wall counts.
    - Back-fill door/window on_wall → wall connected lists
    - Deduplicate all connected lists
    - Recompute total_walls_connected for every zone
    - Recompute all summary counts
    """
    ew      = data.get("external_walls", {})
    iw      = data.get("internal_walls", {})
    doors   = data.get("doors", {})
    windows = data.get("windows", {})
    zones   = data.get("zones", {})
 
    # ── Back-fill: door on_wall → wall connected.doors ───────────────────
    for d_id, d_info in doors.items():
        wall_id = d_info.get("on_wall")
        if wall_id:
            wall = ew.get(wall_id) or iw.get(wall_id)
            if wall:
                conn = wall.setdefault("connected", {})
                conn.setdefault("doors", [])
                if d_id not in conn["doors"]:
                    conn["doors"].append(d_id)
 
    # ── Back-fill: window on_wall → wall connected.windows ───────────────
    for w_id, w_info in windows.items():
        wall_id = w_info.get("on_wall")
        if wall_id:
            wall = ew.get(wall_id) or iw.get(wall_id)
            if wall:
                conn = wall.setdefault("connected", {})
                conn.setdefault("windows", [])
                if w_id not in conn["windows"]:
                    conn["windows"].append(w_id)
 
    # ── Zone wall-count enforcement ───────────────────────────────────────
    for zone_id, zone_info in zones.items():
        ext_list = zone_info.get("connected_external_walls", [])
        int_list = zone_info.get("connected_internal_walls", [])
 
        # Deduplicate (safety guard against duplicate wall IDs)
        zone_info["connected_external_walls"] = list(dict.fromkeys(ext_list))
        zone_info["connected_internal_walls"] = list(dict.fromkeys(int_list))
 
        # Also deduplicate doors and windows within the zone
        zone_info["windows"] = list(dict.fromkeys(zone_info.get("windows", [])))
        zone_info["doors"]   = list(dict.fromkeys(zone_info.get("doors", [])))
 
        # Always recompute total_walls_connected from actual lists
        zone_info["total_walls_connected"] = (
            len(zone_info["connected_external_walls"]) +
            len(zone_info["connected_internal_walls"])
        )
 
    # ── Recompute summary counts ──────────────────────────────────────────
    summary = data.setdefault("summary", {})
    summary["total_external_walls"] = len(ew)
    summary["total_internal_walls"] = len(iw)
    summary["total_doors"]          = len(doors)
    summary["total_windows"]        = len(windows)
    summary["total_zones"]          = len(zones)
 
    return data
 
 
def print_topology(data: dict):
    """Human-readable topology summary printed to stderr for debug."""
    ew      = data.get("external_walls", {})
    iw      = data.get("internal_walls", {})
    doors   = data.get("doors", {})
    windows = data.get("windows", {})
    zones   = data.get("zones", {})
    summary = data.get("summary", {})
 
    print("\n── FLOOR PLAN TOPOLOGY ──────────────────────────", file=sys.stderr)
    print(f"  External Walls : {summary.get('total_external_walls', len(ew))}", file=sys.stderr)
    print(f"  Internal Walls : {summary.get('total_internal_walls', len(iw))}", file=sys.stderr)
    print(f"  Doors          : {summary.get('total_doors', len(doors))}", file=sys.stderr)
    print(f"  Windows        : {summary.get('total_windows', len(windows))}", file=sys.stderr)
    print(f"  Zones          : {summary.get('total_zones', len(zones))}", file=sys.stderr)
 
    print("\n  External Wall Connections:", file=sys.stderr)
    for ew_id, ew_info in ew.items():
        pos  = ew_info.get("position", "?")
        conn = ew_info.get("connected", {})
        print(
            f"    {ew_id} ({pos})  iw={conn.get('internal_walls',[])}  "
            f"d={conn.get('doors',[])}  w={conn.get('windows',[])}",
            file=sys.stderr,
        )
 
    print("\n  Internal Wall Connections:", file=sys.stderr)
    for iw_id, iw_info in iw.items():
        conn = iw_info.get("connected", {})
        print(
            f"    {iw_id}  connects_ew={iw_info.get('connects_external',[])}  "
            f"d={conn.get('doors',[])}  w={conn.get('windows',[])}",
            file=sys.stderr,
        )
 
    print("\n  Zone Connections:", file=sys.stderr)
    for zone_id, zone_info in zones.items():
        label = zone_info.get("label", zone_id)
        ew_c  = zone_info.get("connected_external_walls", [])
        iw_c  = zone_info.get("connected_internal_walls", [])
        total = zone_info.get("total_walls_connected", 0)
        wins  = zone_info.get("windows", [])
        drrs  = zone_info.get("doors", [])
        print(
            f"    {zone_id} ({label})  total_walls={total}  "
            f"ew={ew_c}  iw={iw_c}  windows={wins}  doors={drrs}",
            file=sys.stderr,
        )
 
    print("─────────────────────────────────────────────────\n", file=sys.stderr)
 
 
# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="Floor plan OCR via Claude Vision")
    parser.add_argument("image", help="Path to floor plan image")
    parser.add_argument("--output", "-o", help="Write JSON to this file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()
 
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
 
    if not os.path.isfile(args.image):
        print(f"Error: File not found: {args.image}", file=sys.stderr)
        sys.exit(1)
 
    print(f"Processing: {args.image}", file=sys.stderr)
 
    raw_data    = call_claude_vision(args.image, api_key)
    final_data  = post_process(raw_data)
    print_topology(final_data)
 
    indent = 2 if args.pretty else None
    json_out = json.dumps(final_data, indent=indent)
 
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_out)
        print(f"Saved to: {args.output}", file=sys.stderr)
    else:
        print(json_out)
 
 
if __name__ == "__main__":
    main()
