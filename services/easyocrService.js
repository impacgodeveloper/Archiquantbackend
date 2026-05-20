// // const { spawn } = require("child_process");
// // const path = require("path");

// // function runOCR(imagePath) {
// //   return new Promise((resolve, reject) => {
    
// //     // Ensure the path is absolute (Fixes frontend upload path issues)
// // //     const absoluteImagePath = path.resolve(imagePath);
    
// // //     const script = path.join(__dirname, "../python/easyocr_reader.py");

// // //   const py = spawn(
// // //   process.env.PYTHON_PATH || "python3",
// // //   [script, absoluteImagePath]
// // // );

// // // Use Render's environment variable if it exists, otherwise fall back to your local path
// // const pythonExecutable = process.env.RENDER 
// //   ? 'python3' 
// //   : '/Users/impacgo/Downloads/archiquant_dashboard/backend/venv/bin/python';

// // // Alternatively, you can use relative environment paths:
// // // const pythonExecutable = process.env.NODE_ENV === 'production' ? 'python3' : './venv/bin/python';

// // const pythonProcess = spawn(pythonExecutable, [
// //     // 💡 Pro tip: Use path.join to avoid hardcoding '/opt/render/project/src/...'
// //     path.join(__dirname, 'python', 'easyocr_reader.py'), 
// //     uploadedFilePath
// // ]);
// //     let output = "";
// //     let error = "";

// //     py.stdout.on("data", (data) => {
// //       // Collecting output chunks
// //       output += data.toString();
// //     });

// //     py.stderr.on("data", (data) => {
// //       // Collecting error chunks (Mac often puts warnings here)
// //       error += data.toString();
// //     });

// //     py.on("error", (err) => {
// //       console.error("SPAWN ERROR:", err);
// //       reject(err);
// //     });

// //     py.on("close", (code) => {
// //       console.log("PYTHON EXIT CODE:", code);

// //       if (code !== 0) {
// //         console.error("PYTHON STDERR:", error);
// //         return reject(error || "Python process failed");
// //       }

// //       try {
// //         // Fix for Mac: Find the first '{' and last '}' to ignore PyTorch terminal warnings
// //         const jsonStartIndex = output.indexOf("{");
// //         const jsonEndIndex = output.lastIndexOf("}");

// //         if (jsonStartIndex !== -1 && jsonEndIndex !== -1) {
// //           const cleanJsonString = output.substring(jsonStartIndex, jsonEndIndex + 1);
// //           resolve(JSON.parse(cleanJsonString));
// //         } else {
// //           console.error("RAW PYTHON OUTPUT:", output);
// //           reject("No valid JSON found in Python output.");
// //         }
// //       } catch (e) {
// //         console.error("JSON PARSE ERROR. RAW OUTPUT WAS:", output);
// //         reject("Invalid JSON from Python");
// //       }
// //     });

// //   });
// // }

// // module.exports = { runOCR };
// const { spawn } = require("child_process");
// const path = require("path");

// function runOCR(imagePath) {
//   return new Promise((resolve, reject) => {
    
//     // 1. Ensure the path is absolute (Crucial for handling backend file uploads)
//     const absoluteImagePath = path.resolve(imagePath);
    
//     // 2. Safely locate the python script using path.join
//     const scriptPath = path.join(__dirname, "..", "python", "easyocr_reader.py");

//     // 3. Dynamically set the python path based on the environment
//     const pythonExecutable = '/usr/bin/python3';

//     // 4. Spawn the process (Using 'py' so it hooks into your listeners below)
//     const py = spawn(pythonExecutable, [scriptPath, absoluteImagePath]);

//     let output = "";
//     let error = "";

//     py.stdout.on("data", (data) => {
//       // Collecting output chunks
//       output += data.toString();
//     });

//     py.stderr.on("data", (data) => {
//       // Collecting error chunks (Mac often puts warnings here)
//       error += data.toString();
//     });

//     py.on("error", (err) => {
//       console.error("SPAWN ERROR:", err);
//       reject(err);
//     });

//     py.on("close", (code) => {
//       console.log("PYTHON EXIT CODE:", code);

//       if (code !== 0) {
//         console.error("PYTHON STDERR:", error);
//         return reject(error || "Python process failed");
//       }

//       try {
//         // Fix for Mac: Find the first '{' and last '}' to ignore PyTorch terminal warnings
//         const jsonStartIndex = output.indexOf("{");
//         const jsonEndIndex = output.lastIndexOf("}");

//         if (jsonStartIndex !== -1 && jsonEndIndex !== -1) {
//           const cleanJsonString = output.substring(jsonStartIndex, jsonEndIndex + 1);
//           resolve(JSON.parse(cleanJsonString));
//         } else {
//           console.error("RAW PYTHON OUTPUT:", output);
//           reject("No valid JSON found in Python output.");
//         }
//       } catch (e) {
//         console.error("JSON PARSE ERROR. RAW OUTPUT WAS:", output);
//         reject("Invalid JSON from Python");
//       }
//     });

//   });
// }

// module.exports = { runOCR };

// ocrService.js  —  ArchiQuant Vision OCR  (Node.js / Express)
// =============================================================
// Drop this file into your backend and call analyzeFloorPlan().
// Replaces EasyOCR entirely.
//
// Required env var: ANTHROPIC_API_KEY
// Install:  npm install axios form-data   (or use existing axios)
//
// Usage in your route:
//   const { analyzeFloorPlan } = require('./ocrService');
//   const result = await analyzeFloorPlan(req.file.path);
//   res.json(result);

const fs   = require('fs');
const path = require('path');
const axios = require('axios');

const MODEL      = 'claude-opus-4-5';
const API_URL    = 'https://api.anthropic.com/v1/messages';
const MAX_TOKENS = 4096;

// ─── PROMPT ───────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are an expert architectural floor plan analyser.
Your job is to read a floor plan image and output ONLY a valid JSON object —
no explanation, no markdown fences, no extra text.

Identify and label:
  - External walls  → ew1, ew2, ew3 ... (north=ew1, east=ew2, south=ew3, west=ew4; use ew5+ for extra outer walls)
  - Internal walls  → iw1, iw2, iw3 ...
  - Doors           → d1, d2, d3 ...
  - Windows         → w1, w2, w3 ...

For EACH external wall list every element physically attached to or opening through it:
  - windows cut into it
  - doors cut into it
  - internal walls that meet/intersect it

For EACH internal wall list:
  - which external walls it connects (starts/ends at)
  - doors or windows cut through it

Output schema (strict JSON, no trailing commas, use null for unknowns):
{
  "external_walls": {
    "ew1": {
      "id": "ew1",
      "position": "north",
      "length": null,
      "thickness": null,
      "height": null,
      "connected": { "windows": [], "doors": [], "internal_walls": [] }
    }
  },
  "internal_walls": {
    "iw1": {
      "id": "iw1",
      "length": null,
      "thickness": null,
      "height": null,
      "connects_external": [],
      "connected": { "doors": [], "windows": [] }
    }
  },
  "doors": {
    "d1": { "id": "d1", "width": null, "height": null, "swing": null, "on_wall": "<wall_id>" }
  },
  "windows": {
    "w1": { "id": "w1", "width": null, "height": null, "on_wall": "<wall_id>" }
  },
  "summary": {
    "total_external_walls": 0,
    "total_internal_walls": 0,
    "total_doors": 0,
    "total_windows": 0,
    "zones": []
  }
}

Extract visible dimension labels (T=9, H=10, L=20, D=3x7, W=4x4).
zones = list of room labels visible in plan (e.g. ["BEDROOM","KITCHEN"]).
Output raw JSON only.`;

// ─── MEDIA TYPE ───────────────────────────────────────────────────────────
function getMediaType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const map = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.png': 'image/png',  '.webp': 'image/webp',
    '.gif': 'image/gif',
  };
  return map[ext] || 'image/jpeg';
}

// ─── POST-PROCESS ─────────────────────────────────────────────────────────
// Ensures door/window on_wall values are reflected in their wall's connected list
function postProcess(data) {
  const ew  = data.external_walls || {};
  const iw  = data.internal_walls || {};
  const drs = data.doors   || {};
  const win = data.windows || {};

  const findWall = (id) => ew[id] || iw[id];

  for (const [dId, d] of Object.entries(drs)) {
    const wall = findWall(d.on_wall);
    if (wall) {
      wall.connected = wall.connected || {};
      wall.connected.doors = wall.connected.doors || [];
      if (!wall.connected.doors.includes(dId)) wall.connected.doors.push(dId);
    }
  }

  for (const [wId, w] of Object.entries(win)) {
    const wall = findWall(w.on_wall);
    if (wall) {
      wall.connected = wall.connected || {};
      wall.connected.windows = wall.connected.windows || [];
      if (!wall.connected.windows.includes(wId)) wall.connected.windows.push(wId);
    }
  }

  data.summary = {
    total_external_walls: Object.keys(ew).length,
    total_internal_walls: Object.keys(iw).length,
    total_doors:          Object.keys(drs).length,
    total_windows:        Object.keys(win).length,
    zones: data.summary?.zones || [],
  };

  return data;
}

// ─── MAIN EXPORT ──────────────────────────────────────────────────────────
/**
 * analyzeFloorPlan
 * @param {string} imagePath  - absolute or relative path to the image file
 * @returns {Promise<object>} - parsed connected-node JSON
 */
async function analyzeFloorPlan(imagePath) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY env var is not set');

  const imageBuffer = fs.readFileSync(imagePath);
  const base64Data  = imageBuffer.toString('base64');
  const mediaType   = getMediaType(imagePath);

  const payload = {
    model:      MODEL,
    max_tokens: MAX_TOKENS,
    system:     SYSTEM_PROMPT,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: mediaType, data: base64Data },
          },
          { type: 'text', text: 'Analyse this floor plan and output the JSON as instructed.' },
        ],
      },
    ],
  };

  const response = await axios.post(API_URL, payload, {
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         apiKey,
      'anthropic-version': '2023-06-01',
    },
    timeout: 60000,
  });

  // Extract text from response
  const textBlocks = (response.data.content || [])
    .filter(b => b.type === 'text')
    .map(b => b.text)
    .join('\n')
    .trim();

  // Strip markdown fences if present
  const clean = textBlocks
    .replace(/^```(?:json)?\s*/m, '')
    .replace(/\s*```$/m, '')
    .trim();

  let parsed;
  try {
    parsed = JSON.parse(clean);
  } catch (e) {
    throw new Error(`Claude returned non-JSON response:\n${clean.slice(0, 500)}`);
  }

  return postProcess(parsed);
}

// ─── EXPRESS ROUTE HELPER ─────────────────────────────────────────────────
/**
 * Express middleware that expects multer to have uploaded the file.
 * Use with:
 *   const upload = multer({ dest: 'uploads/' });
 *   router.post('/ocr', upload.single('plan'), ocrRoute);
 */
async function ocrRoute(req, res) {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

  try {
    const result = await analyzeFloorPlan(req.file.path);
    // Clean up temp file
    fs.unlink(req.file.path, () => {});
    res.json({ success: true, data: result });
  } catch (err) {
    console.error('[OCR Error]', err.message);
    res.status(500).json({ error: err.message });
  }
}

module.exports = { analyzeFloorPlan, ocrRoute };

/*
──────────────────────────────────────────
SAMPLE OUTPUT (2-bedroom apartment plan)
──────────────────────────────────────────
{
  "external_walls": {
    "ew1": {
      "id": "ew1", "position": "north", "length": "20ft", "thickness": "9in", "height": "10ft",
      "connected": { "windows": ["w1", "w2"], "doors": [], "internal_walls": ["iw1"] }
    },
    "ew2": {
      "id": "ew2", "position": "east",  "length": "15ft", "thickness": "9in", "height": "10ft",
      "connected": { "windows": ["w3"], "doors": [], "internal_walls": ["iw2"] }
    },
    "ew3": {
      "id": "ew3", "position": "south", "length": "20ft", "thickness": "9in", "height": "10ft",
      "connected": { "windows": [], "doors": ["d1"], "internal_walls": ["iw1"] }
    },
    "ew4": {
      "id": "ew4", "position": "west",  "length": "15ft", "thickness": "9in", "height": "10ft",
      "connected": { "windows": ["w4"], "doors": ["d2"], "internal_walls": ["iw2"] }
    }
  },
  "internal_walls": {
    "iw1": {
      "id": "iw1", "length": "20ft", "thickness": "4in", "height": "10ft",
      "connects_external": ["ew1", "ew3"],
      "connected": { "doors": ["d3"], "windows": [] }
    },
    "iw2": {
      "id": "iw2", "length": "15ft", "thickness": "4in", "height": "10ft",
      "connects_external": ["ew2", "ew4"],
      "connected": { "doors": [], "windows": [] }
    }
  },
  "doors": {
    "d1": { "id": "d1", "width": "3ft", "height": "7ft", "swing": "inward", "on_wall": "ew3" },
    "d2": { "id": "d2", "width": "3ft", "height": "7ft", "swing": "inward", "on_wall": "ew4" },
    "d3": { "id": "d3", "width": "2.5ft", "height": "7ft", "swing": "inward", "on_wall": "iw1" }
  },
  "windows": {
    "w1": { "id": "w1", "width": "4ft", "height": "4ft", "on_wall": "ew1" },
    "w2": { "id": "w2", "width": "4ft", "height": "4ft", "on_wall": "ew1" },
    "w3": { "id": "w3", "width": "3ft", "height": "4ft", "on_wall": "ew2" },
    "w4": { "id": "w4", "width": "3ft", "height": "4ft", "on_wall": "ew4" }
  },
  "summary": {
    "total_external_walls": 4,
    "total_internal_walls": 2,
    "total_doors": 3,
    "total_windows": 4,
    "zones": ["M.BEDROOM", "C.BEDROOM"]
  }
}
*/
