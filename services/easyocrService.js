// const { spawn } = require("child_process");
// const path = require("path");

// function runOCR(imagePath) {
//   return new Promise((resolve, reject) => {
    
//     // Ensure the path is absolute (Fixes frontend upload path issues)
// //     const absoluteImagePath = path.resolve(imagePath);
    
// //     const script = path.join(__dirname, "../python/easyocr_reader.py");

// //   const py = spawn(
// //   process.env.PYTHON_PATH || "python3",
// //   [script, absoluteImagePath]
// // );

// // Use Render's environment variable if it exists, otherwise fall back to your local path
// const pythonExecutable = process.env.RENDER 
//   ? 'python3' 
//   : '/Users/impacgo/Downloads/archiquant_dashboard/backend/venv/bin/python';

// // Alternatively, you can use relative environment paths:
// // const pythonExecutable = process.env.NODE_ENV === 'production' ? 'python3' : './venv/bin/python';

// const pythonProcess = spawn(pythonExecutable, [
//     // 💡 Pro tip: Use path.join to avoid hardcoding '/opt/render/project/src/...'
//     path.join(__dirname, 'python', 'easyocr_reader.py'), 
//     uploadedFilePath
// ]);
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
const { spawn } = require("child_process");
const path = require("path");

function runOCR(imagePath) {
  return new Promise((resolve, reject) => {
    
    // 1. Ensure the path is absolute (Crucial for handling backend file uploads)
    const absoluteImagePath = path.resolve(imagePath);
    
    // 2. Safely locate the python script using path.join
    const scriptPath = path.join(__dirname, "..", "python", "easyocr_reader.py");

    // 3. Dynamically set the python path based on the environment
    const pythonExecutable = '/usr/bin/python3';

    // 4. Spawn the process (Using 'py' so it hooks into your listeners below)
    const py = spawn(pythonExecutable, [scriptPath, absoluteImagePath]);

    let output = "";
    let error = "";

    py.stdout.on("data", (data) => {
      // Collecting output chunks
      output += data.toString();
    });

    py.stderr.on("data", (data) => {
      // Collecting error chunks (Mac often puts warnings here)
      error += data.toString();
    });

    py.on("error", (err) => {
      console.error("SPAWN ERROR:", err);
      reject(err);
    });

    py.on("close", (code) => {
      console.log("PYTHON EXIT CODE:", code);

      if (code !== 0) {
        console.error("PYTHON STDERR:", error);
        return reject(error || "Python process failed");
      }

      try {
        // Fix for Mac: Find the first '{' and last '}' to ignore PyTorch terminal warnings
        const jsonStartIndex = output.indexOf("{");
        const jsonEndIndex = output.lastIndexOf("}");

        if (jsonStartIndex !== -1 && jsonEndIndex !== -1) {
          const cleanJsonString = output.substring(jsonStartIndex, jsonEndIndex + 1);
          resolve(JSON.parse(cleanJsonString));
        } else {
          console.error("RAW PYTHON OUTPUT:", output);
          reject("No valid JSON found in Python output.");
        }
      } catch (e) {
        console.error("JSON PARSE ERROR. RAW OUTPUT WAS:", output);
        reject("Invalid JSON from Python");
      }
    });

  });
}

module.exports = { runOCR };
