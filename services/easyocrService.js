const { spawn } = require("child_process");
const path = require("path");

function runOCR(imagePath) {
  return new Promise((resolve, reject) => {
    
    // Ensure the path is absolute (Fixes frontend upload path issues)
    const absoluteImagePath = path.resolve(imagePath);
    
//     const script = path.join(__dirname, "../python/easyocr_reader.py");

//   const py = spawn(
//   process.env.PYTHON_PATH || "python3",
//   [script, absoluteImagePath]
// );
const script = path.join(__dirname, "../python/easyocr_reader.py");

   const py = spawn("python", [script, absoluteImagePath]);
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
