const { exec } = require("child_process");
const path = require("path");
const fs = require("fs");

function convertPdfToImages(pdfPath, dpi = 500) {

  return new Promise((resolve, reject) => {

    const base = `uploads/plan_${Date.now()}`;
const poppler = process.env.POPPLER_PATH || "pdftoppm";

    const cmd = `"${poppler}" -png -r ${dpi} "${pdfPath}" "${base}"`;

    exec(cmd, (err) => {

      if (err) return reject(err);

      const dir = path.dirname(base);
      const name = path.basename(base);

      const files = fs.readdirSync(dir)
        .filter(f => f.startsWith(name) && f.endsWith(".png"))
        .map(f => path.join(dir, f));

      resolve(files);
    });

  });

}

module.exports = { convertPdfToImages };