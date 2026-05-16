const { runOCR } = require("../services/easyocrService");

exports.processPlan = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "No file uploaded"
      });
    }

    const filePath = req.file.path;

    const result = await runOCR(filePath);

    return res.json({
      success: true,
      data: result
    });

  } catch (err) {
    console.error(err);

    return res.status(500).json({
      success: false,
      message: "Processing failed",
      error: err.toString()
    });
  }
};