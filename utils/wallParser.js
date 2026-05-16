function parseWallData(text) {

  // -------- TEXT NORMALIZATION --------

  let cleaned = text
    .toUpperCase()
    .replace(/-/g, "=")
    .replace(/X/g, "x")
    .replace(/\s+/g, " ")
    .trim();

  cleaned = cleaned.replace("D=3x", "D=3x7");

  const tokens = cleaned.split(" ");

  let ewList = [];
  let iwList = [];

  // -------- WALL DETECTION --------

  for (let i = 0; i < tokens.length; i++) {

    let tok = tokens[i];

    // -------- EXTERNAL WALL --------
    if (tok === "EW") {

      let t = null;
      let h = null;

      for (let j = i + 1; j < Math.min(i + 12, tokens.length); j++) {

        if (tokens[j].startsWith("T=")) {
          t = tokens[j].replace("T=", "");
        }

        if (tokens[j].startsWith("H=")) {
          h = tokens[j].replace("H=", "");
        }

        if (t && h) {
          ewList.push(`${t}-${h}`);
          break;
        }

      }

    }

    // -------- INTERNAL WALL --------
    if (tok === "IW") {

      let t = null;
      let h = null;

      for (let j = i + 1; j < Math.min(i + 12, tokens.length); j++) {

        if (tokens[j].startsWith("T=")) {
          t = tokens[j].replace("T=", "");
        }

        if (tokens[j].startsWith("H=")) {
          h = tokens[j].replace("H=", "");
        }

        if (t && h) {
          iwList.push(`${t}-${h}`);
          break;
        }

      }

    }

  }

  // -------- DOORS --------

  let doors = [...cleaned.matchAll(/D\s*=\s*(\d+x\d+)/g)]
    .map(m => m[1])
    .map(d => d.replace("3x77", "3x7"));

  // -------- WINDOWS --------

  let windows = [...cleaned.matchAll(/W\s*=\s*(\d+x\d+)/g)]
    .map(m => m[1]);

  // -------- COUNTER FUNCTION --------

  function countItems(arr) {

    const map = {};

    arr.forEach(item => {
      map[item] = (map[item] || 0) + 1;
    });

    return map;
  }

  const ewCount = countItems(ewList);
  const iwCount = countItems(iwList);
  const doorCount = countItems(doors);
  const windowCount = countItems(windows);

  // -------- JSON OUTPUT --------

  const data = {};

  for (const key in ewCount) {

    const [t, h] = key.split("-");

    data.external_walls = {
      thickness: Number(t),
      height: Number(h),
      count: ewCount[key]
    };

  }

  data.internal_walls = Object.keys(iwCount).map(key => {

    const [t, h] = key.split("-");

    return {
      thickness: Number(t),
      height: Number(h),
      count: iwCount[key]
    };

  });

  data.doors = Object.keys(doorCount).map(size => ({
    size: size.toLowerCase(),
    count: doorCount[size]
  }));

  data.windows = Object.keys(windowCount).map(size => ({
    size: size.toLowerCase(),
    count: windowCount[size]
  }));

  return data;

}

module.exports = { parseWallData };
