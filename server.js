require("dotenv").config();
const express    = require("express");
const multer     = require("multer");
const cors       = require("cors");
const path       = require("path");
const fs         = require("fs");
const bcrypt     = require("bcryptjs");
const jwt        = require("jsonwebtoken");
const PDFDocument = require("pdfkit");
const ExcelJS    = require("exceljs");
const { runOCR } = require("./services/easyocrService");
const supabase   = require("./config/supabase");

const app = express();
app.use(cors());
app.use(express.json());

if (!fs.existsSync("uploads")) fs.mkdirSync("uploads");

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads"),
  filename:    (req, file, cb) =>
      cb(null, Date.now() + "-" + file.originalname),
});
const upload = multer({ storage });

async function withCompany(company_id) {
  await supabase.rpc("set_config", {
    setting: "app.current_company_id",
    value:   company_id,
  });
}

function authMiddleware(req, res, next) {
  const token =
    req.headers.authorization?.split(" ")[1] ||
    req.query.token;
  if (!token) return res.status(401).json({ error: "No token" });
  try {
    req.user = jwt.verify(token, process.env.JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: "Invalid token" });
  }
}

// ═══════════════════════════════════════════════════════════
// AUTH
// ═══════════════════════════════════════════════════════════

app.post("/auth/register", async (req, res) => {
  const {
    company_name, company_slug, email,
    password, plan = "starter", phone = "",
  } = req.body;

  const planLimits = {
    starter:      { max_projects: 3,      max_users: 4      },
    professional: { max_projects: 999,    max_users: 6      },
    enterprise:   { max_projects: 999999, max_users: 999999 },
  };
  const limits = planLimits[plan] || planLimits.starter;

  try {
    if (!company_name || !company_slug || !email || !password) {
      return res.status(400).json({ error: "Please fill all required fields" });
    }

    const { data: existingCompany } = await supabase
      .from("companies").select("id").eq("slug", company_slug).single();
    if (existingCompany) {
      return res.status(400).json({
        error: "This company ID is already taken. Please choose another."
      });
    }

    const { data: existingUser } = await supabase
      .from("users").select("id").eq("email", email).single();
    if (existingUser) {
      return res.status(400).json({ error: "Email already registered" });
    }

    const { data: company, error: companyErr } = await supabase
      .from("companies")
      .insert([{
        name: company_name, slug: company_slug, plan,
        max_projects: limits.max_projects,
        max_users:    limits.max_users,
        plan_expires_at: new Date(
          Date.now() + 14 * 24 * 60 * 60 * 1000
        ).toISOString(),
      }])
      .select().single();
    if (companyErr) return res.status(400).json({ error: companyErr.message });

    const password_hash = await bcrypt.hash(password, 10);
    const { data: user, error: userErr } = await supabase
      .from("users")
      .insert([{
        company_id: company.id, email, password_hash,
        role: "admin", active: true,
        phone: phone || "", full_name: email.split("@")[0],
      }])
      .select().single();

    if (userErr) {
      await supabase.from("companies").delete().eq("id", company.id);
      return res.status(400).json({ error: userErr.message });
    }

    await supabase.from("company_settings").insert([{ company_id: company.id }]);

    await supabase.from("material_configs").insert([{
      company_id: company.id,
      name: "Standard Red Brick",
      brick_length_m: 0.19, brick_width_m: 0.09, brick_height_m: 0.09,
      mortar_ratio_cement: 1.0, mortar_ratio_sand: 5.0, is_default: true,
    }]);

    await supabase.from("formula_definitions").insert([
      { company_id: company.id, name: "brick_face_area",            expression: "0.75 * 0.25", description: "Standard brick face area in sqft (9×3 inch)",       variables: [], output_unit: "sqft",       is_system_default: true, active: true },
      { company_id: company.id, name: "buffer_percentage",          expression: "10",          description: "Extra buffer percentage for all materials",           variables: [], output_unit: "percentage", is_system_default: true, active: true },
      { company_id: company.id, name: "red_brick_thickness",        expression: "9",           description: "Walls with this thickness (inches) use Red Brick",    variables: [], output_unit: "inches",     is_system_default: true, active: true },
      { company_id: company.id, name: "white_cement_thickness",     expression: "4,6",         description: "Walls with these thicknesses use White Cement Block",  variables: [], output_unit: "inches",     is_system_default: true, active: true },
      { company_id: company.id, name: "thickness_multiplier_4inch", expression: "1.0",         description: "4 inch wall thickness multiplier",                    variables: [], output_unit: "multiplier", is_system_default: true, active: true },
      { company_id: company.id, name: "thickness_multiplier_6inch", expression: "1.5",         description: "6 inch wall thickness multiplier",                    variables: [], output_unit: "multiplier", is_system_default: true, active: true },
      { company_id: company.id, name: "thickness_multiplier_8inch", expression: "2.0",         description: "8 inch wall thickness multiplier",                    variables: [], output_unit: "multiplier", is_system_default: true, active: true },
      { company_id: company.id, name: "thickness_multiplier_9inch", expression: "2.25",        description: "9 inch wall thickness multiplier",                    variables: [], output_unit: "multiplier", is_system_default: true, active: true },
    ]);

    const token = jwt.sign(
      { user_id: user.id, company_id: company.id, role: user.role, plan: company.plan },
      process.env.JWT_SECRET, { expiresIn: "7d" }
    );

    return res.status(201).json({
      token,
      user:    { id: user.id, email: user.email, role: user.role, full_name: user.full_name, phone: user.phone },
      company: { id: company.id, name: company.name, slug: company.slug, plan: company.plan, max_projects: company.max_projects, max_users: company.max_users },
    });
  } catch (err) {
    console.error("REGISTER ERROR:", err);
    return res.status(500).json({ error: err.message || "Internal server error" });
  }
});

app.post("/auth/login", async (req, res) => {
  const { email, password, company_slug } = req.body;
  try {
    const { data: company } = await supabase
      .from("companies").select("id").eq("slug", company_slug).single();
    if (!company) return res.status(404).json({ error: "Company not found" });

    const { data: user } = await supabase
      .from("users").select("*")
      .eq("company_id", company.id).eq("email", email).single();
    if (!user) return res.status(401).json({ error: "Invalid credentials" });

    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) return res.status(401).json({ error: "Invalid credentials" });

    const token = jwt.sign(
      { user_id: user.id, company_id: company.id, role: user.role },
      process.env.JWT_SECRET, { expiresIn: "7d" }
    );
    res.json({ token, user: { ...user, password_hash: undefined } });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ═══════════════════════════════════════════════════════════
// PROFILE
// ═══════════════════════════════════════════════════════════

app.get("/profile", authMiddleware, async (req, res) => {
  const { user_id, company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("users")
    .select("id, email, role, full_name, phone, active, created_at")
    .eq("id", user_id).single();
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.patch("/profile", authMiddleware, async (req, res) => {
  const { user_id, company_id } = req.user;
  await withCompany(company_id);
  const { full_name, phone } = req.body;
  const { data, error } = await supabase
    .from("users")
    .update({ full_name, phone, updated_at: new Date() })
    .eq("id", user_id).eq("company_id", company_id)
    .select("id, email, role, full_name, phone").single();
  if (error) return res.status(500).json({ error: error.message });
  res.json({ success: true, user: data });
});

// ═══════════════════════════════════════════════════════════
// PROJECTS
// ═══════════════════════════════════════════════════════════

app.get("/projects", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("projects").select("*").eq("company_id", company_id)
    .order("created_at", { ascending: false });
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.post("/projects", authMiddleware, async (req, res) => {
  const { company_id, user_id } = req.user;
  await withCompany(company_id);
  const { name, description } = req.body;
  const { data, error } = await supabase
    .from("projects")
    .insert([{ company_id, owner_id: user_id, name, description }])
    .select().single();
  if (error) return res.status(500).json({ error: error.message });
  res.status(201).json(data);
});

app.get("/projects/:id", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("projects").select("*")
    .eq("id", req.params.id).eq("company_id", company_id).single();
  if (error) return res.status(404).json({ error: "Not found" });
  res.json(data);
});

// ═══════════════════════════════════════════════════════════
// FLOOR PLANS + OCR
// ═══════════════════════════════════════════════════════════

app.post(
  "/projects/:project_id/floor-plans",
  authMiddleware, upload.single("file"),
  async (req, res) => {
    const { company_id } = req.user;
    const { project_id } = req.params;
    try {
      const filePath = path.resolve(req.file.path);
      const fileType = req.file.mimetype;
      await withCompany(company_id);

      const { data: plan, error: planErr } = await supabase
        .from("floor_plans")
        .insert([{ company_id, project_id, file_url: req.file.path, file_type: fileType, ocr_status: "processing" }])
        .select().single();
      if (planErr) return res.status(500).json({ error: planErr.message });

      const ocrResult = await runOCR(filePath);
      await supabase.from("floor_plans")
        .update({ ocr_status: "done", raw_ocr_data: ocrResult })
        .eq("id", plan.id);

      // ── Build structural elements with length_m ────
      const totalAreaSqft = ocrResult.summary?.total_area_sqft || 0;
      const totalAreaSqM  = totalAreaSqft * 0.0929;
      const perimeterM    = 4 * Math.sqrt(totalAreaSqM);

      const extWalls = ocrResult.walls?.external || [];
      const intWalls = ocrResult.walls?.internal || [];
      const extCount = extWalls.reduce((s, w) => s + (w.count || 1), 0);
      const intCount = intWalls.reduce((s, w) => s + (w.count || 1), 0);
      const avgExtLenM = extCount > 0 ? perimeterM / extCount : 3.0;
      const avgIntLenM = totalAreaSqM > 0
        ? (totalAreaSqM / (intCount || 1)) / 3.0 : 2.5;

      const elements = [];

      extWalls.forEach((w) => {
        elements.push({
          company_id,
          floor_plan_id: plan.id,
          element_type:  "wall",
          length_m:      parseFloat(avgExtLenM.toFixed(3)),
          thickness_m:   w.thickness_inch * 0.0254,
          height_m:      w.height_ft      * 0.3048,
          metadata:      { type: "external", count: w.count },
        });
      });

      intWalls.forEach((w) => {
        elements.push({
          company_id,
          floor_plan_id: plan.id,
          element_type:  "wall",
          length_m:      parseFloat(avgIntLenM.toFixed(3)),
          thickness_m:   w.thickness_inch * 0.0254,
          height_m:      w.height_ft      * 0.3048,
          metadata:      { type: "internal", count: w.count },
        });
      });

      ocrResult.openings?.doors?.forEach((d) => {
        elements.push({
          company_id,
          floor_plan_id: plan.id,
          element_type:  "door",
          width_m:       d.size_ft.width  * 0.3048,
          height_m:      d.size_ft.height * 0.3048,
          metadata:      { count: d.count },
        });
      });

      ocrResult.openings?.windows?.forEach((w) => {
        elements.push({
          company_id,
          floor_plan_id: plan.id,
          element_type:  "window",
          width_m:       w.size_ft.width  * 0.3048,
          height_m:      w.size_ft.height * 0.3048,
          metadata:      { count: w.count },
        });
      });

      if (elements.length > 0) {
        await supabase.from("structural_elements").insert(elements);
      }

      res.json({ success: true, plan, data: ocrResult });
    } catch (err) {
      res.status(500).json({ success: false, error: err.message });
    }
  }
);

// ═══════════════════════════════════════════════════════════
// BRICK CALCULATION
// ═══════════════════════════════════════════════════════════

app.post("/projects/:project_id/calculate", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  const { project_id } = req.params;
  await withCompany(company_id);

  try {
    const { data: plans } = await supabase
      .from("floor_plans").select("id, raw_ocr_data")
      .eq("project_id", project_id).eq("company_id", company_id)
      .eq("ocr_status", "done").order("created_at", { ascending: false }).limit(1);

    if (!plans?.length) {
      return res.status(404).json({ error: "No OCR data found. Please upload a floor plan first." });
    }

    const { data: formulas } = await supabase
      .from("formula_definitions").select("name, expression")
      .eq("company_id", company_id).eq("active", true);

    const fMap = {};
    (formulas || []).forEach((f) => { fMap[f.name] = f.expression; });

    const BRICK_FACE_SQFT     = parseFloat(eval(fMap["brick_face_area"] || "0.75 * 0.25"));
    const BUFFER_PCT          = parseFloat(fMap["buffer_percentage"] || "10");
    const redBrickThicknesses = (fMap["red_brick_thickness"] || "9").split(",").map(Number);

    const { data: compSettings } = await supabase
      .from("company_settings")
      .select("default_cement_mix, default_plaster_thickness, default_sand_unit, wastage_pct")
      .eq("company_id", company_id).single();

    const defaultMix       = compSettings?.default_cement_mix        || "1:4";
    const defaultThickness = compSettings?.default_plaster_thickness || "18mm";
    const defaultSandUnit  = compSettings?.default_sand_unit         || "tons";

    const thicknessMultiplier = (inch) => {
      const key = `thickness_multiplier_${inch}inch`;
      if (fMap[key]) return parseFloat(fMap[key]);
      if (inch <= 4) return 1.0;
      if (inch <= 6) return 1.5;
      if (inch <= 8) return 2.0;
      return 2.25;
    };

    const thicknessInFeet = (inch) => {
      if (inch === 9) return 0.75;
      if (inch === 6) return 0.50;
      if (inch === 4) return 0.33;
      if (inch === 8) return 0.67;
      return inch / 12;
    };

    const ocr      = plans[0].raw_ocr_data;
    const walls    = ocr.walls    || {};
    const openings = ocr.openings || {};
    const summary  = ocr.summary  || {};
    const zones    = ocr.zones    || [];

    const totalAreaSqft = summary.total_area_sqft || 0;
    const totalAreaSqM  = totalAreaSqft * 0.0929;
    const perimeterFt   = 4 * Math.sqrt(totalAreaSqM) * 3.281;

    const internal     = walls.internal || [];
    const external     = walls.external || [];
    const extWallCount = external.reduce((s, w) => s + (w.count || 1), 0);
    const intWallCount = internal.reduce((s, w) => s + (w.count || 1), 0);
    const avgExtLenFt  = extWallCount > 0 ? perimeterFt / extWallCount : 10;
    const avgIntLenFt  = totalAreaSqft > 0 ? (totalAreaSqft / (intWallCount || 1)) / 10 : 8;

    const calcWallBricks = (wallList, isExternal) => {
      return wallList.map((wall) => {
        const nos        = wall.count || 1;
        const L          = isExternal ? avgExtLenFt : avgIntLenFt;
        const H          = wall.height_ft || 10;
        const thick      = wall.thickness_inch || 9;
        const B          = thicknessInFeet(thick);
        const multiplier = thicknessMultiplier(thick);

        const wallFaceSqft    = L * H;
        const wallVolumeCuFt  = L * B * H * nos;
        const wallVolumeCuM   = wallVolumeCuFt * 0.0283168;
        const bricksPerFace   = wallFaceSqft / BRICK_FACE_SQFT;
        const bricksRaw       = bricksPerFace * multiplier * nos;
        const bricksWithBuffer = Math.ceil(bricksRaw * (1 + BUFFER_PCT / 100));
        const brickType = redBrickThicknesses.includes(thick) ? "red_brick" : "white_cement";

        return {
          type:              isExternal ? "external" : "internal",
          brick_type:        brickType,
          description:       `${isExternal ? "Ext" : "Int"} Wall ${thick}"`,
          thickness_inch:    thick,
          thickness_ft:      B,
          multiplier,
          L:                 parseFloat(L.toFixed(2)),
          H, nos,
          wall_face_sqft:    parseFloat(wallFaceSqft.toFixed(2)),
          wall_volume_cuft:  parseFloat(wallVolumeCuFt.toFixed(3)),
          wall_volume_cum:   parseFloat(wallVolumeCuM.toFixed(4)),
          bricks_per_face:   Math.ceil(bricksPerFace),
          bricks_raw:        Math.ceil(bricksRaw),
          bricks_with_10pct: bricksWithBuffer,
        };
      });
    };

    const allBreakdown = [
      ...calcWallBricks(external, true),
      ...calcWallBricks(internal, false),
    ];

    const calcOpeningDeduction = (openingList) => {
      let totalDedBricks = 0;
      const items = [];
      for (const o of openingList) {
        const nos       = o.count || 1;
        const L         = o.size_ft?.width  || 0;
        const H         = o.size_ft?.height || 0;
        const faceSqft  = L * H * nos;
        const dedBricks = Math.ceil(faceSqft / BRICK_FACE_SQFT);
        totalDedBricks += dedBricks;
        items.push({ description: `${L}×${H}ft`, nos, face_sqft: parseFloat(faceSqft.toFixed(2)), bricks_deducted: dedBricks });
      }
      return { items, total_bricks: totalDedBricks };
    };

    const windowDed = calcOpeningDeduction(openings.windows || []);
    const doorDed   = calcOpeningDeduction(openings.doors   || []);
    const totalDed  = windowDed.total_bricks + doorDed.total_bricks;

    const grossBricks = allBreakdown.reduce((s, w) => s + w.bricks_raw, 0);
    const netBricks   = Math.max(0, grossBricks - totalDed);
    const finalBricks = Math.ceil(netBricks * (1 + BUFFER_PCT / 100));

    const redRows    = allBreakdown.filter((w) => w.brick_type === "red_brick");
    const whiteRows  = allBreakdown.filter((w) => w.brick_type === "white_cement");
    const redGross   = redRows.reduce((s, w) => s + w.bricks_raw, 0);
    const whiteGross = whiteRows.reduce((s, w) => s + w.bricks_raw, 0);
    const total      = redGross + whiteGross || 1;

    const redDed   = Math.ceil(totalDed * (redGross   / total));
    const whiteDed = Math.ceil(totalDed * (whiteGross / total));
    const redNet   = Math.max(0, redGross   - redDed);
    const whiteNet = Math.max(0, whiteGross - whiteDed);
    const redFinal   = Math.ceil(redNet   * (1 + BUFFER_PCT / 100));
    const whiteFinal = Math.ceil(whiteNet * (1 + BUFFER_PCT / 100));

    const totalGrossVolumeCuFt = allBreakdown.reduce((s, w) => s + w.wall_volume_cuft, 0);
    const totalDedVolumeCuFt   = [...(openings.windows || []), ...(openings.doors || [])].reduce((s, o) => {
      const nos = o.count || 1;
      const L   = o.size_ft?.width  || 0;
      const H   = o.size_ft?.height || 0;
      return s + (L * 0.625 * H * nos);
    }, 0);
    const netVolumeCuFt = Math.max(0, totalGrossVolumeCuFt - totalDedVolumeCuFt);
    const netVolumeCuM  = parseFloat((netVolumeCuFt * 0.0283168).toFixed(4));

    const cementMasterData = {
      "1:3": { "12mm": 2.6, "18mm": 3.5 },
      "1:4": { "12mm": 2.0, "18mm": 2.7 },
      "1:5": { "12mm": 1.7, "18mm": 2.3 },
      "1:6": { "12mm": 1.5, "18mm": 2.0 },
    };
    const sandMasterData = {
      "1:3": { "cum": 1.25, "tons": 2.1 },
      "1:4": { "cum": 1.35, "tons": 2.2 },
      "1:5": { "cum": 1.40, "tons": 2.3 },
      "1:6": { "cum": 1.50, "tons": 2.5 },
    };

    const cementSandCalc = {};
    for (const mix of ["1:3", "1:4", "1:5", "1:6"]) {
      cementSandCalc[mix] = {
        mix,
        cement_bags_12mm: parseFloat((netVolumeCuM * cementMasterData[mix]["12mm"]).toFixed(2)),
        cement_bags_18mm: parseFloat((netVolumeCuM * cementMasterData[mix]["18mm"]).toFixed(2)),
        sand_cum:         parseFloat((netVolumeCuM * sandMasterData[mix]["cum"]).toFixed(3)),
        sand_tons:        parseFloat((netVolumeCuM * sandMasterData[mix]["tons"]).toFixed(3)),
      };
    }

    const cementBags = defaultThickness === "12mm"
      ? cementSandCalc[defaultMix].cement_bags_12mm
      : cementSandCalc[defaultMix].cement_bags_18mm;
    const sandCuM  = cementSandCalc[defaultMix].sand_cum;
    const sandTons = cementSandCalc[defaultMix].sand_tons;

    const wallCementSand = allBreakdown.map((w) => {
      const volCuM = w.wall_volume_cum;
      return {
        description:      w.description,
        thickness_inch:   w.thickness_inch,
        nos:              w.nos,
        volume_cuft:      w.wall_volume_cuft,
        volume_cum:       volCuM,
        cement_bags_12mm: parseFloat((volCuM * cementMasterData[defaultMix]["12mm"]).toFixed(2)),
        cement_bags_18mm: parseFloat((volCuM * cementMasterData[defaultMix]["18mm"]).toFixed(2)),
        sand_cum:         parseFloat((volCuM * sandMasterData[defaultMix]["cum"]).toFixed(3)),
        sand_tons:        parseFloat((volCuM * sandMasterData[defaultMix]["tons"]).toFixed(3)),
      };
    });

    const result = {
      project_id, floor_plan_id: plans[0].id, company_id,
      buffer_pct: BUFFER_PCT, brick_face_sqft: BRICK_FACE_SQFT,

      formulas_used: {
        brick_face_area: BRICK_FACE_SQFT, buffer_pct: BUFFER_PCT,
        red_brick_thickness: redBrickThicknesses,
        default_mix: defaultMix, default_thickness: defaultThickness,
        multipliers: {
          "4inch": thicknessMultiplier(4), "6inch": thicknessMultiplier(6),
          "8inch": thicknessMultiplier(8), "9inch": thicknessMultiplier(9),
        },
      },

      zone_summary: zones.map((z) => ({ name: z.name, zone_id: z.zone_id, area_sqft: z.area_sqft, size: z.size })),
      wall_breakdown: allBreakdown,

      deductions: {
        windows: windowDed, doors: doorDed,
        total_bricks_deducted: totalDed,
      },

      red_brick: {
        label: "Red Brick", walls: redRows,
        gross_bricks: Math.ceil(redGross), deducted: redDed,
        net_bricks: redNet, final_with_10pct: redFinal,
      },

      white_cement: {
        label: "White Cement Block", walls: whiteRows,
        gross_bricks: Math.ceil(whiteGross), deducted: whiteDed,
        net_bricks: whiteNet, final_with_10pct: whiteFinal,
      },

      grand_total: {
        gross_bricks: Math.ceil(grossBricks), total_deducted: totalDed,
        net_bricks: netBricks, final_bricks: finalBricks,
        red_bricks: redFinal, white_cement_blocks: whiteFinal,
        formula: `(${Math.ceil(grossBricks)} gross) - (${totalDed} openings) = ${netBricks} net + ${BUFFER_PCT}% = ${finalBricks} total`,
      },

      volume_summary: {
        gross_volume_cuft: parseFloat(totalGrossVolumeCuFt.toFixed(3)),
        deduction_cuft:    parseFloat(totalDedVolumeCuFt.toFixed(3)),
        net_volume_cuft:   parseFloat(netVolumeCuFt.toFixed(3)),
        net_volume_cum:    netVolumeCuM,
      },

      cement: {
        default_mix: defaultMix, default_thickness: defaultThickness,
        total_bags:  cementBags, all_mixes: cementSandCalc,
        per_wall:    wallCementSand,
        note: "Based on client Master Data — bags per m³ of brickwork",
      },

      sand: {
        default_mix: defaultMix, total_cum: sandCuM, total_tons: sandTons,
        all_mixes: Object.fromEntries(
          Object.entries(cementSandCalc).map(([k, v]) => [k, { sand_cum: v.sand_cum, sand_tons: v.sand_tons }])
        ),
        per_wall: wallCementSand,
        note: "Based on client Master Data — m³/tons per m³ of brickwork",
      },

      ocr_summary: summary,
    };

    // ── Save — no duplicates ──────────────────────────
    await withCompany(company_id);
    const { data: existingEst } = await supabase
      .from("material_estimations").select("id")
      .eq("project_id", project_id).eq("company_id", company_id).limit(1);

    if (!existingEst || existingEst.length === 0) {
      await supabase.from("material_estimations").insert([{
        company_id,
        project_id,
        total_volume_m3:  netVolumeCuM,
        total_bricks:     finalBricks,
        total_cement_kg:  parseFloat((cementBags * 50).toFixed(2)),
        total_sand_kg:    parseFloat((sandTons * 1000).toFixed(2)),
        formula_snapshot: result,
      }]);
    }

    // ── Always return result ──────────────────────────
    res.json({ success: true, ...result });

  } catch (err) {
    console.error("Calculation error:", err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// ═══════════════════════════════════════════════════════════
// MATERIAL CONFIGS
// ═══════════════════════════════════════════════════════════

app.get("/material-configs", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("material_configs").select("*").eq("company_id", company_id);
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.post("/material-configs", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("material_configs")
    .insert([{ company_id, ...req.body }]).select().single();
  if (error) return res.status(500).json({ error: error.message });
  res.status(201).json(data);
});

// ═══════════════════════════════════════════════════════════
// FORMULAS
// ═══════════════════════════════════════════════════════════

app.post("/formulas/seed", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  try {
    const { data: existing } = await supabase
      .from("formula_definitions").select("id")
      .eq("company_id", company_id).limit(1);

    if (existing?.length > 0) {
      return res.json({ success: true, message: "Formulas already exist for this company" });
    }

    const { data, error } = await supabase.from("formula_definitions").insert([
      { company_id, name: "brick_face_area",            expression: "0.75 * 0.25", description: "Standard brick face area in sqft (9×3 inch)",       variables: [], output_unit: "sqft",       is_system_default: true, active: true },
      { company_id, name: "buffer_percentage",          expression: "10",          description: "Extra buffer percentage for all materials",           variables: [], output_unit: "percentage", is_system_default: true, active: true },
      { company_id, name: "red_brick_thickness",        expression: "9",           description: "Walls with this thickness (inches) use Red Brick",    variables: [], output_unit: "inches",     is_system_default: true, active: true },
      { company_id, name: "white_cement_thickness",     expression: "4,6",         description: "Walls with these thicknesses use White Cement Block",  variables: [], output_unit: "inches",     is_system_default: true, active: true },
      { company_id, name: "thickness_multiplier_4inch", expression: "1.0",         description: "4 inch wall thickness multiplier",                    variables: [], output_unit: "multiplier", is_system_default: true, active: true },
      { company_id, name: "thickness_multiplier_6inch", expression: "1.5",         description: "6 inch wall thickness multiplier",                    variables: [], output_unit: "multiplier", is_system_default: true, active: true },
      { company_id, name: "thickness_multiplier_8inch", expression: "2.0",         description: "8 inch wall thickness multiplier",                    variables: [], output_unit: "multiplier", is_system_default: true, active: true },
      { company_id, name: "thickness_multiplier_9inch", expression: "2.25",        description: "9 inch wall thickness multiplier",                    variables: [], output_unit: "multiplier", is_system_default: true, active: true },
    ]).select();

    if (error) return res.status(500).json({ error: error.message });
    res.json({ success: true, message: `${data.length} formulas created`, formulas: data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/formulas", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("formula_definitions").select("*")
    .eq("company_id", company_id).eq("active", true).order("name");
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.patch("/formulas/:id", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { expression, description } = req.body;
  const { data, error } = await supabase
    .from("formula_definitions")
    .update({ expression, description })
    .eq("id", req.params.id).eq("company_id", company_id)
    .select().single();
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

// ═══════════════════════════════════════════════════════════
// ESTIMATIONS
// ═══════════════════════════════════════════════════════════

app.get("/projects/:project_id/estimations", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("material_estimations").select("*, estimation_details(*)")
    .eq("project_id", req.params.project_id).eq("company_id", company_id);
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.post("/projects/:project_id/estimations", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("material_estimations")
    .insert([{ company_id, project_id: req.params.project_id, ...req.body }])
    .select().single();
  if (error) return res.status(500).json({ error: error.message });
  res.status(201).json(data);
});

// ═══════════════════════════════════════════════════════════
// COMPANY SETTINGS
// ═══════════════════════════════════════════════════════════

app.get("/settings", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("company_settings").select("*")
    .eq("company_id", company_id).single();
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.patch("/settings", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("company_settings")
    .update({ ...req.body, updated_at: new Date() })
    .eq("company_id", company_id).select().single();
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

// ═══════════════════════════════════════════════════════════
// TEAM MANAGEMENT
// ═══════════════════════════════════════════════════════════

app.get("/team", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("users")
    .select("id, email, role, full_name, phone, active, created_at")
    .eq("company_id", company_id).order("created_at", { ascending: true });
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.post("/team/invite", authMiddleware, async (req, res) => {
  const { company_id, role: callerRole } = req.user;
  if (callerRole !== "admin") {
    return res.status(403).json({ error: "Only admins can invite users" });
  }
  const { email, password, role = "sub_user", full_name = "", phone = "" } = req.body;
  if (!email || !password) {
    return res.status(400).json({ error: "Email and password are required" });
  }
  await withCompany(company_id);
  try {
    const { data: company } = await supabase
      .from("companies").select("max_users").eq("id", company_id).single();
    const { data: currentUsers } = await supabase
      .from("users").select("id").eq("company_id", company_id);

    if (company?.max_users && currentUsers?.length >= company.max_users) {
      return res.status(400).json({
        error: `User limit reached. Your plan allows ${company.max_users} users. Upgrade to add more.`
      });
    }

    const { data: existing } = await supabase
      .from("users").select("id").eq("company_id", company_id).eq("email", email).single();
    if (existing) {
      return res.status(400).json({ error: "A user with this email already exists in your company" });
    }

    const password_hash = await bcrypt.hash(password, 10);
    const { data: user, error } = await supabase
      .from("users")
      .insert([{ company_id, email, password_hash, role, active: true, full_name: full_name || email.split("@")[0], phone }])
      .select("id, email, role, full_name, phone, active, created_at").single();

    if (error) return res.status(500).json({ error: error.message });
    res.status(201).json({ success: true, user });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.patch("/team/:user_id/toggle", authMiddleware, async (req, res) => {
  const { company_id, role: callerRole, user_id: callerId } = req.user;
  if (callerRole !== "admin") return res.status(403).json({ error: "Only admins can manage users" });
  if (req.params.user_id === callerId) return res.status(400).json({ error: "You cannot deactivate yourself" });
  await withCompany(company_id);
  const { data: current } = await supabase
    .from("users").select("active").eq("id", req.params.user_id).eq("company_id", company_id).single();
  if (!current) return res.status(404).json({ error: "User not found" });
  const { data, error } = await supabase
    .from("users").update({ active: !current.active })
    .eq("id", req.params.user_id).eq("company_id", company_id)
    .select("id, email, role, active").single();
  if (error) return res.status(500).json({ error: error.message });
  res.json({ success: true, user: data });
});

app.delete("/team/:user_id", authMiddleware, async (req, res) => {
  const { company_id, role: callerRole, user_id: callerId } = req.user;
  if (callerRole !== "admin") return res.status(403).json({ error: "Only admins can delete users" });
  if (req.params.user_id === callerId) return res.status(400).json({ error: "You cannot delete yourself" });
  await withCompany(company_id);
  const { error } = await supabase
    .from("users").delete().eq("id", req.params.user_id).eq("company_id", company_id);
  if (error) return res.status(500).json({ error: error.message });
  res.json({ success: true });
});

// ═══════════════════════════════════════════════════════════
// MASTER RATES
// ═══════════════════════════════════════════════════════════

app.post("/master-rates/seed", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  try {
    const { data: existing } = await supabase
      .from("master_rates").select("id").eq("company_id", company_id).limit(1);
    if (existing?.length > 0) {
      return res.json({ success: true, message: "Rates already exist for this company" });
    }
    const defaultRates = [
      { company_id, material: "Red Brick (9\")",          category: "Bricks",  rate: 8.20,   unit: "piece", gst_pct: 18, loading: 0.20, transport_km: 0, distance_km: 0, unloading: 0.47 },
      { company_id, material: "White Cement Block (6\")", category: "Bricks",  rate: 12.00,  unit: "piece", gst_pct: 18, loading: 0.20, transport_km: 0, distance_km: 0, unloading: 0.47 },
      { company_id, material: "White Cement Block (4\")", category: "Bricks",  rate: 10.00,  unit: "piece", gst_pct: 18, loading: 0.20, transport_km: 0, distance_km: 0, unloading: 0.47 },
      { company_id, material: "Cement (1:3 CM)",          category: "Cement",  rate: 380.00, unit: "bag",   gst_pct: 18, loading: 3.00, transport_km: 0, distance_km: 0, unloading: 1.50 },
      { company_id, material: "Cement (1:4 CM)",          category: "Cement",  rate: 380.00, unit: "bag",   gst_pct: 18, loading: 3.00, transport_km: 0, distance_km: 0, unloading: 1.50 },
      { company_id, material: "Cement (1:5 CM)",          category: "Cement",  rate: 380.00, unit: "bag",   gst_pct: 18, loading: 3.00, transport_km: 0, distance_km: 0, unloading: 1.50 },
      { company_id, material: "Cement (1:6 CM)",          category: "Cement",  rate: 380.00, unit: "bag",   gst_pct: 18, loading: 3.00, transport_km: 0, distance_km: 0, unloading: 1.50 },
      { company_id, material: "River Sand",               category: "Sand",    rate: 600.00, unit: "ton",   gst_pct: 5,  loading: 15.00, transport_km: 0, distance_km: 0, unloading: 0 },
      { company_id, material: "M-Sand",                   category: "Sand",    rate: 450.00, unit: "ton",   gst_pct: 5,  loading: 15.00, transport_km: 0, distance_km: 0, unloading: 0 },
      { company_id, material: "Mason (Skilled)",          category: "Labour",  rate: 800.00, unit: "day",   gst_pct: 0,  loading: 0, transport_km: 0, distance_km: 0, unloading: 0 },
      { company_id, material: "Helper (Unskilled)",       category: "Labour",  rate: 500.00, unit: "day",   gst_pct: 0,  loading: 0, transport_km: 0, distance_km: 0, unloading: 0 },
    ];
    const { data, error } = await supabase.from("master_rates").insert(defaultRates).select();
    if (error) return res.status(500).json({ error: error.message });
    res.json({ success: true, message: `${data.length} default rates created`, data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/master-rates", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("master_rates").select("*")
    .eq("company_id", company_id).eq("active", true)
    .order("category").order("material");
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.patch("/master-rates/:id", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { rate, gst_pct, loading, transport_km, distance_km, unloading, notes } = req.body;
  const { data, error } = await supabase
    .from("master_rates")
    .update({ rate, gst_pct, loading, transport_km, distance_km, unloading, notes, updated_at: new Date() })
    .eq("id", req.params.id).eq("company_id", company_id)
    .select().single();
  if (error) return res.status(500).json({ error: error.message });
  res.json({ success: true, data });
});

app.post("/master-rates", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { data, error } = await supabase
    .from("master_rates").insert([{ company_id, ...req.body }]).select().single();
  if (error) return res.status(500).json({ error: error.message });
  res.status(201).json({ success: true, data });
});

app.delete("/master-rates/:id", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  await withCompany(company_id);
  const { error } = await supabase
    .from("master_rates").delete()
    .eq("id", req.params.id).eq("company_id", company_id);
  if (error) return res.status(500).json({ error: error.message });
  res.json({ success: true });
});

// ═══════════════════════════════════════════════════════════
// REVIEW & BUDGET
// ═══════════════════════════════════════════════════════════

app.get("/projects/:project_id/review", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  const { project_id } = req.params;
  await withCompany(company_id);

  try {
    const { data: estimations } = await supabase
      .from("material_estimations").select("*")
      .eq("project_id", project_id).eq("company_id", company_id)
      .order("created_at", { ascending: false }).limit(1);

    if (!estimations?.length) {
      return res.status(404).json({ error: "No estimation found. Please run Costing first." });
    }

    const { data: rates } = await supabase
      .from("master_rates").select("*")
      .eq("company_id", company_id).eq("active", true);

    if (!rates?.length) {
      return res.status(404).json({ error: "No master rates found. Please setup Master List first." });
    }

    const snap = estimations[0].formula_snapshot;

    const findRate = (category, keyword) =>
      rates.find(r => r.category === category && r.material.toLowerCase().includes(keyword.toLowerCase()));

    const costAtSite = (r) => {
      if (!r) return 0;
      const base      = parseFloat(r.rate)         || 0;
      const gst       = base * (parseFloat(r.gst_pct) || 0) / 100;
      const loading   = parseFloat(r.loading)      || 0;
      const transport = parseFloat(r.transport_km) || 0;
      const unloading = parseFloat(r.unloading)    || 0;
      return base + gst + loading + transport + unloading;
    };

    const redBricks   = snap?.red_brick?.final_with_10pct    || 0;
    const whiteBricks = snap?.white_cement?.final_with_10pct || 0;
    const cementBags  = snap?.cement?.total_bags             || 0;
    const sandTons    = snap?.sand?.total_tons               || 0;
    const volCuM      = snap?.volume_summary?.net_volume_cum || 0;

    const redRate    = findRate('Bricks', 'Red Brick');
    const whiteRate  = findRate('Bricks', 'White Cement');
    const cementRate = findRate('Cement', '1:4');
    const sandRate   = findRate('Sand',   'River Sand');
    const masonRate  = findRate('Labour', 'Mason');
    const helperRate = findRate('Labour', 'Helper');

    const redCostPerUnit   = costAtSite(redRate);
    const whiteCostPerUnit = costAtSite(whiteRate);
    const cementPerBag     = costAtSite(cementRate);
    const sandPerTon       = costAtSite(sandRate);

    const redBrickCost    = redBricks   * redCostPerUnit;
    const whiteBrickCost  = whiteBricks * whiteCostPerUnit;
    const totalBrickCost  = redBrickCost + whiteBrickCost;
    const totalCementCost = cementBags  * cementPerBag;
    const totalSandCost   = sandTons    * sandPerTon;

    const masonDays       = Math.ceil(volCuM / 10 * 30);
    const helperDays      = masonDays;
    const masonCost       = masonDays  * (parseFloat(masonRate?.rate)  || 800);
    const helperCost      = helperDays * (parseFloat(helperRate?.rate) || 500);
    const totalLabourCost = masonCost + helperCost;

    const totalMaterialCost = totalBrickCost + totalCementCost + totalSandCost;
    const totalCost         = totalMaterialCost + totalLabourCost;

    const breakdown = [
      { category: 'Red Bricks',          qty: redBricks,                         unit: 'pieces',   rate: parseFloat(redCostPerUnit.toFixed(2)),   total: parseFloat(redBrickCost.toFixed(2)),    pct: totalCost > 0 ? parseFloat((redBrickCost    / totalCost * 100).toFixed(1)) : 0, color: 'red'    },
      { category: 'White Cement Blocks', qty: whiteBricks,                       unit: 'pieces',   rate: parseFloat(whiteCostPerUnit.toFixed(2)), total: parseFloat(whiteBrickCost.toFixed(2)),  pct: totalCost > 0 ? parseFloat((whiteBrickCost  / totalCost * 100).toFixed(1)) : 0, color: 'blue'   },
      { category: 'Cement',              qty: parseFloat(cementBags.toFixed(2)), unit: 'bags',     rate: parseFloat(cementPerBag.toFixed(2)),     total: parseFloat(totalCementCost.toFixed(2)), pct: totalCost > 0 ? parseFloat((totalCementCost / totalCost * 100).toFixed(1)) : 0, color: 'blue'   },
      { category: 'Sand',                qty: parseFloat(sandTons.toFixed(2)),   unit: 'tons',     rate: parseFloat(sandPerTon.toFixed(2)),       total: parseFloat(totalSandCost.toFixed(2)),   pct: totalCost > 0 ? parseFloat((totalSandCost   / totalCost * 100).toFixed(1)) : 0, color: 'teal'   },
      { category: 'Labour',              qty: masonDays + helperDays,            unit: 'man-days', rate: parseFloat(((masonCost + helperCost) / (masonDays + helperDays || 1)).toFixed(2)), total: parseFloat(totalLabourCost.toFixed(2)), pct: totalCost > 0 ? parseFloat((totalLabourCost / totalCost * 100).toFixed(1)) : 0, color: 'purple' },
    ];

    res.json({
      success: true, project_id, generated_at: new Date().toISOString(),
      quantities: { red_bricks: redBricks, white_bricks: whiteBricks, cement_bags: parseFloat(cementBags.toFixed(2)), sand_tons: parseFloat(sandTons.toFixed(2)), volume_cum: volCuM, mason_days: masonDays, helper_days: helperDays },
      rates_used: { red_brick_per_piece: parseFloat(redCostPerUnit.toFixed(2)), white_brick_per_piece: parseFloat(whiteCostPerUnit.toFixed(2)), cement_per_bag: parseFloat(cementPerBag.toFixed(2)), sand_per_ton: parseFloat(sandPerTon.toFixed(2)), mason_per_day: parseFloat(masonRate?.rate || 800), helper_per_day: parseFloat(helperRate?.rate || 500) },
      cost_summary: { bricks: parseFloat(totalBrickCost.toFixed(2)), cement: parseFloat(totalCementCost.toFixed(2)), sand: parseFloat(totalSandCost.toFixed(2)), labour: parseFloat(totalLabourCost.toFixed(2)), materials: parseFloat(totalMaterialCost.toFixed(2)), total: parseFloat(totalCost.toFixed(2)) },
      breakdown,
      note: "Labour estimated at 1 mason + 1 helper per 10m³ per day × 30 days",
    });
  } catch (err) {
    console.error("Review error:", err);
    res.status(500).json({ error: err.message });
  }
});

// ═══════════════════════════════════════════════════════════
// TAKEOFF (QTO)
// ═══════════════════════════════════════════════════════════

app.get("/projects/:project_id/takeoff", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  const { project_id } = req.params;
  await withCompany(company_id);

  try {
    const { data: plans } = await supabase
      .from("floor_plans").select("id, raw_ocr_data, created_at")
      .eq("project_id", project_id).eq("company_id", company_id)
      .eq("ocr_status", "done").order("created_at", { ascending: false }).limit(1);

    if (!plans?.length) {
      return res.status(404).json({ error: "No OCR data found. Please upload a floor plan first." });
    }

    const { data: estimations } = await supabase
      .from("material_estimations").select("*")
      .eq("project_id", project_id).eq("company_id", company_id)
      .order("created_at", { ascending: false }).limit(1);

    const snap = estimations?.[0]?.formula_snapshot || {};
    const ocr  = plans[0].raw_ocr_data;

    const walls    = ocr.walls    || {};
    const openings = ocr.openings || {};
    const summary  = ocr.summary  || {};
    const zones    = ocr.zones    || [];
    const internal = walls.internal || [];
    const external = walls.external || [];

    const totalAreaSqft = summary.total_area_sqft || 0;
    const totalAreaSqM  = parseFloat((totalAreaSqft * 0.0929).toFixed(3));
    const perimeterM    = parseFloat((4 * Math.sqrt(totalAreaSqM)).toFixed(2));

    const thicknessInFeet = (inch) => {
      if (inch === 9) return 0.75;
      if (inch === 6) return 0.50;
      if (inch === 4) return 0.33;
      if (inch === 8) return 0.67;
      return inch / 12;
    };

    const avgExtLenFt = external.length > 0
      ? (perimeterM * 3.281) / external.reduce((s, w) => s + (w.count || 1), 0) : 10;
    const avgIntLenFt = internal.length > 0
      ? (totalAreaSqft / internal.reduce((s, w) => s + (w.count || 1), 0)) / 10 : 8;

    const buildWallRows = (wallList, isExt) =>
      wallList.map(w => {
        const nos     = w.count || 1;
        const L       = isExt ? avgExtLenFt : avgIntLenFt;
        const H       = w.height_ft || 10;
        const B       = thicknessInFeet(w.thickness_inch || 9);
        const volCuFt = parseFloat((L * B * H * nos).toFixed(3));
        const volCuM  = parseFloat((volCuFt * 0.0283168).toFixed(4));
        return { description: `${isExt ? 'Ext' : 'Int'} Wall ${w.thickness_inch || 9}"`, nos, L: parseFloat(L.toFixed(2)), B, H, qty_cuft: volCuFt, qty_cum: volCuM, unit: 'Cu.Ft', type: isExt ? 'external' : 'internal' };
      });

    const allWallRows        = [...buildWallRows(external, true), ...buildWallRows(internal, false)];
    const totalBrickworkCuFt = parseFloat(allWallRows.reduce((s, w) => s + w.qty_cuft, 0).toFixed(3));
    const totalBrickworkCuM  = parseFloat((totalBrickworkCuFt * 0.0283168).toFixed(4));

    const windowRows = (openings.windows || []).map(w => ({
      description: `Window ${w.size_ft?.width || 0}×${w.size_ft?.height || 0}ft`,
      nos: w.count || 1, width_ft: w.size_ft?.width || 0, height_ft: w.size_ft?.height || 0,
      area_sqft: parseFloat(((w.size_ft?.width || 0) * (w.size_ft?.height || 0) * (w.count || 1)).toFixed(2)),
      unit: 'Sqft', type: 'window',
    }));
    const doorRows = (openings.doors || []).map(d => ({
      description: `Door ${d.size_ft?.width || 0}×${d.size_ft?.height || 0}ft`,
      nos: d.count || 1, width_ft: d.size_ft?.width || 0, height_ft: d.size_ft?.height || 0,
      area_sqft: parseFloat(((d.size_ft?.width || 0) * (d.size_ft?.height || 0) * (d.count || 1)).toFixed(2)),
      unit: 'Sqft', type: 'door',
    }));

    const totalWindowArea  = windowRows.reduce((s, w) => s + w.area_sqft, 0);
    const totalDoorArea    = doorRows.reduce((s, d)   => s + d.area_sqft, 0);
    const wallSurfaceArea  = parseFloat(allWallRows.reduce((s, w) => s + (w.L * w.H * w.nos), 0).toFixed(2));
    const plasterArea      = parseFloat(Math.max(0, wallSurfaceArea - totalWindowArea - totalDoorArea).toFixed(2));
    const electricalPoints = Math.ceil(totalAreaSqft / 50);
    const plumbingPoints   = Math.ceil(totalAreaSqft / 100);

    res.json({
      success: true, project_id, floor_plan_id: plans[0].id, generated_at: new Date().toISOString(),
      summary: { total_area_sqft: totalAreaSqft, total_area_sqm: totalAreaSqM, perimeter_m: perimeterM, total_walls: allWallRows.length, total_windows: windowRows.length, total_doors: doorRows.length, total_zones: zones.length },
      tabs: {
        brickwork: { label: 'Brickwork', rows: allWallRows, totals: { total_cuft: totalBrickworkCuFt, total_cum: totalBrickworkCuM } },
        sitework:  { label: 'Sitework',  rows: [
          { description: 'Earth Work Excavation',      nos: 1, qty: parseFloat((totalAreaSqM * 0.6).toFixed(3)),         unit: 'Cu.M', notes: 'Estimated at 0.6m depth' },
          { description: 'PCC 1:4:8 Under Foundation', nos: 1, qty: parseFloat((totalAreaSqM * 0.15).toFixed(3)),        unit: 'Cu.M', notes: '150mm thick' },
          { description: 'Backfilling',                nos: 1, qty: parseFloat((totalAreaSqM * 0.2).toFixed(3)),         unit: 'Cu.M', notes: '1/3 of excavation' },
          { description: 'Sand Filling in Plinth',     nos: 1, qty: parseFloat((totalAreaSqM * 0.3).toFixed(3)),         unit: 'Cu.M', notes: '300mm thick bed' },
        ]},
        structure: { label: 'Structure', rows: [
          { description: 'RCC M20 (Columns + Beams)', nos: 1, qty: parseFloat((totalAreaSqM * 0.08).toFixed(3)),         unit: 'Cu.M', notes: 'Estimated 8% of floor area' },
          { description: 'RCC Slab (150mm)',           nos: 1, qty: parseFloat((totalAreaSqM * 0.15).toFixed(3)),         unit: 'Cu.M', notes: '150mm thick' },
          { description: 'Steel Reinforcement',        nos: 1, qty: parseFloat((totalAreaSqM * 0.08 * 120).toFixed(1)), unit: 'Kg',   notes: '120 kg/m³ of RCC' },
          { description: 'Brickwork (Total)',          nos: 1, qty: totalBrickworkCuFt,                                   unit: 'Cu.Ft',notes: 'From OCR floor plan' },
        ]},
        finishing: { label: 'Finishing', rows: [
          { description: 'Wall Plastering (12mm)', nos: 2, qty: parseFloat((plasterArea * 2).toFixed(2)), unit: 'Sqft', notes: 'Both sides of walls' },
          { description: 'Ceiling Plastering',     nos: 1, qty: parseFloat(totalAreaSqft.toFixed(2)),     unit: 'Sqft', notes: 'Total floor area' },
          { description: 'Flooring (Vitrified)',   nos: 1, qty: parseFloat(totalAreaSqft.toFixed(2)),     unit: 'Sqft', notes: 'Total floor area' },
          { description: 'White Wash / Paint',     nos: 2, qty: parseFloat((plasterArea * 2).toFixed(2)), unit: 'Sqft', notes: '2 coats' },
        ]},
        mep: { label: 'MEP', rows: [
          { description: 'Electrical Points', nos: 1, qty: electricalPoints,                unit: 'Points', notes: 'Estimated 1 per 50 sqft' },
          { description: 'Plumbing Points',   nos: 1, qty: plumbingPoints,                  unit: 'Points', notes: 'Estimated 1 per 100 sqft' },
          { description: 'Drainage Points',   nos: 1, qty: Math.ceil(plumbingPoints * 0.8), unit: 'Points', notes: '80% of plumbing points' },
        ]},
        openings: { label: 'Openings', windows: windowRows, doors: doorRows, totals: { window_area_sqft: parseFloat(totalWindowArea.toFixed(2)), door_area_sqft: parseFloat(totalDoorArea.toFixed(2)), total_area_sqft: parseFloat((totalWindowArea + totalDoorArea).toFixed(2)) } },
      },
      zones: zones.map(z => ({ name: z.name, area_sqft: z.area_sqft || 0, area_sqm: parseFloat(((z.area_sqft || 0) * 0.0929).toFixed(2)), pct_of_total: totalAreaSqft > 0 ? parseFloat(((z.area_sqft || 0) / totalAreaSqft * 100).toFixed(1)) : 0, size: z.size || {} })),
      ocr_summary: summary,
    });
  } catch (err) {
    console.error("Takeoff error:", err);
    res.status(500).json({ error: err.message });
  }
});

// ═══════════════════════════════════════════════════════════
// EXPORT — PDF BOQ
// ═══════════════════════════════════════════════════════════

app.get("/projects/:project_id/export/pdf", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  const { project_id } = req.params;
  await withCompany(company_id);

  try {
    const { data: project }     = await supabase.from("projects").select("name, description").eq("id", project_id).eq("company_id", company_id).single();
    const { data: company }     = await supabase.from("companies").select("name").eq("id", company_id).single();
    const { data: estimations } = await supabase.from("material_estimations").select("*").eq("project_id", project_id).eq("company_id", company_id).order("created_at", { ascending: false }).limit(1);
    if (!estimations?.length) return res.status(404).json({ error: "No estimation found. Run Costing first." });
    const { data: rates } = await supabase.from("master_rates").select("*").eq("company_id", company_id).eq("active", true);

    const snap = estimations[0].formula_snapshot;

    const costAtSite = (r) => {
      if (!r) return 0;
      const base      = parseFloat(r.rate)         || 0;
      const gst       = base * (parseFloat(r.gst_pct) || 0) / 100;
      const loading   = parseFloat(r.loading)      || 0;
      const transport = parseFloat(r.transport_km) || 0;
      const unloading = parseFloat(r.unloading)    || 0;
      return base + gst + loading + transport + unloading;
    };

    const findRate = (category, keyword) =>
      (rates || []).find(r => r.category === category && r.material.toLowerCase().includes(keyword.toLowerCase()));

    const redBricks   = snap?.red_brick?.final_with_10pct    || 0;
    const whiteBricks = snap?.white_cement?.final_with_10pct || 0;
    const cementBags  = snap?.cement?.total_bags             || 0;
    const sandTons    = snap?.sand?.total_tons               || 0;
    const volCuM      = snap?.volume_summary?.net_volume_cum || 0;
    const masonDays   = Math.ceil(volCuM / 10 * 30);

    const redRate    = findRate('Bricks', 'Red Brick');
    const whiteRate  = findRate('Bricks', 'White Cement');
    const cementRate = findRate('Cement', '1:4');
    const sandRate   = findRate('Sand',   'River Sand');
    const masonRate  = findRate('Labour', 'Mason');
    const helperRate = findRate('Labour', 'Helper');

    const redCost    = redBricks   * costAtSite(redRate);
    const whiteCost  = whiteBricks * costAtSite(whiteRate);
    const cementCost = cementBags  * costAtSite(cementRate);
    const sandCost   = sandTons    * costAtSite(sandRate);
    const labourCost = (masonDays * (parseFloat(masonRate?.rate) || 800)) + (masonDays * (parseFloat(helperRate?.rate) || 500));
    const totalCost  = redCost + whiteCost + cementCost + sandCost + labourCost;

    const doc = new PDFDocument({ margin: 50, size: 'A4' });
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="BOQ_${project?.name || project_id}_${Date.now()}.pdf"`);
    doc.pipe(res);

    const primaryColor = '#1E6FD9';
    const darkColor    = '#0F172A';
    const grayColor    = '#64748B';
    const lightGray    = '#F1F5F9';

    doc.rect(0, 0, doc.page.width, 80).fill(darkColor);
    doc.fillColor('white').fontSize(20).font('Helvetica-Bold').text('ArchiQuant', 50, 20);
    doc.fillColor('#94A3B8').fontSize(9).font('Helvetica').text('BUILDING CONSTRUCTION SUITE', 50, 44);
    doc.fillColor('white').fontSize(14).font('Helvetica-Bold').text('BILL OF QUANTITIES (BOQ)', 50, 58);

    doc.fillColor(darkColor).fontSize(11).font('Helvetica-Bold').text(company?.name || 'Company', 50, 100);
    doc.fillColor(grayColor).fontSize(10).font('Helvetica').text(`Project: ${project?.name || 'Project'}`, 50, 116);
    doc.text(`Generated: ${new Date().toLocaleDateString('en-IN')}`, 50, 130);
    doc.text(`Report Type: Material Quantity & Cost BOQ`, 50, 144);

    doc.moveTo(50, 165).lineTo(doc.page.width - 50, 165).strokeColor(primaryColor).lineWidth(2).stroke();
    doc.rect(50, 175, doc.page.width - 100, 60).fill(lightGray);
    doc.fillColor(darkColor).fontSize(10).font('Helvetica-Bold').text('PROJECT SUMMARY', 60, 183);

    const summaryItems = [
      ['Floor Area',    `${(snap?.ocr_summary?.total_area_sqft || 0).toFixed(0)} Sqft`],
      ['Red Bricks',    `${Math.ceil(redBricks)} pcs`],
      ['Cement Blocks', `${Math.ceil(whiteBricks)} pcs`],
      ['Cement Bags',   `${parseFloat(cementBags).toFixed(1)} bags`],
      ['Sand',          `${parseFloat(sandTons).toFixed(2)} tons`],
    ];
    let sx = 60;
    summaryItems.forEach(([label, value]) => {
      doc.fillColor(grayColor).fontSize(8).font('Helvetica').text(label, sx, 197);
      doc.fillColor(primaryColor).fontSize(10).font('Helvetica-Bold').text(value, sx, 209);
      sx += 95;
    });

    let y = 255;
    doc.fillColor(darkColor).fontSize(12).font('Helvetica-Bold').text('DETAILED BILL OF QUANTITIES', 50, y);
    y += 20;

    const cols = [30, 200, 60, 80, 80, 80];
    const colX = [50];
    cols.forEach((w, i) => colX.push(colX[i] + w));

    doc.rect(50, y, doc.page.width - 100, 22).fill(darkColor);
    ['Sl.', 'Description', 'Qty', 'Unit', 'Rate (₹)', 'Amount (₹)'].forEach((h, i) => {
      doc.fillColor('white').fontSize(9).font('Helvetica-Bold')
         .text(h, colX[i] + 4, y + 7, { width: cols[i] - 8, align: i > 1 ? 'right' : 'left' });
    });
    y += 22;

    const boqRows = [
      ['01', 'Red Brick (9" Walls)',               Math.ceil(redBricks).toString(),    'Pieces', costAtSite(redRate).toFixed(2),    redCost.toFixed(2)],
      ['02', 'White Cement Block (4"/6")',          Math.ceil(whiteBricks).toString(),  'Pieces', costAtSite(whiteRate).toFixed(2),  whiteCost.toFixed(2)],
      ['03', 'Cement in CM 1:4 (18mm thk)',         parseFloat(cementBags).toFixed(2), 'Bags',   costAtSite(cementRate).toFixed(2), cementCost.toFixed(2)],
      ['04', 'River Sand',                          parseFloat(sandTons).toFixed(3),   'Tons',   costAtSite(sandRate).toFixed(2),   sandCost.toFixed(2)],
      ['05', `Mason (Skilled) — ${masonDays} days`,    masonDays.toString(),            'Days',   (parseFloat(masonRate?.rate) || 800).toFixed(2), (masonDays * (parseFloat(masonRate?.rate) || 800)).toFixed(2)],
      ['06', `Helper (Unskilled) — ${masonDays} days`, masonDays.toString(),            'Days',   (parseFloat(helperRate?.rate) || 500).toFixed(2), (masonDays * (parseFloat(helperRate?.rate) || 500)).toFixed(2)],
    ];

    boqRows.forEach((row, idx) => {
      doc.rect(50, y, doc.page.width - 100, 20).fill(idx % 2 === 0 ? 'white' : '#F8FAFC');
      row.forEach((cell, i) => {
        doc.fillColor(i === 1 ? darkColor : grayColor).fontSize(9).font(i === 1 ? 'Helvetica-Bold' : 'Helvetica')
           .text(cell, colX[i] + 4, y + 6, { width: cols[i] - 8, align: i > 1 ? 'right' : 'left' });
      });
      doc.moveTo(50, y + 20).lineTo(doc.page.width - 50, y + 20).strokeColor('#E2E8F0').lineWidth(0.5).stroke();
      y += 20;
    });

    doc.rect(50, y, doc.page.width - 100, 28).fill(primaryColor);
    doc.fillColor('white').fontSize(11).font('Helvetica-Bold').text('GRAND TOTAL', 54, y + 8);
    doc.fillColor('white').fontSize(12).font('Helvetica-Bold').text(`Rs. ${totalCost.toFixed(2)}`, colX[5] + 4, y + 8, { width: cols[5] - 8, align: 'right' });
    y += 40;

    doc.fillColor(grayColor).fontSize(9).font('Helvetica').text(`Amount in Lakhs: Rs. ${(totalCost / 100000).toFixed(2)} Lakhs`, 50, y);
    y += 30;

    doc.fillColor(darkColor).fontSize(11).font('Helvetica-Bold').text('RATES USED (incl. GST + Loading)', 50, y);
    y += 16;

    [
      ['Red Brick',          `Rs. ${costAtSite(redRate).toFixed(2)}/piece`],
      ['White Cement Block', `Rs. ${costAtSite(whiteRate).toFixed(2)}/piece`],
      ['Cement',             `Rs. ${costAtSite(cementRate).toFixed(2)}/bag`],
      ['Sand',               `Rs. ${costAtSite(sandRate).toFixed(2)}/ton`],
      ['Mason',              `Rs. ${parseFloat(masonRate?.rate || 800).toFixed(2)}/day`],
      ['Helper',             `Rs. ${parseFloat(helperRate?.rate || 500).toFixed(2)}/day`],
    ].forEach(([label, value]) => {
      doc.rect(50, y, doc.page.width - 100, 18).fill(y % 36 === 0 ? lightGray : 'white');
      doc.fillColor(grayColor).fontSize(9).font('Helvetica').text(label, 60, y + 5);
      doc.fillColor(primaryColor).fontSize(9).font('Helvetica-Bold').text(value, 200, y + 5);
      y += 18;
    });

    y += 20;
    doc.moveTo(50, y).lineTo(doc.page.width - 50, y).strokeColor('#E2E8F0').lineWidth(1).stroke();
    y += 10;
    doc.fillColor(grayColor).fontSize(8).font('Helvetica')
       .text('This BOQ is generated by ArchiQuant. Quantities are estimated from OCR floor plan analysis. Actual quantities may vary.', 50, y, { width: doc.page.width - 100, align: 'center' });

    doc.end();
  } catch (err) {
    console.error("PDF export error:", err);
    if (!res.headersSent) res.status(500).json({ error: err.message });
  }
});

// ═══════════════════════════════════════════════════════════
// EXPORT — EXCEL BOQ
// ═══════════════════════════════════════════════════════════

app.get("/projects/:project_id/export/excel", authMiddleware, async (req, res) => {
  const { company_id } = req.user;
  const { project_id } = req.params;
  await withCompany(company_id);

  try {
    const { data: project }     = await supabase.from("projects").select("name").eq("id", project_id).eq("company_id", company_id).single();
    const { data: company }     = await supabase.from("companies").select("name").eq("id", company_id).single();
    const { data: estimations } = await supabase.from("material_estimations").select("*").eq("project_id", project_id).eq("company_id", company_id).order("created_at", { ascending: false }).limit(1);
    if (!estimations?.length) return res.status(404).json({ error: "No estimation found." });
    const { data: rates } = await supabase.from("master_rates").select("*").eq("company_id", company_id).eq("active", true);

    const snap = estimations[0].formula_snapshot;

    const costAtSite = (r) => {
      if (!r) return 0;
      const base      = parseFloat(r.rate)         || 0;
      const gst       = base * (parseFloat(r.gst_pct) || 0) / 100;
      const loading   = parseFloat(r.loading)      || 0;
      const transport = parseFloat(r.transport_km) || 0;
      const unloading = parseFloat(r.unloading)    || 0;
      return base + gst + loading + transport + unloading;
    };

    const findRate = (category, keyword) =>
      (rates || []).find(r => r.category === category && r.material.toLowerCase().includes(keyword.toLowerCase()));

    const redBricks   = snap?.red_brick?.final_with_10pct    || 0;
    const whiteBricks = snap?.white_cement?.final_with_10pct || 0;
    const cementBags  = snap?.cement?.total_bags             || 0;
    const sandTons    = snap?.sand?.total_tons               || 0;
    const volCuM      = snap?.volume_summary?.net_volume_cum || 0;
    const masonDays   = Math.ceil(volCuM / 10 * 30);

    const redRate    = findRate('Bricks', 'Red Brick');
    const whiteRate  = findRate('Bricks', 'White Cement');
    const cementRate = findRate('Cement', '1:4');
    const sandRate   = findRate('Sand',   'River Sand');
    const masonRate  = findRate('Labour', 'Mason');
    const helperRate = findRate('Labour', 'Helper');

    const redCost    = redBricks   * costAtSite(redRate);
    const whiteCost  = whiteBricks * costAtSite(whiteRate);
    const cementCost = cementBags  * costAtSite(cementRate);
    const sandCost   = sandTons    * costAtSite(sandRate);
    const masonCost  = masonDays   * (parseFloat(masonRate?.rate)  || 800);
    const helperCost = masonDays   * (parseFloat(helperRate?.rate) || 500);
    const totalCost  = redCost + whiteCost + cementCost + sandCost + masonCost + helperCost;

    const wb = new ExcelJS.Workbook();
    wb.creator = 'ArchiQuant';
    wb.created = new Date();

    const boqSheet = wb.addWorksheet('BOQ', { pageSetup: { paperSize: 9, orientation: 'portrait' } });
    boqSheet.columns = [
      { key: 'sl', width: 6 }, { key: 'desc', width: 40 }, { key: 'qty', width: 14 },
      { key: 'unit', width: 12 }, { key: 'rate', width: 16 }, { key: 'amount', width: 18 },
    ];

    boqSheet.mergeCells('A1:F1');
    boqSheet.getCell('A1').value = 'ArchiQuant — Bill of Quantities (BOQ)';
    boqSheet.getCell('A1').font  = { bold: true, size: 16, color: { argb: 'FF1E6FD9' } };
    boqSheet.getCell('A1').alignment = { horizontal: 'center' };
    boqSheet.getRow(1).height = 30;

    boqSheet.mergeCells('A2:F2');
    boqSheet.getCell('A2').value = `Company: ${company?.name}  |  Project: ${project?.name}  |  Date: ${new Date().toLocaleDateString('en-IN')}`;
    boqSheet.getCell('A2').font      = { size: 10, color: { argb: 'FF64748B' } };
    boqSheet.getCell('A2').alignment = { horizontal: 'center' };
    boqSheet.getRow(2).height = 20;
    boqSheet.addRow([]);

    const headerRow = boqSheet.addRow(['Sl.No', 'Description', 'Quantity', 'Unit', 'Rate (₹)', 'Amount (₹)']);
    headerRow.height = 22;
    headerRow.eachCell(cell => {
      cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF0F172A' } };
      cell.font      = { bold: true, color: { argb: 'FFFFFFFF' }, size: 10 };
      cell.alignment = { horizontal: 'center', vertical: 'middle' };
      cell.border    = { top: { style: 'thin', color: { argb: 'FF1E6FD9' } }, bottom: { style: 'thin', color: { argb: 'FF1E6FD9' } }, left: { style: 'thin', color: { argb: 'FF1E6FD9' } }, right: { style: 'thin', color: { argb: 'FF1E6FD9' } } };
    });

    [
      ['01', 'Red Brick (9" Walls)',               Math.ceil(redBricks),              'Pieces', costAtSite(redRate).toFixed(2),    redCost.toFixed(2)],
      ['02', 'White Cement Block (4" / 6")',        Math.ceil(whiteBricks),            'Pieces', costAtSite(whiteRate).toFixed(2),  whiteCost.toFixed(2)],
      ['03', 'Cement in CM 1:4 (18mm plaster)',    parseFloat(cementBags).toFixed(2), 'Bags',   costAtSite(cementRate).toFixed(2), cementCost.toFixed(2)],
      ['04', 'River Sand',                          parseFloat(sandTons).toFixed(3),  'Tons',   costAtSite(sandRate).toFixed(2),   sandCost.toFixed(2)],
      ['05', `Mason (Skilled) — ${masonDays} days`,    masonDays, 'Days', (parseFloat(masonRate?.rate)  || 800).toFixed(2), masonCost.toFixed(2)],
      ['06', `Helper (Unskilled) — ${masonDays} days`, masonDays, 'Days', (parseFloat(helperRate?.rate) || 500).toFixed(2), helperCost.toFixed(2)],
    ].forEach((row, idx) => {
      const dataRow = boqSheet.addRow(row);
      dataRow.height = 18;
      const bgColor  = idx % 2 === 0 ? 'FFFFFFFF' : 'FFF8FAFC';
      dataRow.eachCell((cell, colNum) => {
        cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: bgColor } };
        cell.font      = { size: 10, bold: colNum === 2, color: { argb: colNum === 2 ? 'FF1E293B' : 'FF64748B' } };
        cell.alignment = { horizontal: colNum > 2 ? 'right' : 'left', vertical: 'middle' };
        cell.border    = { bottom: { style: 'thin', color: { argb: 'FFE2E8F0' } } };
      });
    });

    const totalRow = boqSheet.addRow(['', 'GRAND TOTAL', '', '', '', totalCost.toFixed(2)]);
    totalRow.height = 24;
    totalRow.eachCell(cell => {
      cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF1E6FD9' } };
      cell.font      = { bold: true, color: { argb: 'FFFFFFFF' }, size: 11 };
      cell.alignment = { horizontal: 'right', vertical: 'middle' };
    });
    boqSheet.getCell(`B${totalRow.number}`).alignment = { horizontal: 'left', vertical: 'middle' };
    boqSheet.addRow([]);
    boqSheet.mergeCells(`A${boqSheet.rowCount}:F${boqSheet.rowCount}`);
    boqSheet.getCell(`A${boqSheet.rowCount}`).value = `Total Amount: Rs. ${(totalCost / 100000).toFixed(2)} Lakhs`;
    boqSheet.getCell(`A${boqSheet.rowCount}`).font  = { italic: true, color: { argb: 'FF64748B' }, size: 9 };

    const brickSheet = wb.addWorksheet('Brick Calculation');
    brickSheet.columns = [
      { key: 'desc', width: 25 }, { key: 'type', width: 12 }, { key: 'nos', width: 8 },
      { key: 'L', width: 10 }, { key: 'B', width: 10 }, { key: 'H', width: 10 },
      { key: 'volCuft', width: 14 }, { key: 'volCuM', width: 14 }, { key: 'bricks', width: 14 },
    ];
    brickSheet.mergeCells('A1:I1');
    brickSheet.getCell('A1').value = 'Brick Work Quantity Calculation';
    brickSheet.getCell('A1').font  = { bold: true, size: 14, color: { argb: 'FFDC2626' } };
    brickSheet.getCell('A1').alignment = { horizontal: 'center' };
    brickSheet.getRow(1).height = 28;
    brickSheet.addRow([]);

    const brickHeader = brickSheet.addRow(['Description', 'Type', 'Nos', 'L (ft)', 'B (ft)', 'H (ft)', 'Vol (Cu.Ft)', 'Vol (m³)', '+10% Bricks']);
    brickHeader.height = 20;
    brickHeader.eachCell(cell => {
      cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFDC2626' } };
      cell.font      = { bold: true, color: { argb: 'FFFFFFFF' }, size: 10 };
      cell.alignment = { horizontal: 'center', vertical: 'middle' };
    });

    [...(snap?.red_brick?.walls || []), ...(snap?.white_cement?.walls || [])].forEach((w, idx) => {
      const volCuft = (w.L || 0) * (w.thickness_ft || w.B || 0) * (w.H || 0) * (w.nos || 1);
      const row = brickSheet.addRow([
        w.description || '', w.brick_type || w.type || '', w.nos || 1,
        (w.L || 0).toFixed(2), w.thickness_ft || w.B || 0, w.H || 0,
        (w.wall_volume_cuft || volCuft).toFixed(3),
        (w.wall_volume_cum  || volCuft * 0.0283168).toFixed(4),
        w.bricks_with_10pct || 0,
      ]);
      row.height = 16;
      row.eachCell(cell => {
        cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: idx % 2 === 0 ? 'FFFFFFFF' : 'FFFFF5F5' } };
        cell.font      = { size: 9 };
        cell.alignment = { horizontal: 'center' };
        cell.border    = { bottom: { style: 'thin', color: { argb: 'FFFFE2E2' } } };
      });
      brickSheet.getCell(`A${row.number}`).alignment = { horizontal: 'left' };
    });

    const brickTotalRow = brickSheet.addRow(['TOTAL', '', '', '', '', '',
      (snap?.volume_summary?.gross_volume_cuft || 0).toFixed(3),
      (snap?.volume_summary?.net_volume_cum    || 0).toFixed(4),
      snap?.grand_total?.final_bricks || 0,
    ]);
    brickTotalRow.height = 20;
    brickTotalRow.eachCell(cell => {
      cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFDC2626' } };
      cell.font      = { bold: true, color: { argb: 'FFFFFFFF' }, size: 10 };
      cell.alignment = { horizontal: 'center' };
    });

    const csSheet = wb.addWorksheet('Cement & Sand');
    csSheet.columns = [
      { key: 'mix', width: 12 }, { key: 'bags12', width: 18 }, { key: 'bags18', width: 18 },
      { key: 'sandCuM', width: 16 }, { key: 'sandTon', width: 16 },
    ];
    csSheet.mergeCells('A1:E1');
    csSheet.getCell('A1').value = 'Cement & Sand Quantities (Based on Client Master Data)';
    csSheet.getCell('A1').font  = { bold: true, size: 14, color: { argb: 'FF1E6FD9' } };
    csSheet.getCell('A1').alignment = { horizontal: 'center' };
    csSheet.getRow(1).height = 28;
    csSheet.addRow([]);
    csSheet.mergeCells('A3:E3');
    csSheet.getCell('A3').value = `Net Brickwork Volume: ${(snap?.volume_summary?.net_volume_cum || 0).toFixed(4)} m³`;
    csSheet.getCell('A3').font  = { italic: true, color: { argb: 'FF64748B' } };
    csSheet.addRow([]);

    const csHeader = csSheet.addRow(['Mix', 'Cement (12mm) bags', 'Cement (18mm) bags', 'Sand (m³)', 'Sand (Tons)']);
    csHeader.height = 20;
    csHeader.eachCell(cell => {
      cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF1E6FD9' } };
      cell.font      = { bold: true, color: { argb: 'FFFFFFFF' }, size: 10 };
      cell.alignment = { horizontal: 'center', vertical: 'middle' };
    });

    const allMixes = snap?.cement?.all_mixes || {};
    ['1:3', '1:4', '1:5', '1:6'].forEach((mix, idx) => {
      const m   = allMixes[mix] || {};
      const row = csSheet.addRow([`CM ${mix}`, m.cement_bags_12mm || 0, m.cement_bags_18mm || 0, m.sand_cum || 0, m.sand_tons || 0]);
      row.height = 18;
      row.eachCell(cell => {
        cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: mix === '1:4' ? 'FFEFF6FF' : (idx % 2 === 0 ? 'FFFFFFFF' : 'FFF8FAFC') } };
        cell.font      = { size: 10, bold: mix === '1:4', color: { argb: mix === '1:4' ? 'FF1E6FD9' : 'FF475569' } };
        cell.alignment = { horizontal: 'center' };
        cell.border    = { bottom: { style: 'thin', color: { argb: 'FFE2E8F0' } } };
      });
    });

    const ratesSheet = wb.addWorksheet('Master Rates');
    ratesSheet.columns = [
      { key: 'material', width: 30 }, { key: 'category', width: 14 }, { key: 'rate', width: 12 },
      { key: 'unit', width: 10 }, { key: 'gst', width: 10 }, { key: 'loading', width: 12 },
      { key: 'transport', width: 16 }, { key: 'distance', width: 14 }, { key: 'unloading', width: 12 }, { key: 'total', width: 16 },
    ];
    ratesSheet.mergeCells('A1:J1');
    ratesSheet.getCell('A1').value = 'Master Rate List';
    ratesSheet.getCell('A1').font  = { bold: true, size: 14, color: { argb: 'FF7C3AED' } };
    ratesSheet.getCell('A1').alignment = { horizontal: 'center' };
    ratesSheet.getRow(1).height = 28;
    ratesSheet.addRow([]);

    const ratesHeader = ratesSheet.addRow(['Material', 'Category', 'Rate', 'Unit', 'GST%', 'Loading', 'Transport/km', 'Distance km', 'Unloading', 'Total at Site']);
    ratesHeader.height = 20;
    ratesHeader.eachCell(cell => {
      cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF7C3AED' } };
      cell.font      = { bold: true, color: { argb: 'FFFFFFFF' }, size: 10 };
      cell.alignment = { horizontal: 'center', vertical: 'middle' };
    });

    (rates || []).forEach((r, idx) => {
      const row = ratesSheet.addRow([r.material || '', r.category || '', r.rate || 0, r.unit || '', r.gst_pct || 0, r.loading || 0, r.transport_km || 0, r.distance_km || 0, r.unloading || 0, costAtSite(r).toFixed(2)]);
      row.height = 16;
      row.eachCell(cell => {
        cell.fill      = { type: 'pattern', pattern: 'solid', fgColor: { argb: idx % 2 === 0 ? 'FFFFFFFF' : 'FFF5F3FF' } };
        cell.font      = { size: 9 };
        cell.alignment = { horizontal: 'center' };
        cell.border    = { bottom: { style: 'thin', color: { argb: 'FFE2E8F0' } } };
      });
      ratesSheet.getCell(`A${row.number}`).alignment = { horizontal: 'left' };
    });

    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.setHeader('Content-Disposition', `attachment; filename="BOQ_${project?.name || project_id}_${Date.now()}.xlsx"`);
    await wb.xlsx.write(res);
    res.end();

  } catch (err) {
    console.error("Excel export error:", err);
    if (!res.headersSent) res.status(500).json({ error: err.message });
  }
});

// ═══════════════════════════════════════════════════════════
// START SERVER
// ═══════════════════════════════════════════════════════════

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));