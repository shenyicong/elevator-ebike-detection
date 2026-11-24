/**
 * 浏览器端 YOLO11n 检测脚本
 * ------------------------------------------------------------
 * 流程：加载 ONNX 模型 → 读取图片 → letterbox 预处理 → 推理
 *       → 置信度过滤 + NMS → 把框画到 Canvas 上
 * 所有计算都在用户浏览器本地完成，图片不会上传到任何服务器。
 *
 * 模型文件放在：assets/models/best.onnx
 * （从 Ultralytics Platform 模型页 → Export → ONNX 下载后放进来）
 */

// ============ 模型配置 ============
// 两个模型都用你自己训练的权重导出（从 Ultralytics Platform 下载的 .pt → ONNX）
const MODELS = {
  ds3: {
    path: "assets/models/ds3_merged.onnx",
    names: ["Baby-carriage", "Electric-bicycle", "bicycle", "person", "wheelchair"],
    label: "模型3 · 合并集5类（mAP50 89.5%，推荐）",
  },
  ds1: {
    path: "assets/models/ds1_elevator.onnx",
    names: ["Baby-carriage", "Electric-bicycle", "bicycle", "person", "wheelchair"],
    label: "模型1 · 数据集1的5类",
  },
  ds2: {
    path: "assets/models/ds2_liftebike.onnx",
    names: ["motorcycle"],
    label: "模型2 · 数据集2的1类（电梯内电动车）",
  },
};

const IMG_SIZE = 640;        // 模型输入尺寸（导出 ONNX 时的 imgsz）
const CONF_THRESH = 0.25;    // 置信度阈值：低于此值的框丢弃
const IOU_THRESH = 0.45;     // NMS 的 IoU 阈值：重叠框合并

// 每类别的框颜色（循环使用）
const COLORS = [
  [46, 204, 113], [239, 68, 68], [59, 130, 246], [245, 158, 11],
  [168, 85, 247], [20, 184, 166], [244, 63, 94], [100, 116, 139],
];

// ============ 全局状态 ============
let session = null;          // 当前 onnxruntime 推理会话
let sessions = {};           // 按模型 key 缓存会话，切换时不重复加载
let ortReady = false;        // onnxruntime 库是否已就绪
let currentImage = null;     // 当前图片 Image 对象
let currentModelKey = "ds3"; // 当前选中的模型（默认最佳：合并集）
let useWebGpu = false;       // 是否成功启用 WebGPU 加速（用于状态提示）
let lastInferMs = 0;         // 最近一次推理耗时（毫秒）

// onnxruntime 运行时来源：本地文件优先（随仓库一起部署，不依赖境外 CDN），
// 万一本地文件缺失再依次尝试公共 CDN。jsdelivr 在国内常被墙，已放到最后。
const ORT_SOURCES = [
  "assets/ort/",                                              // ① 本地（最稳）
  "https://unpkg.com/onnxruntime-web@1.19.2/dist/",           // ② unpkg 兜底
  "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.19.2/", // ③ cdnjs 兜底
];
let ortBase = ORT_SOURCES[0]; // 实际生效的来源，wasm 文件也从这里取
function classNames() { return MODELS[currentModelKey].names; }

/**
 * 加载 onnxruntime-web：
 * - 支持 WebGPU 的浏览器加载 ort.webgpu.min.js（走核显，快几十倍）
 * - 不支持则加载 ort.min.js（CPU WASM，慢但通用）
 * wasmPaths 在两种情况下都指向 CDN 目录，WebGPU 会自动加载 .jsep.wasm。
 */
let ortLoadingPromise = null; // 去重：预热与点击可能并发，共用同一次加载
async function loadOrt(onProgress) {
  if (window.ort) { ortReady = true; return; }
  if (ortLoadingPromise) {
    // 已有加载在进行中，复用它，避免重复注入 script 导致相互超时
    try { await ortLoadingPromise; } catch (e) { /* 由发起方处理 */ }
    if (window.ort) { ortReady = true; return; }
  }
  ortLoadingPromise = (async () => {
  // 真正探测 GPU adapter：有些浏览器有 navigator.gpu 但拿不到适配器
  let hasWebGpu = false;
  try {
    if (navigator.gpu) {
      hasWebGpu = !!(await navigator.gpu.requestAdapter());
    }
  } catch (e) {
    hasWebGpu = false; // 非安全上下文（file://）会抛错
  }
  const file = hasWebGpu ? "ort.webgpu.min.js" : "ort.min.js";

  // 依次尝试每个来源，任一成功即用；单个来源 8 秒超时，避免卡死
  let lastErr = null;
  for (const base of ORT_SOURCES) {
    try {
      if (onProgress) onProgress(`⏳ 正在加载检测引擎（${base.startsWith("assets") ? "本地" : "备用源"}）…`);
      await loadScript(base + file, 15000);
      ortBase = base;                    // 记住成功的来源，wasm 从同一处取
      useWebGpu = hasWebGpu;
      break;
    } catch (err) {
      lastErr = err;
      console.warn(`运行时来源 ${base} 加载失败：`, err);
      // WebGPU 版失败也可能只是该源没这个文件，下一轮统一再试
    }
  }
  if (!window.ort) {
    throw new Error("检测引擎加载失败：本地文件与备用 CDN 均不可用，请检查部署或网络");
  }
  // wasm / .mjs 必须与 js 同源；且要转成绝对 URL——
  // onnxruntime 内部用 import() 加载 .mjs，相对路径（如 assets/ort/）会报
  // "Failed to resolve module specifier"，必须给完整地址。CDN 源本身已是绝对地址。
  window.ort.env.wasm.wasmPaths = new URL(ortBase, window.location.href).href;
  ortReady = true;
  })();
  try { await ortLoadingPromise; } finally { ortLoadingPromise = null; }
}

/** 带超时的 script 加载：超时后移除节点，交给下一个来源重试 */
function loadScript(src, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      s.remove();
      reject(new Error("加载超时：" + src));
    }, timeoutMs);
    s.src = src;
    s.onload = () => { if (settled) return; settled = true; clearTimeout(timer); resolve(); };
    s.onerror = () => { if (settled) return; settled = true; clearTimeout(timer); s.remove(); reject(new Error("加载失败：" + src)); };
    document.head.appendChild(s);
  });
}

// ============ 加载 YOLO 模型 ============
async function loadModel(key, onProgress) {
  key = key || currentModelKey;
  if (sessions[key]) { session = sessions[key]; return session; }
  await loadOrt(onProgress);
  const opt = {
    executionProviders: useWebGpu ? ["webgpu"] : ["wasm"],
    graphOptimizationLevel: "all",
  };
  try {
    if (onProgress) onProgress(`⏳ 正在加载模型（约 10MB，首次需十几秒）…`);
    session = await window.ort.InferenceSession.create(MODELS[key].path, opt);
  } catch (err) {
    // WebGPU 创建会话失败（算子不支持/驱动问题）→ 自动降级 WASM
    if (useWebGpu) {
      console.warn("WebGPU 会话创建失败，自动降级 WASM：", err);
      useWebGpu = false;
      session = await window.ort.InferenceSession.create(MODELS[key].path, {
        executionProviders: ["wasm"], graphOptimizationLevel: "all",
      });
    } else {
      throw err;
    }
  }
  sessions[key] = session;
  return session;
}

/**
 * letterbox 预处理：把图片等比缩放并填充到 640x640
 * 返回 { data: Float32Array(CHW 归一化像素), ratio, padX, padY }
 */
function preprocess(img) {
  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const ratio = Math.min(IMG_SIZE / iw, IMG_SIZE / ih);
  const nw = Math.round(iw * ratio), nh = Math.round(ih * ratio);
  const padX = (IMG_SIZE - nw) / 2, padY = (IMG_SIZE - nh) / 2;

  // 画到中间画布（灰底填充）
  const cv = document.createElement("canvas");
  cv.width = IMG_SIZE; cv.height = IMG_SIZE;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  ctx.drawImage(img, padX, padY, nw, nh);
  const rgba = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;

  // RGB → CHW，归一化到 0~1
  const data = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    data[i] = rgba[i * 4] / 255;               // R
    data[plane + i] = rgba[i * 4 + 1] / 255;   // G
    data[2 * plane + i] = rgba[i * 4 + 2] / 255; // B
  }
  return { data, ratio, padX, padY, iw, ih };
}

/**
 * NMS：非极大值抑制，去掉重叠的重复框
 */
function nms(boxes, iouThresh) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  while (boxes.length) {
    const cur = boxes.shift();
    keep.push(cur);
    boxes = boxes.filter((b) => iou(cur, b) < iouThresh);
  }
  return keep;
}
function iou(a, b) {
  const xx1 = Math.max(a.x1, b.x1), yy1 = Math.max(a.y1, b.y1);
  const xx2 = Math.min(a.x2, b.x2), yy2 = Math.min(a.y2, b.y2);
  const w = Math.max(0, xx2 - xx1), h = Math.max(0, yy2 - yy1);
  const inter = w * h;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

/**
 * 后处理：解析模型输出 → 过滤 → 映射回原图坐标 → NMS
 * YOLO11 ONNX 输出形状通常为 [1, 4+nc, 8400]
 */
function postprocess(output, meta) {
  const dims = output.dims;              // 例如 [1, 9, 8400]
  const data = output.data;
  // Ultralytics ONNX 输出通常为 [1, 4+nc, 8400]（属性在前、anchor 在后）
  let numAnchors, numAttrs, transposed;
  if (dims[1] < dims[2]) {
    // [1, attrs, anchors]：按 data[attr * anchors + anchor] 读取
    numAttrs = dims[1];
    numAnchors = dims[2];
    transposed = true;
  } else {
    // [1, anchors, attrs]：按 data[anchor * attrs + attr] 读取
    numAnchors = dims[1];
    numAttrs = dims[2];
    transposed = false;
  }
  const clsCount = numAttrs - 4;

  const boxes = [];
  for (let a = 0; a < numAnchors; a++) {
    let cx, cy, bw, bh, bestScore = 0, bestCls = 0;
    for (let j = 0; j < numAttrs; j++) {
      const v = transposed ? data[j * numAnchors + a] : data[a * numAttrs + j];
      if (j === 0) cx = v;
      else if (j === 1) cy = v;
      else if (j === 2) bw = v;
      else if (j === 3) bh = v;
      else if (v > bestScore) { bestScore = v; bestCls = j - 4; }
    }
    if (bestScore < CONF_THRESH) continue;

    // 模型输出坐标基于 640 画布 → 去掉 padding、除以缩放比，映射回原图
    const x1 = (cx - bw / 2 - meta.padX) / meta.ratio;
    const y1 = (cy - bh / 2 - meta.padY) / meta.ratio;
    const x2 = (cx + bw / 2 - meta.padX) / meta.ratio;
    const y2 = (cy + bh / 2 - meta.padY) / meta.ratio;
    boxes.push({
      x1: Math.max(0, x1), y1: Math.max(0, y1),
      x2: Math.min(meta.iw, x2), y2: Math.min(meta.ih, y2),
      score: bestScore, cls: bestCls,
    });
  }
  return nms(boxes, IOU_THRESH);
}

/** 在 Canvas 上画出检测框 + 类别标签 + 置信度 */
function drawBoxes(canvas, img, boxes) {
  const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
  const offX = (canvas.width - img.naturalWidth * scale) / 2;
  const offY = (canvas.height - img.naturalHeight * scale) / 2;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, offX, offY, img.naturalWidth * scale, img.naturalHeight * scale);

  ctx.lineWidth = 3;
  ctx.font = "bold 18px 'Microsoft YaHei', sans-serif";
  boxes.forEach((b) => {
    const x = offX + b.x1 * scale, y = offY + b.y1 * scale;
    const w = (b.x2 - b.x1) * scale, h = (b.y2 - b.y1) * scale;
    const color = COLORS[b.cls % COLORS.length];
    const css = `rgb(${color[0]},${color[1]},${color[2]})`;
    ctx.strokeStyle = css;
    ctx.strokeRect(x, y, w, h);
    // 标签框
    const label = `${classNames()[b.cls] || b.cls} ${(b.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = css;
    ctx.fillRect(x, Math.max(0, y - 26), tw + 16, 26);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 8, Math.max(18, y - 7));
  });
}

// ============ 主流程：检测一张图片 ============
async function detect(img) {
  const status = document.getElementById("detect-status");
  status.className = "detect-status loading";
  status.textContent = "⏳ 正在初始化…";
  try {
    await loadModel(currentModelKey, (msg) => { status.textContent = msg; });
    status.textContent = "🔍 正在推理检测…";
    await new Promise((r) => setTimeout(r, 30)); // 让状态文字刷新

    const meta = preprocess(img);
    const feeds = {};
    const inputName = session.inputNames[0];
    feeds[inputName] = new window.ort.Tensor("float32", meta.data, [1, 3, IMG_SIZE, IMG_SIZE]);
    const t0 = performance.now();
    let results;
    try {
      results = await session.run(feeds);
    } catch (err) {
      // WebGPU 推理期失败 → 降级 WASM 会话后重试一次
      if (useWebGpu) {
        console.warn("WebGPU 推理失败，降级 WASM 重试：", err);
        useWebGpu = false;
        sessions = {}; // 清掉所有 WebGPU 会话，避免切换模型时又拿到旧的
        await loadModel(currentModelKey);
        results = await session.run(feeds);
      } else {
        throw err;
      }
    }
    lastInferMs = performance.now() - t0;
    const output = results[session.outputNames[0]];
    const boxes = postprocess(output, meta);

    const canvas = document.getElementById("detect-canvas");
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    drawBoxes(canvas, img, boxes);
    canvas.style.display = "block";
    document.getElementById("detect-placeholder").style.display = "none";

    status.className = "detect-status ok";
    const backend = useWebGpu ? "⚡ WebGPU 核显加速" : "CPU 兼容模式";
    status.textContent = `✅ 检测完成，发现 ${boxes.length} 个目标 · ${backend} · ${lastInferMs.toFixed(0)} ms`;
    renderResultList(boxes);
  } catch (err) {
    console.error(err);
    status.className = "detect-status err";
    status.textContent = "❌ " + (err.message || "检测失败");
    if (String(err).toLowerCase().includes("fetch") || String(err).includes(".onnx")) {
      status.innerHTML = "⚠️ 模型加载失败，请检查网络后刷新页面重试。";
    }
  }
}

function renderResultList(boxes) {
  const el = document.getElementById("detect-results");
  if (!boxes.length) { el.innerHTML = '<p class="detect-empty">未检测到目标，换一张试试</p>'; return; }
  el.innerHTML = boxes.map((b) => {
    const color = COLORS[b.cls % COLORS.length];
    return `<div class="detect-item">
      <span class="dot" style="background:rgb(${color.join(',')})"></span>
      <b>${classNames()[b.cls] || b.cls}</b>
      <span class="conf">${(b.score * 100).toFixed(1)}%</span>
    </div>`;
  }).join("");
}

// ============ 绑定上传交互 ============
function initUploader() {
  const input = document.getElementById("file-input");
  const drop = document.getElementById("detect-drop");
  input.addEventListener("change", (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });

  // 点击上传区（含"选择图片"按钮）时打开系统文件选择框
  drop.addEventListener("click", () => input.click());
  // input.click() 触发的 click 会冒泡回 drop，必须阻止，否则会递归无限调用
  input.addEventListener("click", (e) => e.stopPropagation());

  ["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });

  // 示例图按钮
  document.querySelectorAll("[data-demo]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => { currentImage = img; detect(img); };
      img.src = btn.getAttribute("data-demo");
    });
  });

  // 模型切换：切换后如果有当前图片就重新检测
  const sel = document.getElementById("model-select");
  if (sel) {
    sel.addEventListener("change", async (e) => {
      currentModelKey = e.target.value;
      const status = document.getElementById("detect-status");
      status.className = "detect-status loading";
      status.textContent = `⏳ 正在加载 ${MODELS[currentModelKey].label} …`;
      try { await loadModel(currentModelKey, (msg) => { status.textContent = msg; }); } catch (err) { console.error(err); }
      if (currentImage) { detect(currentImage); }
      else { status.className = "detect-status"; status.textContent = "模型已切换，等待上传图片…"; }
    });
  }
}

function handleFile(file) {
  if (!file.type.startsWith("image/")) { alert("请选择图片文件"); return; }
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => { currentImage = img; detect(img); };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

document.addEventListener("DOMContentLoaded", () => {
  initUploader();
  // 后台预热默认模型：页面一打开就悄悄下载引擎+权重并建好会话，
  // 用户点检测时通常已就绪，避免首次等待十几秒。失败也不影响正常使用。
  setTimeout(() => { loadModel(currentModelKey).catch((e) => console.warn("后台预热失败：", e)); }, 300);
});
