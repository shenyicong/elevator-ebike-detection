/**
 * 浏览器 WASM 推理基线测速（对应线上在线检测页的执行方式）。
 *
 * 前置：
 *   npm install onnxruntime-web@1.19.2
 * 运行：
 *   node scripts/benchmark_wasm.mjs
 *
 * 说明：线上页面在用户浏览器里用 onnxruntime-web 的 wasm 后端，
 * GitHub Pages 无法设置 COOP/COEP 头，因此只能单线程（numThreads=1）。
 * 本脚本用同样的单线程配置测量作为对照基线。
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ort = await import("onnxruntime-web");
const ROOT = path.resolve(__dirname, "..");

// Node 下 wasmPaths 必须是 file:// URL
ort.env.wasm.wasmPaths =
  pathToFileURL(path.join(__dirname, "../node_modules/onnxruntime-web/dist/") + path.sep).href;
ort.env.wasm.numThreads = 1; // 与 GitHub Pages 线上一致

const models = {
  ds1_elevator: path.join(ROOT, "docs/assets/models/ds1_elevator.onnx"),
  ds2_liftebike: path.join(ROOT, "docs/assets/models/ds2_liftebike.onnx"),
};

const S = 640;
const data = new Float32Array(3 * S * S);
for (let i = 0; i < data.length; i++) data[i] = Math.random() * 0.3;

function stats(ts) {
  ts.sort((a, b) => a - b);
  const mean = ts.reduce((a, b) => a + b, 0) / ts.length;
  return {
    mean: +mean.toFixed(2),
    median: +ts[Math.floor(ts.length / 2)].toFixed(2),
    p95: +ts[Math.floor(ts.length * 0.95) - 1].toFixed(2),
    fps: +(1000 / mean).toFixed(1),
  };
}

const result = {};
for (const [name, p] of Object.entries(models)) {
  const sess = await ort.InferenceSession.create(fs.readFileSync(p), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const feed = { [sess.inputNames[0]]: new ort.Tensor("float32", data, [1, 3, S, S]) };
  for (let i = 0; i < 10; i++) await sess.run(feed);
  const ts = [];
  for (let i = 0; i < 50; i++) {
    const t0 = performance.now();
    await sess.run(feed);
    ts.push(performance.now() - t0);
  }
  result[name] = stats(ts);
  const s = result[name];
  console.log(`${name.padEnd(16)} WASM(1 thread) mean=${s.mean}ms median=${s.median}ms ` +
    `p95=${s.p95}ms ~${s.fps} fps`);
}
fs.writeFileSync(path.join(ROOT, "benchmark_wasm_results.json"), JSON.stringify(result, null, 2));
console.log("结果已保存 benchmark_wasm_results.json");
