const $ = id => document.getElementById(id);
let catalog, framework = 'pytorch', generation = 0;
const fmt = (value, digits = 2) => value == null ? '—' : Number(value).toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
const dateLabel = value => value ? value.split('-').reverse().join('/') : 'Quan sát kế tiếp';
const selectedModel = () => catalog.models.find(m => m.dataset === $('dataset').value && m.framework === framework);

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Yêu cầu không hợp lệ. Kiểm tra lại dữ liệu đầu vào.');
  return body;
}

function configureDataset() {
  const dataset = $('dataset').value;
  $('target-date').replaceChildren(...catalog.dates[dataset].map(date => new Option(dateLabel(date), date)));
  $('target-date').value = catalog.dates[dataset].at(-1);
  $('sample-download').href = `/api/sample/${dataset}`;
  $('csv-text').value = '';
  $('csv-file').value = '';
  $('csv-text').placeholder = ['date', ...selectedModel().features].join(',');
  $('definition').textContent = dataset === 'stock'
    ? 'AAPL: giá đóng cửa phiên kế tiếp, từ 20 phiên OHLCV. Dữ liệu S&P 500 giai đoạn 2013–2018.'
    : 'Bán lẻ: doanh số đơn hàng dương của quan sát kế tiếp, từ 14 ngày có dữ liệu; đã loại giao dịch hủy. Không phải doanh thu thuần.';
  updateModel();
}

function updateModel() {
  const m = selectedModel();
  $('model-info').textContent = `RNN 32 đơn vị ẩn · ${m.window} bước × ${m.features.length} đặc trưng · ${m.parameters.toLocaleString('en-US')} tham số · Best epoch ${m.best_epoch}`;
  $('framework-tag').textContent = framework === 'pytorch' ? 'PyTorch' : 'Keras';
  for (const name of ['pytorch', 'keras']) {
    $(name).classList.toggle('active', framework === name);
    $(name).setAttribute('aria-pressed', framework === name);
  }
}

function plot(data) {
  const svg = $('chart'), ns = 'http://www.w3.org/2000/svg';
  svg.replaceChildren();
  const W = 900, H = 260, margin = {left: 60, right: 12, top: 12, bottom: 32};
  const values = [...data.actual, ...data.predicted, ...data.baseline];
  const spread = Math.max(...values) - Math.min(...values) || 1;
  const low = Math.min(...values) - spread * .08, high = Math.max(...values) + spread * .08;
  const x = i => margin.left + i / (data.samples - 1) * (W - margin.left - margin.right);
  const y = v => margin.top + (high - v) / (high - low) * (H - margin.top - margin.bottom);
  function add(tag, attrs, text) {
    const el = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    if (text != null) el.textContent = text;
    svg.appendChild(el); return el;
  }
  for (let i = 0; i < 5; i++) {
    const value = low + (high - low) * i / 4, py = y(value);
    add('line', {x1: margin.left, x2: W - margin.right, y1: py, y2: py, stroke: '#e8edf3'});
    add('text', {x: margin.left - 10, y: py + 4, 'text-anchor': 'end', fill: '#7c8ba0', 'font-size': 11}, value >= 1000 ? `${fmt(value / 1000, 0)}k` : fmt(value, 0));
  }
  for (const [key, color, dash] of [['baseline','#a9b6c5','4 4'], ['actual','#284766',''], ['predicted','#b8263c','']]) {
    add('polyline', {points: data[key].map((value, i) => `${x(i)},${y(value)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': key === 'baseline' ? 1.2 : 2, 'stroke-dasharray': dash, 'stroke-linejoin': 'round'});
  }
  for (let step = 0; step < 5; step++) {
    const i = Math.round((data.samples - 1) * step / 4);
    add('text', {x: x(i), y: H - 7, 'text-anchor': step === 0 ? 'start' : step === 4 ? 'end' : 'middle', fill: '#7c8ba0', 'font-size': 11}, dateLabel(data.dates[i]));
  }
}

function showEvaluation(data) {
  $('chart-subtitle').textContent = `${data.samples} mẫu Test · ${dateLabel(data.dates[0])} – ${dateLabel(data.dates.at(-1))} · ${data.currency}`;
  for (const key of ['mae','rmse']) $(key).textContent = fmt(data.metrics[key]);
  $('mape').textContent = `${fmt(data.metrics.mape)}%`;
  $('r2').textContent = fmt(data.metrics.r2, 4);
  const change = (1 - data.metrics.mae / data.baseline_metrics.mae) * 100;
  $('comparison').textContent = `MAE baseline: ${fmt(data.baseline_metrics.mae)} ${data.currency}. RNN ${change >= 0 ? 'giảm' : 'tăng'} ${fmt(Math.abs(change))}% MAE trên cùng các nhãn Test.`;
  plot(data);
}

function showPrediction(data) {
  $('result-title').textContent = data.target_date ? `Dự báo ngày ${dateLabel(data.target_date)}` : 'Dự báo quan sát kế tiếp';
  $('prediction').textContent = `${fmt(data.prediction)} ${data.currency}`;
  $('baseline').textContent = `${fmt(data.baseline)} ${data.currency}`;
  $('actual').textContent = data.actual == null ? 'Chưa có nhãn' : `${fmt(data.actual)} ${data.currency}`;
  $('prediction-note').textContent = data.change_pct == null ? 'Giá trị sau inverse-transform' : `${data.change_pct >= 0 ? '+' : ''}${fmt(data.change_pct)}% so với quan sát gần nhất`;
  $('actual-note').textContent = data.actual == null ? 'Không tính sai số cho CSV chưa có nhãn' : `Sai số tuyệt đối: ${fmt(data.absolute_error)} ${data.currency}`;
  $('window-info').textContent = `Cửa sổ đầu vào: ${dateLabel(data.window_start)} → ${dateLabel(data.window_end)} · ${data.window_size} quan sát thực`;
  $('latency').textContent = `Suy luận: ${fmt(data.inference_ms, 1)} ms`;
  $('warnings').textContent = data.warnings.join(' ');
  $('warnings').hidden = !data.warnings.length;
}

async function run() {
  const ticket = ++generation;
  updateModel();
  $('predict').disabled = true;
  $('error').hidden = true;
  $('warnings').hidden = true;
  $('result-title').textContent = 'Đang chạy mô hình…';
  for (const id of ['prediction','baseline','actual','mae','rmse','mape','r2','comparison','window-info','latency','prediction-note','actual-note']) $(id).textContent = '—';
  $('chart').replaceChildren();
  const request = {dataset: $('dataset').value, framework};
  if ($('source').value === 'csv') request.csv_text = $('csv-text').value;
  else request.target_date = $('target-date').value;
  try {
    const [prediction, evaluation] = await Promise.all([
      api('/api/predict', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(request)}),
      api(`/api/evaluate/${request.dataset}/${request.framework}`)
    ]);
    if (ticket !== generation) return;
    showPrediction(prediction); showEvaluation(evaluation);
  } catch (error) {
    if (ticket !== generation) return;
    $('error').textContent = error.message;
    $('error').hidden = false;
    $('result-title').textContent = 'Chưa có kết quả dự báo';
    $('chart-subtitle').textContent = 'Kiểm tra dữ liệu đầu vào và thử lại';
  } finally {
    if (ticket === generation) $('predict').disabled = false;
  }
}

$('dataset').addEventListener('change', () => {configureDataset(); run();});
for (const name of ['pytorch','keras']) $(name).addEventListener('click', () => {framework = name; run();});
$('predict').addEventListener('click', run);
$('target-date').addEventListener('change', run);
$('source').addEventListener('change', () => {
  const custom = $('source').value === 'csv';
  $('sample-fields').hidden = custom; $('csv-fields').hidden = !custom;
  if (!custom) run();
});
$('csv-file').addEventListener('change', async () => {
  const file = $('csv-file').files[0];
  if (!file) return;
  if (file.size > 250000) {
    $('csv-text').value = '';
    $('error').textContent = 'Tệp CSV vượt giới hạn 250 KB.';
    $('error').hidden = false; return;
  }
  $('csv-text').value = await file.text();
});

async function init() {
  try {
    const [health, data] = await Promise.all([api('/api/health'), api('/api/catalog')]);
    catalog = data;
    $('health').textContent = `${health.loaded_models} mô hình sẵn sàng`;
    configureDataset(); await run();
  } catch (error) {
    $('health').textContent = 'Không kết nối được API';
    $('error').textContent = error.message; $('error').hidden = false;
  }
}
init();
