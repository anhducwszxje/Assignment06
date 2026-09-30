"""Load trusted project checkpoints and run the notebook preprocessing pipeline."""
from pathlib import Path
import io
import json
import threading
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class ForecastRNN(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])


class Predictor:
    def __init__(self, directory):
        self.meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        self.scalers = json.loads((directory / 'scalers.json').read_text(encoding='utf-8'))
        self.lock = threading.Lock()
        if self.meta['framework'] == 'pytorch':
            self.model = ForecastRNN(len(self.meta['features']), self.meta['hidden_size'])
            self.model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
            self.model.eval()
        else:
            from tensorflow import keras
            self.model = keras.models.load_model(directory / 'model.keras', compile=False)

    def predict(self, windows):
        scaled = (np.asarray(windows, dtype=np.float64) * self.scalers['feature_scale']
                  + self.scalers['feature_offset']).astype(np.float32)
        outputs = []
        started = time.perf_counter()
        with self.lock:
            for start in range(0, len(scaled), self.meta['batch_size']):
                batch = scaled[start:start + self.meta['batch_size']]
                if self.meta['framework'] == 'pytorch':
                    with torch.inference_mode():
                        value = self.model(torch.from_numpy(batch)).numpy()
                else:
                    value = self.model(batch, training=False).numpy()
                outputs.append(value)
        prediction = np.concatenate(outputs).astype(np.float64)
        prediction = (prediction - self.scalers['target_offset']) / self.scalers['target_scale']
        elapsed = (time.perf_counter() - started) * 1000
        if not np.isfinite(prediction).all():
            raise ValueError('Mô hình không tạo được dự báo hữu hạn cho dữ liệu này.')
        return prediction.ravel(), elapsed


def read_history(csv_text, meta):
    try:
        frame = pd.read_csv(io.StringIO(csv_text), float_precision='round_trip')
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('Không đọc được CSV. Hãy dùng tệp mẫu với dấu phẩy phân cột.') from exc
    columns = ['date', *meta['features']]
    if list(frame.columns) != columns:
        raise ValueError('CSV cần đúng các cột theo thứ tự: ' + ', '.join(columns))
    if not meta['window'] <= len(frame) <= 2000:
        raise ValueError(f"CSV cần từ {meta['window']} đến 2000 dòng quan sát.")
    try:
        dates = pd.to_datetime(frame['date'], format='%Y-%m-%d', errors='raise')
        values = frame[meta['features']].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('Ngày phải có dạng YYYY-MM-DD; các đặc trưng phải là số.') from exc
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Ngày phải hợp lệ, tăng dần và không trùng lặp.')
    if not np.isfinite(values).all():
        raise ValueError('Dữ liệu có ô trống, NaN hoặc giá trị vô hạn.')
    if (values < 0).any():
        raise ValueError('Các đặc trưng đầu vào không được âm.')
    if meta['dataset'] == 'stock':
        if (values[:, :4] <= 0).any():
            raise ValueError('Giá Open, High, Low và Close phải lớn hơn 0.')
        if ((values[:, 1] < values[:, [0, 2, 3]].max(axis=1)).any()
                or (values[:, 2] > values[:, [0, 1, 3]].min(axis=1)).any()):
            raise ValueError('OHLC không hợp lệ: High phải lớn nhất và Low phải nhỏ nhất.')
    else:
        orders = values[:, 2]
        if (orders <= 0).any() or not np.equal(orders, np.floor(orders)).all():
            raise ValueError('transaction_count phải là số nguyên dương.')
        if not np.allclose(values[:, 3], values[:, 0] / orders, rtol=1e-5, atol=0.01):
            raise ValueError('avg_order_value phải bằng daily_revenue / transaction_count.')
    frame['date'] = dates.dt.strftime('%Y-%m-%d')
    return frame


def metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    sse = float(np.sum(error ** 2))
    sst = float(np.sum((actual - np.mean(actual)) ** 2))
    nonzero = np.abs(actual) > 0
    return {'mae': float(np.mean(np.abs(error))),
            'rmse': float(np.sqrt(np.mean(error ** 2))),
            'mape': float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
            'r2': 1 - sse / sst if sst > 0 else None}
