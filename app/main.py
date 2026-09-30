"""HTTP service for the four exported PyTorch and Keras models."""
import os
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.inference import ROOT, Predictor, metrics, read_history

Dataset = Literal['stock', 'retail']
Framework = Literal['pytorch', 'keras']
predictors = {}
histories = {}


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    for dataset in ('stock', 'retail'):
        for framework in ('pytorch', 'keras'):
            key = f'{dataset}_{framework}'
            predictor = Predictor(ROOT / 'artifacts' / key)
            predictors[key] = predictor
            if dataset not in histories:
                histories[dataset] = read_history((ROOT / 'samples' / f'{dataset}.csv').read_text(), predictor.meta)
            raw = histories[dataset][predictor.meta['features']].to_numpy(dtype=float)
            predictor.predict(raw[:predictor.meta['window']][None, ...])
    yield
    predictors.clear()
    histories.clear()


app = FastAPI(title='Assignment 06 - RNN Forecast Lab', version='1.0.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT / 'app' / 'static'), name='static')


class PredictionRequest(BaseModel):
    dataset: Dataset
    framework: Framework
    target_date: str | None = Field(default=None, max_length=10)
    csv_text: str | None = Field(default=None, max_length=250_000)


@app.get('/', include_in_schema=False)
def home():
    return FileResponse(ROOT / 'app' / 'static' / 'index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'loaded_models': len(predictors), 'models': list(predictors),
            'device': 'cpu', 'version': app.version}


@app.get('/api/catalog')
def catalog():
    return {'models': [p.meta for p in predictors.values()],
            'dates': {key: frame.date.tolist()[predictors[f'{key}_pytorch'].meta['window']:]
                      for key, frame in histories.items()}}


@app.get('/api/sample/{dataset}')
def sample(dataset: Dataset):
    return FileResponse(ROOT / 'samples' / f'{dataset}.csv', media_type='text/csv',
                        filename=f'{dataset}_history.csv')


@app.post('/api/predict')
def predict(request: PredictionRequest):
    predictor = predictors[f'{request.dataset}_{request.framework}']
    meta = predictor.meta
    actual = None
    try:
        if request.csv_text is not None:
            if request.target_date is not None:
                raise ValueError('Chọn dữ liệu mẫu hoặc CSV tải lên, không dùng đồng thời.')
            frame = read_history(request.csv_text, meta)
            target_date = None
        else:
            if request.target_date is None:
                raise ValueError('Cần chọn ngày dự báo của dữ liệu mẫu.')
            history = histories[request.dataset]
            matches = history.index[history.date == request.target_date].tolist()
            if not matches or matches[0] < meta['window']:
                raise ValueError('Ngày dự báo không thuộc tập Test hoặc chưa đủ lịch sử.')
            index = matches[0]
            # The target row is never passed into the model input window.
            frame = history.iloc[:index]
            actual = float(history.iloc[index][meta['features'][meta['target_index']]])
            target_date = request.target_date
        window = frame.tail(meta['window'])
        raw = window[meta['features']].to_numpy(dtype=float)
        values, elapsed = predictor.predict(raw[None, ...])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    predicted = float(values[0])
    previous = float(raw[-1, meta['target_index']])
    out_of_range = (raw < predictor.scalers['feature_min']) | (raw > predictor.scalers['feature_max'])
    warnings = []
    if out_of_range.any():
        warnings.append(f'{int(out_of_range.sum())}/{raw.size} giá trị đầu vào nằm ngoài khoảng Min-Max của Train; giữ nguyên phép chuẩn hóa.')
    if predicted < 0:
        warnings.append('Mô hình hồi quy trả giá trị âm; kết quả được giữ nguyên để kiểm tra, không tự cắt về 0.')
    return {'dataset': request.dataset, 'framework': request.framework,
            'currency': meta['currency'], 'target_date': target_date,
            'prediction': predicted, 'actual': actual, 'baseline': previous,
            'absolute_error': abs(predicted - actual) if actual is not None else None,
            'change_pct': (predicted / previous - 1) * 100 if previous != 0 else None,
            'window_start': window.date.iloc[0], 'window_end': window.date.iloc[-1],
            'window_size': len(window), 'inference_ms': elapsed, 'warnings': warnings,
            'history': [{'date': row[0], 'value': float(row[1 + meta['target_index']])}
                        for row in window.itertuples(index=False, name=None)]}


@app.get('/api/evaluate/{dataset}/{framework}')
def evaluate(dataset: Dataset, framework: Framework):
    predictor = predictors[f'{dataset}_{framework}']
    meta = predictor.meta
    frame = histories[dataset]
    raw = frame[meta['features']].to_numpy(dtype=float)
    length = meta['window']
    windows = np.stack([raw[i:i + length] for i in range(len(raw) - length)])
    predicted, elapsed = predictor.predict(windows)
    actual = raw[length:, meta['target_index']]
    baseline = raw[length - 1:-1, meta['target_index']]
    return {'dataset': dataset, 'framework': framework, 'currency': meta['currency'],
            'metrics': metrics(actual, predicted), 'baseline_metrics': metrics(actual, baseline),
            'samples': len(actual), 'inference_ms': elapsed,
            'dates': frame.date.tolist()[length:], 'actual': actual.tolist(),
            'predicted': predicted.tolist(), 'baseline': baseline.tolist()}
