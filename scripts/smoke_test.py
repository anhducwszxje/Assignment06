"""Verify exported models through the running HTTP service, including invalid CSV."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8006')
    args = parser.parse_args()
    checks = 0
    report = {'tested_at': datetime.now(timezone.utc).isoformat(), 'base_url': args.base_url,
              'models': {}, 'invalid_inputs': []}
    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        response = client.get('/api/health')
        assert response.status_code == 200 and response.json()['loaded_models'] == 4
        checks += 1
        for path in ['/', '/static/app.js', '/static/style.css', '/docs', '/openapi.json', '/api/catalog']:
            assert client.get(path).status_code == 200, path
            checks += 1
        for dataset in ['stock', 'retail']:
            sample = client.get(f'/api/sample/{dataset}')
            assert sample.status_code == 200 and 'attachment' in sample.headers['content-disposition']
            checks += 1
            frame = pd.read_csv(ROOT / 'samples' / f'{dataset}.csv')
            for framework in ['pytorch', 'keras']:
                key = f'{dataset}_{framework}'
                directory = ROOT / 'artifacts' / key
                meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
                assert hashlib.sha256((ROOT / meta['source_notebook']).read_bytes()).hexdigest() == meta['source_sha256']
                ref = np.load(directory / 'reference.npz')
                result = client.get(f'/api/evaluate/{dataset}/{framework}')
                assert result.status_code == 200
                evaluated = result.json()
                np.testing.assert_array_equal(evaluated['actual'], ref['actual'])
                np.testing.assert_array_equal(evaluated['baseline'], ref['baseline'])
                np.testing.assert_allclose(evaluated['predicted'], ref['predicted'], rtol=0, atol=0.05)
                assert evaluated['dates'] == ref['dates'][meta['window']:].tolist()
                checks += 1
                for index in [0, len(ref['actual']) // 2, len(ref['actual']) - 1]:
                    payload = {'dataset': dataset, 'framework': framework, 'target_date': evaluated['dates'][index]}
                    response = client.post('/api/predict', json=payload)
                    assert response.status_code == 200, response.text
                    one = response.json()
                    assert one['window_end'] < one['target_date']
                    assert abs(one['prediction'] - ref['predicted'][index]) < 0.05
                    assert one['actual'] == ref['actual'][index]
                    assert one['baseline'] == ref['baseline'][index]
                    checks += 1
                custom = client.post('/api/predict', json={'dataset': dataset, 'framework': framework,
                    'csv_text': frame.iloc[:-1].to_csv(index=False)})
                assert custom.status_code == 200, custom.text
                assert custom.json()['actual'] is None and custom.json()['target_date'] is None
                assert abs(custom.json()['prediction'] - ref['predicted'][-1]) < 0.05
                checks += 1
                http_ms, infer_ms = [], []
                for _ in range(10):
                    started = time.perf_counter()
                    response = client.post('/api/predict', json=payload)
                    assert response.status_code == 200
                    http_ms.append((time.perf_counter() - started) * 1000)
                    infer_ms.append(response.json()['inference_ms'])
                report['models'][key] = {
                    'samples': evaluated['samples'], 'best_epoch': meta['best_epoch'],
                    'max_abs_difference': float(np.max(np.abs(np.asarray(evaluated['predicted']) - ref['predicted']))),
                    'metrics': evaluated['metrics'], 'last_prediction': one,
                    'warm_requests': 10, 'http_median_ms': float(np.median(http_ms)),
                    'http_p95_ms': float(np.percentile(http_ms, 95)),
                    'inference_median_ms': float(np.median(infer_ms)),
                }
        stock = pd.read_csv(ROOT / 'samples/stock.csv').tail(20)
        duplicate = stock.copy(); duplicate.iloc[-1, 0] = duplicate.iloc[-2, 0]
        missing = stock.copy(); missing.iloc[-1, 1] = np.nan
        negative = stock.copy(); negative.iloc[-1, 5] = -1
        ohlc = stock.copy(); ohlc.iloc[-1, 2] = 1
        cases = [('too_short', stock.head(19)), ('missing_column', stock.drop(columns='volume')),
                 ('duplicate_date', duplicate), ('reverse_date', stock.iloc[::-1]),
                 ('nan', missing), ('negative_volume', negative), ('invalid_ohlc', ohlc)]
        for name, invalid in cases:
            response = client.post('/api/predict', json={'dataset':'stock', 'framework':'pytorch',
                                   'csv_text':invalid.to_csv(index=False)})
            assert response.status_code == 400, (name, response.text)
            report['invalid_inputs'].append({'case':name, 'status':400, 'detail':response.json()['detail']})
            checks += 1
        retail = pd.read_csv(ROOT / 'samples/retail.csv').tail(14)
        retail.iloc[-1, 4] = 1
        for name, payload, status in [
            ('invalid_aov', {'dataset':'retail','framework':'keras','csv_text':retail.to_csv(index=False)},400),
            ('unknown_framework', {'dataset':'stock','framework':'other','target_date':'2018-02-07'},422),
            ('unknown_date', {'dataset':'stock','framework':'pytorch','target_date':'2099-01-01'},400),
            ('mixed_sources', {'dataset':'stock','framework':'keras','target_date':'2018-02-07','csv_text':stock.to_csv(index=False)},400),
            ('oversize_csv', {'dataset':'stock','framework':'pytorch','csv_text':'x'*250001},422),
        ]:
            response = client.post('/api/predict', json=payload)
            assert response.status_code == status, (name,response.text)
            report['invalid_inputs'].append({'case':name,'status':status})
            checks += 1
    report['checks_passed'] = checks
    (ROOT / 'artifacts/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
