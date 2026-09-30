"""Execute the existing notebooks in memory and export their selected models."""
from pathlib import Path
import argparse
import json
import os
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'01': 'stock_pytorch', '02': 'retail_pytorch',
          '03': 'stock_keras', '04': 'retail_keras'}

EPILOGUE = r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, platform
out = Path(EXPORT_PATH)
out.mkdir(parents=True, exist_ok=True)
is_torch = MODEL_ID.endswith('pytorch')
dataset = MODEL_ID.split('_')[0]
window = int(globals().get('seq_len', globals().get('sequence_length', 14)))
raw_test = globals().get('test_data', globals().get('test_raw_data'))
train_raw = globals().get('train_data', globals().get('train_raw_data'))
if is_torch:
    torch.save(model.cpu().state_dict(), out / 'model.pt')
    runtime = {'torch': torch.__version__}
    losses = train_losses
else:
    model.save(out / 'model.keras')
    runtime = {'tensorflow': tf.__version__, 'keras': keras.__version__}
    losses = history.history['loss']
scalers = {
    'feature_scale': feature_scaler.scale_.tolist(),
    'feature_offset': feature_scaler.min_.tolist(),
    'feature_min': feature_scaler.data_min_.tolist(),
    'feature_max': feature_scaler.data_max_.tolist(),
    'target_scale': target_scaler.scale_.tolist(),
    'target_offset': target_scaler.min_.tolist(),
}
(out / 'scalers.json').write_text(json.dumps(scalers, indent=2), encoding='utf-8')
metadata = {
    'id': MODEL_ID, 'dataset': dataset,
    'framework': 'pytorch' if is_torch else 'keras',
    'features': feature_cols, 'target_index': 3 if dataset == 'stock' else 0,
    'window': window, 'hidden_size': 32,
    'batch_size': 32 if dataset == 'stock' else 16,
    'currency': 'USD' if dataset == 'stock' else 'GBP',
    'best_epoch': int(best_epoch), 'epochs_run': len(losses),
    'parameters': sum(p.numel() for p in model.parameters()) if is_torch else model.count_params(),
    'train_end': str(pd.Timestamp(train_dates[-1]).date()),
    'validation_end': str(pd.Timestamp(val_dates[-1]).date()),
    'test_start': str(test_target_dates[0].date()),
    'test_end': str(test_target_dates[-1].date()),
    'test_samples': len(actual_eval),
    'metrics': rnn_metrics, 'baseline_metrics': baseline_metrics,
    'source_notebook': SOURCE_NOTEBOOK,
    'source_sha256': hashlib.sha256(Path(SOURCE_NOTEBOOK).read_bytes()).hexdigest(),
    'exported_at': datetime.now(timezone.utc).isoformat(),
    'python': platform.python_version(), 'runtime': runtime,
}
(out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
np.savez_compressed(out / 'reference.npz', raw_test=raw_test.astype(np.float64),
    dates=pd.to_datetime(test_dates).strftime('%Y-%m-%d').to_numpy(dtype='U10'),
    actual=actual_eval, predicted=predicted_eval, baseline=baseline_pred)
sample = pd.DataFrame(raw_test, columns=feature_cols)
sample.insert(0, 'date', pd.to_datetime(test_dates).strftime('%Y-%m-%d'))
sample_dir = out.parent.parent / 'samples'
sample_dir.mkdir(exist_ok=True)
sample.to_csv(sample_dir / (dataset + '.csv'), index=False)
print('EXPORT_RESULT:' + json.dumps(metadata, ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', choices=list(MODELS), nargs='+', default=list(MODELS))
    args = parser.parse_args()
    os.environ['PYTHONHASHSEED'] = '42'
    logs = ROOT / 'artifacts' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for key in args.notebook:
        path = next(ROOT.glob(key + '*.ipynb'))
        notebook = nbformat.read(path, as_version=4)
        prefix = '\n'.join([
            'EXPORT_PATH = ' + repr(str(ROOT / 'artifacts' / MODELS[key])),
            'MODEL_ID = ' + repr(MODELS[key]),
            'SOURCE_NOTEBOOK = ' + repr(path.name),
        ])
        notebook.cells.append(nbformat.v4.new_code_cell(prefix + '\n' + EPILOGUE))
        print('Executing and exporting:', path.name, flush=True)
        client = NotebookClient(notebook, timeout=1200, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT)}})
        client.create_kernel_manager().kernel_spec.argv = [
            sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client.execute()
        text = '\n'.join(o.get('text', '') for cell in notebook.cells
                         for o in cell.get('outputs', []) if o.output_type == 'stream')
        (logs / f'{MODELS[key]}.log').write_text(text, encoding='utf-8')
        result = json.loads(text.rsplit('EXPORT_RESULT:', 1)[1])
        print(json.dumps({'id': MODELS[key], 'best_epoch': result['best_epoch'],
                          'metrics': result['metrics']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    main()
