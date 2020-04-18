import os
import time
import runway
import sys
import logging
import json
import multiprocessing
from flask import Flask, jsonify, send_file
from gevent.pywsgi import WSGIServer


def fields_to_array(fields):
    ret = []
    for name, field in fields.items():
        field = runway.utils.cast_to_obj(field)
        field.name = name
        ret.append(field)
    return ret


class MetricsProxy(object):
    def __init__(self, ctx):
        self._ctx = ctx

    def __setitem__(self, key, value):
        self._ctx._state['metrics'][key] = value
        if key not in self._ctx._state['history']:
            self._ctx._state['history'][key] = []
        self._ctx._state['history'][key].append([
            runway.utils.timestamp_millis(),
            self._ctx._state['step'],
            value
        ])
        self._ctx.refresh()


class ArtifactsProxy(object):
    def __init__(self, key, ctx, keep_max=10):
        self._key = key
        self._ctx = ctx
        self._keep_max = keep_max

    def _delete_old_artifacts(self, key):
        artifacts = self._ctx._state[self._key][key]
        paths = self._ctx._state['paths']
        while len(artifacts) > self._keep_max:
            idx_to_remove = 0
            min_t = sys.maxsize
            for idx, [t, _, artifact_id] in enumerate(artifacts):
                if t < min_t:
                    idx_to_remove = idx
                    min_t = t
            path_to_remove = paths[artifacts[idx_to_remove][2]]
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
            artifacts.remove(artifacts[idx_to_remove])

    def add(self, key, path):
        if key not in self._ctx._state[self._key]:
            self._ctx._state[self._key][key] = []
        _, ext = os.path.splitext(path)
        artifact_id = runway.utils.generate_uuid() + ext
        self._ctx._state['paths'][artifact_id] = path
        self._ctx._state[self._key][key].append([
            runway.utils.timestamp_millis(),
            self._ctx._state['step'],
            artifact_id
        ])
        self._delete_old_artifacts(key)
        self._ctx.refresh()


class TrainingContext(object):
    def __init__(self, queue):
        self._queue = queue
        self._state = {
            'step': 0,
            'history': {},
            'metrics': {},
            'checkpoints': {},
            'samples': {},
            'paths': {}
        }
        self.metrics = MetricsProxy(self)
        self.checkpoints = ArtifactsProxy('checkpoints', self, keep_max=10)
        self.samples = ArtifactsProxy('samples', self, keep_max=100)

    def refresh(self):
        self._queue.put(self._state)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'step' or key == 'training_status':
            self._state[key] = value
            self.refresh()


def training_server_process(q):
    state = {
        'step': 0,
        'trainingStatus': 'STARTING',
        'history': {},
        'metrics': {},
        'checkpoints': {},
        'samples': {}
    }

    app = Flask('RunwayML Training Server')
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.ERROR)

    metrics = []

    time_started = time.time()

    def refresh_state():
        nonlocal state
        try:
            while not q.empty():
                state = q.get_nowait()
        except:
            return

    @app.route('/healthcheck', methods=['GET'])
    def healthcheck_route():
        return jsonify(dict(success=True))

    @app.route('/status')
    def status():
        refresh_state()
        return jsonify({
            'step': state['step'],
            'trainingStatus': state['training_status'],
            'metrics': state['metrics'],
            'timeElapsed': time.time() - time_started
        })

    @app.route('/history')
    def history():
        refresh_state()
        return jsonify(state['history'])

    @app.route('/samples')
    def samples():
        refresh_state()
        return jsonify(state['samples'])

    @app.route('/checkpoints')
    def checkpoints():
        refresh_state()
        return jsonify(state['checkpoints'])

    @app.route('/samples/<key>/<file_id>')
    def download_sample(key, file_id):
        refresh_state()
        path = state['paths'][file_id]
        return send_file(path, as_attachment=True)

    @app.route('/checkpoints/<key>/<file_id>')
    def download_checkpoint(key, file_id):
        refresh_state()
        path = state['paths'][file_id]
        return send_file(path, as_attachment=True)

    http_server = WSGIServer((os.getenv('RW_HOST', '0.0.0.0'), int(os.getenv('RW_PORT', '8000'))), app, log=logger)
    http_server.serve_forever()


def training_main_process(fn, datasets, options, queue):
    ctx = TrainingContext(queue)
    ctx.training_status = 'TRAINING'
    try:
        fn(datasets, options, ctx)
        print('Training succeeded!')
        # Provide some time for the sidecar to upload artifacts
        time.sleep(60 * 5)
        ctx.training_status = 'SUCCEEDED'
    except Exception as e:
        ctx.training_status = 'FAILED'
        import traceback
        print('Training failed with the following exception:', repr(e))
        traceback.print_exc()


def run(train_fn, schema):
    datasets_fields = fields_to_array(schema['datasets'])
    options_fields = fields_to_array(schema['options'])
    samples_fields = fields_to_array(schema['samples'])
    metrics_fields = fields_to_array(schema['metrics'])

    if os.getenv('RW_META', '0') == '1':
        print(json.dumps(dict(
            datasets=[field.to_dict() for field in datasets_fields],
            options=[field.to_dict() for field in options_fields],
            samples=[field.to_dict() for field in samples_fields],
            metrics=[field.to_dict() for field in metrics_fields]
        )))
        sys.exit(0)

    datasets = json.loads(os.getenv('RW_DATASETS', '{}'))
    datasets = runway.utils.deserialize_data(datasets, datasets_fields)

    options = json.loads(os.getenv('RW_MODEL_OPTIONS', '{}'))
    options = runway.utils.deserialize_data(options, options_fields)

    ipc_queue = multiprocessing.Queue()
    multiprocessing.Process(target=training_server_process, args=(ipc_queue,)).start()
    training_main_process(train_fn, datasets, options, ipc_queue)
