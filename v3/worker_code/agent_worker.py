"""SOC-Agent LLM worker: one process per GPU. Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python agent_worker.py <job_file.json>
The job file lists the runs this worker owns; each run is resumable."""
import os, sys, json, time, traceback

job = json.load(open(sys.argv[1]))
sys.path.insert(0, job['lib_dir'])
import soc_agent_lib as sal

LOG = open(job['log'], 'a', buffering=1)
def log(msg):
    LOG.write(f'{time.strftime("%H:%M:%S")} {msg}\n')

try:
    cfg = json.load(open(job['cfg']))
    data = json.load(open(job['alerts']))
    alerts, gt, ids = data['alerts'], data['gt'], data['ids']
    log(f'worker {job["worker"]} start; CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
    eng = sal.make_engine(cfg, job['hf_cache'])
    txt, _ = eng.generate([[{'role': 'system', 'content': 'Reply with JSON only.'},
                            {'role': 'user', 'content': 'Reply with only: {"status": "ok"}'}]], 32)[0]
    ok = (sal.extract_json(txt) or {}).get('status') == 'ok'
    json.dump({'model_id': eng.model_id, 'revision': eng.revision, 'connectivity_ok': ok,
               'compute_dtype': getattr(eng, 'compute_dtype', None),
               'quantization': getattr(eng, 'quantization', None)}, open(job['info'], 'w'))
    log(f'model ready: {eng.model_id} rev={eng.revision} connectivity_ok={ok}')
    rag = sal.RAG(job['kb_dir'], cfg['rag']['embedding_model'], cfg['rag']['top_k_retrieval'])
    _cache = {job['alerts']: (alerts, gt, ids)}
    for r in job['runs']:
        src = r.get('alerts', job['alerts'])          # v3 runs use their own packet file
        if src not in _cache:
            d = json.load(open(src)); _cache[src] = (d['alerts'], d['gt'], d['ids'])
        r_alerts, r_gt, r_ids = _cache[src]
        sub = r_ids if r['n'] is None else r_ids[:r['n']]
        if r.get('kind', 'variant') == 'condition':
            log(f'--- run {r["name"]}: condition={r["condition"]} prompt={r["prompt_set"]} '
                f'stages={r["stages"]} seed={r["seed"]} alerts={len(sub)}')
            sal.run_condition(r['condition'], r['prompt_set'], r['seed'], sub, r['out'], r_alerts, r_gt,
                              eng, cfg, rag, stages=r['stages'], batch_size=r.get('batch_size'), log=log)
        else:
            log(f'--- run {r["name"]}: variant={r["variant"]} seed={r["seed"]} alerts={len(sub)}')
            sal.run_variant(r['variant'], r['seed'], sub, r['out'], r_alerts, r_gt, eng, cfg, rag,
                            batch_size=r.get('batch_size'), log=log)
        log(f'--- run {r["name"]} finished')
    log('ALL RUNS DONE')
    open(job['done'], 'w').write('ok')
except Exception:
    log('WORKER FAILED:\n' + traceback.format_exc())
    open(job['done'], 'w').write('failed')
    sys.exit(1)
