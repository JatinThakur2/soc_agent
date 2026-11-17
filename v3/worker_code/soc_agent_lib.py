"""SOC-Agent agent library (written by the notebook; used by the notebook and by
the per-GPU worker processes). JSON parsing + schema validation, LLM engines,
prompts, the batched multi-agent runner, and ATT&CK retrieval."""
import os, re, json, time, copy, random, hashlib
import numpy as np


def set_global_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as _s
        _s(seed)
    except Exception:
        pass


# ── Robust JSON extraction ─────────────────────────────────────────────────
def _try_parse(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def _find_matching_brace(text, start):
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False; continue
        if c == '\\' and in_str:
            esc = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1

def extract_json(text):
    """Return the first JSON object found in `text`, or None."""
    text = re.sub(r'```(?:json)?\s*', '', str(text)).strip()
    p = _try_parse(text)
    if isinstance(p, dict):
        return p
    s = text.find('{')
    e = _find_matching_brace(text, s) if s >= 0 else -1
    if s >= 0 and e > s:
        p = _try_parse(text[s:e + 1])
        if isinstance(p, dict):
            return p
    return None


# ── Output schemas ─────────────────────────────────────────────────────────
SEV_LEVELS = ['Informational', 'Low', 'Medium', 'High', 'Critical']
_SEV_ALIASES = {s.lower(): s for s in SEV_LEVELS}
_SEV_ALIASES.update({'info': 'Informational', 'none': 'Informational', 'crit': 'Critical'})

def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ('true', 'yes', '1'):
        return True
    if isinstance(v, str) and v.strip().lower() in ('false', 'no', '0'):
        return False
    return None

def _as_sev(v):
    return _SEV_ALIASES.get(str(v).strip().lower()) if v is not None else None

def _as_str_list(v):
    if v is None:
        return None
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x if isinstance(x, str) else json.dumps(x) for x in v]
    return None

def _as_tech_ids(v):
    if v is None:
        return None
    out = []
    for x in (v if isinstance(v, list) else [v]):
        tid = x.get('id', '') if isinstance(x, dict) else str(x)
        m = re.search(r'T\d{4}(?:\.\d{3})?', str(tid).upper())
        if m:
            out.append(m.group(0))
    return out

def validate_output(stage, obj):
    """Return (normalised_obj, list_of_errors). Empty error list = valid."""
    if not isinstance(obj, dict):
        return None, ['not a JSON object']
    o, err = dict(obj), []
    def need(key, conv, label):
        val = conv(o.get(key))
        if val is None:
            err.append(f'{key}: {label}')
        else:
            o[key] = val
    if stage == 'triage':
        need('severity', _as_sev, 'one of ' + '|'.join(SEV_LEVELS))
        need('escalate', _as_bool, 'boolean')
        o['reasoning'] = str(o.get('reasoning', ''))
    elif stage == 'investigation':
        techs = o.get('techniques')
        ids = _as_tech_ids(techs)
        if ids is None:
            err.append('techniques: list')
        else:
            names = {}
            if isinstance(techs, list):
                for t in techs:
                    if isinstance(t, dict) and t.get('id'):
                        names[str(t['id']).upper()] = {'name': t.get('name', ''), 'tactic': t.get('tactic', '')}
            o['techniques'] = [{'id': i, **names.get(i, {'name': '', 'tactic': ''})} for i in ids]
        need('evidence', _as_str_list, 'list of strings')
        o['gaps'] = _as_str_list(o.get('gaps')) or []
        conf = str(o.get('confidence', '')).lower()
        o['confidence'] = conf if conf in ('high', 'medium', 'low') else 'unknown'
        need('likely_false_positive', _as_bool, 'boolean')
    elif stage == 'response':
        need('immediate_actions', _as_str_list, 'list of strings')
        need('block_source', _as_bool, 'boolean')
        need('escalate', _as_bool, 'boolean')
        o['mitigation_steps'] = _as_str_list(o.get('mitigation_steps')) or []
        o['block_justification'] = str(o.get('block_justification', ''))
        o['reasoning'] = str(o.get('reasoning', ''))
    elif stage == 'coordinator':
        need('final_severity', _as_sev, 'one of ' + '|'.join(SEV_LEVELS))
        need('escalate', _as_bool, 'boolean')
        disp = str(o.get('disposition', '')).lower().replace(' ', '_')
        if disp not in ('true_positive', 'false_positive', 'uncertain'):
            err.append('disposition: true_positive|false_positive|uncertain')
        o['disposition'] = disp
        need('attck_techniques', _as_tech_ids, 'list of technique IDs')
        need('recommended_actions', _as_str_list, 'list of strings')
        o['executive_summary'] = str(o.get('executive_summary', ''))
    return o, err

# Fallbacks are used only when an output is still invalid after the retries;
# they are counted and reported (output-validity table), never hidden.
FALLBACKS = {
    'triage':        {'severity': 'Medium', 'escalate': True, 'reasoning': ''},
    'investigation': {'techniques': [], 'evidence': [], 'gaps': [], 'confidence': 'unknown',
                      'likely_false_positive': False},
    'response':      {'immediate_actions': [], 'block_source': False, 'block_justification': '',
                      'mitigation_steps': [], 'escalate': True, 'reasoning': ''},
    'coordinator':   {'executive_summary': '', 'final_severity': 'Medium', 'escalate': True,
                      'disposition': 'uncertain', 'attck_techniques': [], 'recommended_actions': []},
}


# ── LLM engines: generate(list_of_message_lists, max_new_tokens) -> [(text, n_tokens)]
class HFEngine:
    """Local HuggingFace model, 4-bit NF4, batched generation with left padding.
    Weights are cached on LOCAL disk; a corrupted cache is deleted and re-downloaded once."""
    def __init__(self, cfg, cache_dir):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        a = cfg['agents']
        self.torch = torch
        self.model_id = a['hf_model']
        self.cache_dir = str(cache_dir); os.makedirs(self.cache_dir, exist_ok=True)
        token = os.environ.get('HF_TOKEN') or None
        gpu = torch.cuda.is_available()
        # bfloat16 on GPUs that support it (Ampere / Ada / Hopper), else float16 (T4)
        cdt = torch.bfloat16 if (gpu and a.get('compute_dtype', 'auto') in ('auto', 'bfloat16')
                                 and torch.cuda.is_bf16_supported()) else torch.float16
        bnb = None
        if a.get('hf_use_4bit', True) and gpu:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                     bnb_4bit_compute_dtype=cdt,
                                     bnb_4bit_use_double_quant=True)
        print(f'Loading {self.model_id} (4-bit={bnb is not None}) ...', flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=a.get('hf_revision'),
                                                 token=token, cache_dir=self.cache_dir)
        self.tok.padding_side = 'left'
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        import transformers
        from packaging.version import Version
        dkey = 'dtype' if Version(transformers.__version__) >= Version('4.56') else 'torch_dtype'
        kw = dict(revision=a.get('hf_revision'), quantization_config=bnb,
                  device_map={'': 0} if gpu else None, token=token, cache_dir=self.cache_dir,
                  **{dkey: cdt if gpu else torch.float32})
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'
            if not any(k in msg for k in ('Safetensor', 'header', 'metadata', 'EOF', 'corrupt',
                                          'Unable to load weights', 'invalid load key')):
                raise
            print(f'Cached weights look corrupted ({msg[:120]}); re-downloading ...', flush=True)
            import shutil
            shutil.rmtree(os.path.join(self.cache_dir, 'models--' + self.model_id.replace('/', '--')),
                          ignore_errors=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, force_download=True, **kw)
        self.model.eval()
        self.revision = getattr(self.model.config, '_commit_hash', None) or a.get('hf_revision')
        self.temperature, self.top_p = a['temperature'], a['top_p']
        self.compute_dtype = str(cdt).replace('torch.', '') if gpu else 'float32'
        self.quantization = '4-bit NF4' if bnb is not None else 'none'

    def generate(self, msgs_list, max_new_tokens):
        torch = self.torch
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs_list]
        enc = self.tok(texts, return_tensors='pt', padding=True).to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if self.temperature and self.temperature > 0:
            kw.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        gen = out[:, enc['input_ids'].shape[1]:]
        n_tok = (gen != self.tok.pad_token_id).sum(dim=1).tolist()
        return list(zip(self.tok.batch_decode(gen, skip_special_tokens=True), n_tok))


class APIEngine:
    """Groq / Anthropic through LangChain chat models (sequential calls)."""
    def __init__(self, cfg):
        from langchain_core.messages import SystemMessage, HumanMessage
        self._S, self._H = SystemMessage, HumanMessage
        a = cfg['agents']; self.backend = a['backend']
        if a['backend'] == 'groq':
            from langchain_groq import ChatGroq
            self.model_id = a['groq_model']
            self.llm = ChatGroq(model=self.model_id, temperature=max(a['temperature'], 0),
                                api_key=os.environ.get('GROQ_API_KEY', ''))
        else:
            from langchain_anthropic import ChatAnthropic
            self.model_id = a['anthropic_model']
            self.llm = ChatAnthropic(model=self.model_id, temperature=max(a['temperature'], 0))
        self.revision = self.model_id

    def generate(self, msgs_list, max_new_tokens):
        outs = []
        for m in msgs_list:
            lc = [self._S(content=x['content']) if x['role'] == 'system' else self._H(content=x['content'])
                  for x in m]
            r = self.llm.bind(max_tokens=max_new_tokens).invoke(lc)
            txt = r.content if hasattr(r, 'content') else str(r)
            outs.append((txt, len(txt.split())))
            if self.backend == 'groq':
                time.sleep(0.5)
        return outs


class MockEngine:
    """Deterministic rule-based stand-in for a DRY RUN without an LLM.
    Its outputs are NOT results and must never be reported."""
    model_id, revision = 'mock', 'mock'
    _SEV = {'DDoS': 'Critical', 'Infiltration': 'Critical', 'Heartbleed': 'Critical',
            'PortScan': 'Low', 'BENIGN': 'Informational'}
    def generate(self, msgs_list, max_new_tokens):
        outs = []
        for m in msgs_list:
            sysm, user = m[0]['content'], m[-1]['content']
            if '"status"' in user:
                outs.append(('{"status": "ok"}', 5)); continue
            h = int(hashlib.md5(user.encode()).hexdigest(), 16)
            lab = re.search(r'"predicted_label":\s*"([^"]+)"', user)
            lab = lab.group(1) if lab else 'BENIGN'
            conf = re.search(r'"confidence":\s*([0-9.]+)', user)
            conf = float(conf.group(1)) if conf else 0.5
            esc = lab.upper() != 'BENIGN' and (conf > 0.7 or h % 3 == 0)
            sev = self._SEV.get(lab, 'High' if h % 2 else 'Medium') if esc else 'Low'
            ids = re.findall(r'T\d{4}(?:\.\d{3})?', user) or ['T1046']
            if h % 37 == 0 and 'previous reply' not in user:
                outs.append(('Sorry, here is my analysis without JSON.', 9)); continue
            if 'Triage' in sysm[:60]:
                o = {'severity': sev, 'escalate': esc, 'reasoning': 'mock'}
            elif 'Investigation' in sysm[:60]:
                o = {'techniques': [{'id': ids[h % len(ids)], 'name': '', 'tactic': ''}],
                     'evidence': ['confidence field'], 'gaps': [], 'confidence': 'medium',
                     'likely_false_positive': not esc}
            elif 'Response' in sysm[:60]:
                o = {'immediate_actions': ['Review flow'], 'block_source': esc,
                     'block_justification': '', 'mitigation_steps': ['Patch'],
                     'escalate': esc, 'reasoning': 'mock'}
            else:
                o = {'executive_summary': 'mock', 'final_severity': sev, 'escalate': esc,
                     'disposition': 'true_positive' if esc else 'false_positive',
                     'attck_techniques': ids[:1], 'recommended_actions': ['Review']}
            outs.append((json.dumps(o), 40))
        return outs


def make_engine(cfg, cache_dir):
    b = cfg['agents']['backend']
    if b == 'huggingface':
        return HFEngine(cfg, cache_dir)
    if b == 'mock':
        return MockEngine()
    return APIEngine(cfg)


# ── ATT&CK retrieval (FAISS cosine over MiniLM embeddings) ───────────────────
class RAG:
    def __init__(self, kb_dir, embedding_model, top_k):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.faiss = faiss
        self.index = faiss.read_index(os.path.join(kb_dir, 'faiss_index.bin'))
        self.techniques = json.load(open(os.path.join(kb_dir, 'techniques.json')))
        self.embedder = SentenceTransformer(embedding_model)
        self.top_k = top_k

    def query(self, text, top_k=None):
        q = self.embedder.encode([text], convert_to_numpy=True).astype('float32')
        self.faiss.normalize_L2(q)
        sims, idxs = self.index.search(q, top_k or self.top_k)
        T = self.techniques
        return [{'id': T[i]['id'], 'name': T[i]['name'], 'tactics': ', '.join(T[i]['tactics']),
                 'description_snippet': T[i]['description'][:400], 'score': round(float(s), 4)}
                for s, i in zip(sims[0], idxs[0]) if 0 <= i < len(T)]

def build_rag_query(alert):
    """Retrieval query = predicted class + service + flag text (documented in the paper)."""
    q = f"{alert.get('predicted_label', '')} attack on {alert.get('service', alert.get('context', {}).get('service', ''))} service"
    if alert.get('flag_anomaly') not in (None, '', 'None'):
        q += f", TCP flags: {alert['flag_anomaly']}"
    return q


# ── Prompts (exported verbatim for the paper appendix) ──────────────────────
ESCALATE_DEF = ('"escalate": true means forward the alert to a human analyst as a likely real attack; '
                'false means it is most likely a false positive or benign activity and can be closed. '
                'The ML detector can be wrong: judge whether the evidence is consistent with its label.')

PROMPTS = {
'triage': f"""You are a Tier-1 SOC Triage Analyst. Given a structured alert packet produced by a
machine-learning network intrusion detector, assign a severity tier and decide escalation.

Consider: the detector's confidence and attack probability; whether the evidence features
(with their SHAP attributions) are consistent with the predicted attack; the targeted service;
and the TCP flag pattern.
Severity tiers: Critical | High | Medium | Low | Informational.
{ESCALATE_DEF}

Output ONLY valid JSON, no text outside it:
{{"severity": "Critical|High|Medium|Low|Informational", "escalate": true|false,
  "reasoning": "2-3 sentences"}}""",

'investigation': f"""You are a SOC Investigation Analyst. Given an alert, the triage result and MITRE ATT&CK
context, identify the ATT&CK techniques the activity corresponds to.

Rules: list at most 3 techniques, most relevant first; prefer techniques that appear in the
provided ATT&CK context; use exact ATT&CK IDs (e.g. T1499 or T1499.002). Each evidence item must
cite a specific field of the alert packet. If the evidence does not support an attack, set
"likely_false_positive" to true and return an empty technique list.

Output ONLY valid JSON:
{{"techniques": [{{"id": "T1498", "name": "...", "tactic": "..."}}],
  "evidence": ["field: observation"], "gaps": ["missing information"],
  "confidence": "high|medium|low", "likely_false_positive": true|false}}""",

'response': f"""You are a SOC Response Analyst. Based on the alert, triage and investigation, recommend
containment and mitigation. All actions are ADVISORY: a human analyst must approve them.
Recommend blocking the source only when the evidence supports a real attack.
{ESCALATE_DEF}

Output ONLY valid JSON:
{{"immediate_actions": ["action"], "block_source": true|false, "block_justification": "reason",
  "mitigation_steps": ["step"], "escalate": true|false, "reasoning": "2-3 sentences"}}""",

'coordinator': f"""You are the SOC Coordinator. Combine the triage, investigation and response outputs into
the final report for a human analyst. Resolve disagreements between agents explicitly.
{ESCALATE_DEF}
"disposition": "true_positive" (real attack), "false_positive" (benign traffic mislabelled by the
detector) or "uncertain".

Output ONLY valid JSON:
{{"executive_summary": "2-3 sentences", "final_severity": "Critical|High|Medium|Low|Informational",
  "disposition": "true_positive|false_positive|uncertain", "escalate": true|false,
  "attck_techniques": ["T1498"], "recommended_actions": ["action"]}}""",
}
REPAIR = ('Your previous reply was not valid for the required format ({errors}). '
          'Reply again with ONLY a JSON object that follows the schema in the instructions.')

VARIANTS = {
    'full':             dict(rag=True,  shap=True,  investigation=True),
    'no_rag':           dict(rag=False, shap=True,  investigation=True),
    'no_shap':          dict(rag=True,  shap=False, investigation=True),
    'no_investigation': dict(rag=True,  shap=True,  investigation=False),
}
STAGES = ['triage', 'investigation', 'response', 'coordinator']


def _alert_view(alert, v):
    a = copy.deepcopy(alert)
    if not v['shap']:
        a.pop('evidence', None); a.pop('evidence_source', None)
    return a

def _user_msg(stage, st, v):
    A = json.dumps(st['alert'], indent=1)
    T = json.dumps(st.get('triage'), indent=1)
    I = (json.dumps(st['investigation'], indent=1) if st.get('investigation') is not None
         else '(investigation stage disabled)')
    if stage == 'triage':
        return f'Alert packet:\n{A}'
    if stage == 'investigation':
        ctx = (json.dumps(st['rag_hits'], indent=1) if v['rag'] else
               '(no ATT&CK context provided; use your own knowledge of ATT&CK)')
        return f'Alert:\n{A}\n\nMITRE ATT&CK context:\n{ctx}\n\nTriage:\n{T}'
    if stage == 'response':
        return f'Alert:\n{A}\n\nTriage:\n{T}\n\nInvestigation:\n{I}'
    R = json.dumps(st.get('response'), indent=1)
    return f'Alert:\n{A}\n\nTriage:\n{T}\n\nInvestigation:\n{I}\n\nResponse:\n{R}'

def run_stage(stage, states, v, eng, cfg):
    max_new = cfg['agents']['max_new_tokens'][stage]
    msgs = [[{'role': 'system', 'content': PROMPTS[stage]},
             {'role': 'user', 'content': _user_msg(stage, st, v)}] for st in states]
    t0 = time.time(); outs = eng.generate(msgs, max_new); dt = time.time() - t0
    meta = [{'retries': 0, 'out_tokens': n, 'raw': txt[:4000]} for txt, n in outs]
    parsed = [validate_output(stage, extract_json(txt)) for txt, _ in outs]
    for _ in range(cfg['agents']['max_retries']):
        bad = [k for k, (o, e) in enumerate(parsed) if o is None or e]
        if not bad:
            break
        rmsgs = [msgs[k] + [{'role': 'assistant', 'content': outs[k][0]},
                            {'role': 'user', 'content': REPAIR.format(errors='; '.join(parsed[k][1]))}]
                 for k in bad]
        t1 = time.time(); routs = eng.generate(rmsgs, max_new); dt += time.time() - t1
        for k, (txt, n) in zip(bad, routs):
            outs[k] = (txt, n); parsed[k] = validate_output(stage, extract_json(txt))
            meta[k]['retries'] += 1; meta[k]['out_tokens'] += n; meta[k]['raw'] = txt[:4000]
    for st, (o, e), m in zip(states, parsed, meta):
        ok = o is not None and not e
        m.update(valid_first_try=(m['retries'] == 0 and ok), valid=ok, fallback=not ok,
                 errors=e if o is not None else ['no JSON object'])
        st[stage] = o if ok else copy.deepcopy(FALLBACKS[stage])
        st[f'{stage}_meta'] = m
        st[f'{stage}_time'] = dt / len(states)

def run_variant(variant, seed, alert_ids, out_path, alerts, gt, eng, cfg, rag,
                batch_size=None, log=print):
    """Run one system variant over `alert_ids`; resumable (one JSON file per run)."""
    v = VARIANTS[variant]
    bs = batch_size or cfg['agents']['batch_size']
    out_path = str(out_path)
    done = {r['alert_id']: r for r in json.load(open(out_path))} if os.path.exists(out_path) else {}
    todo = [a for a in alert_ids if a not in done]
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    t_start = time.time()
    for b_i, ids in enumerate(batches):
        set_global_seed(seed * 100_003 + alert_ids.index(ids[0]))   # independent of resume point
        states = [{'alert_id': a, 'variant': variant, 'seed': seed, 'ground_truth': gt[a],
                   'alert': _alert_view(alerts[a], v)} for a in ids]
        try:
            run_stage('triage', states, v, eng, cfg)
            if v['investigation']:
                for st in states:
                    st['rag_hits'] = rag.query(build_rag_query(st['alert'])) if v['rag'] else []
                run_stage('investigation', states, v, eng, cfg)
            else:
                for st in states:
                    st['investigation'] = None; st['investigation_time'] = 0.0
            run_stage('response', states, v, eng, cfg)
            run_stage('coordinator', states, v, eng, cfg)
        except Exception as ex:
            log(f'batch {b_i} failed: {ex}')
            for st in states:
                st['error'] = str(ex)
        for st in states:
            st['total_time'] = sum(st.get(f'{s}_time', 0.0) for s in STAGES)
            done[st['alert_id']] = st
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(list(done.values()), f, default=str)
        os.replace(tmp, out_path)                       # atomic: never a half-written file
        el = time.time() - t_start; n_done = sum(len(b) for b in batches[:b_i + 1])
        eta = el / max(n_done, 1) * (len(todo) - n_done)
        log(f'{variant} seed={seed}: {len(done)}/{len(alert_ids)} alerts '
            f'({el / 60:.1f} min, ETA {eta / 60:.1f} min)')
    return [done[a] for a in alert_ids if a in done]


# =============================================================================
# v3 — CONTROLLED EXPLANATION EXPERIMENTS
# =============================================================================
# A v3 alert packet has separate blocks so that every condition shows the SAME
# raw measurements and only adds / removes detector information:
#   measurements : fixed list of raw flow features (identical in all conditions)
#   context      : service, port category, duration bucket, ports, packet counts
#   detector     : predicted_label, confidence, p_attack
#   flipped      : the detector output with the attack/benign decision inverted
#   shap         : top-5 SHAP attributions for the predicted class (genuine)
#   shap_shuffled: 5 random features with attribution values taken from another
#                  alert (realistic magnitudes, no information about this flow)
# No alert identifier, stratum, or label is ever shown to the model.
CONDITIONS = {
    'raw':           dict(detector=None,      shap=None,       wording=None),
    'pred':          dict(detector='genuine', shap=None,       wording=None),
    'pred_shap':     dict(detector='genuine', shap='genuine',  wording='standard'),
    'pred_shuffled': dict(detector='genuine', shap='shuffled', wording='standard'),
    'pred_shap_explicit': dict(detector='genuine', shap='genuine', wording='explicit'),
    'flipped':       dict(detector='flipped', shap=None,       wording=None),
}

SHAP_NOTE = {
    'standard': 'SHAP (XGBoost, predicted class)',
    'explicit': ("SHAP attributions for the detector's predicted class '{label}'. A positive value "
                 "means the feature pushed the detector TOWARD '{label}', a negative value pushed it "
                 "away. They describe how the detector reached its label; they are not independent "
                 "evidence that the flow is malicious."),
}

def v3_view(packet, condition):
    """Build exactly what the model sees for one condition."""
    c = CONDITIONS[condition]
    view = {'flow_measurements': packet['measurements'], 'flow_context': packet['context']}
    if c['detector'] == 'genuine':
        view['detector_output'] = packet['detector']
    elif c['detector'] == 'flipped':
        view['detector_output'] = packet['flipped']
    if c['shap']:
        attr = packet['shap'] if c['shap'] == 'genuine' else packet['shap_shuffled']
        lab = packet['detector']['predicted_label']
        view['detector_explanation'] = {'method': SHAP_NOTE[c['wording']].format(label=lab),
                                        'attributions': attr}
    return view

_ESC_V3 = ('"escalate": true means forward the flow to a human analyst as a likely real attack; '
           'false means it is most likely benign and can be closed.')
_OUT_V3 = ('Output ONLY valid JSON, no text outside it:\n'
           '{"severity": "Critical|High|Medium|Low|Informational", "escalate": true|false,\n'
           '  "reasoning": "2-3 sentences"}')

def triage_prompt_v3(condition, prompt_set='P1'):
    """Triage system prompt for a condition. P1 and P2 are two independent phrasings
    (prompt-robustness check); both describe only the blocks that are present."""
    c = CONDITIONS[condition]
    has_det, has_shap = c['detector'] is not None, c['shap'] is not None
    if prompt_set == 'P1':
        parts = ['You are a Tier-1 SOC Triage Analyst. You receive the measurements of one network flow']
        if has_det:
            parts[0] += ', the output of a machine-learning intrusion detector'
        if has_shap:
            parts[0] += " and the detector's feature attributions"
        parts[0] += '.'
        parts.append('Decide whether the flow is a likely real attack that must be forwarded to a human '
                     'analyst, and assign a severity tier. Base your judgement on what the measurements show.')
        if has_det:
            parts.append('The detector can be wrong: check whether the measurements are consistent with its output.')
        parts.append('Severity tiers: Critical | High | Medium | Low | Informational.')
        parts.append(_ESC_V3)
        parts.append(_OUT_V3)
        return '\n'.join(parts)
    # P2: different wording and order
    src = ['flow measurements']
    if has_det: src.append('an ML detector verdict')
    if has_shap: src.append('attribution scores for that verdict')
    lines = ['Role: first-line security analyst (SOC Triage, Tier 1).',
             'Input: ' + ', '.join(src) + ' for a single network connection.',
             'Task 1: decide "escalate" (true = send to a human analyst, it is probably an attack; '
             'false = close it, it is probably normal traffic).',
             'Task 2: choose a severity: Critical, High, Medium, Low or Informational.']
    if has_det:
        lines.append('Treat the detector verdict as a hint that may be mistaken, not as a fact.')
    lines.append('Answer with a single JSON object and nothing else, using exactly these keys:')
    lines.append('{"severity": "...", "escalate": true or false, "reasoning": "2-3 sentences"}')
    return '\n'.join(lines)

def build_rag_query_v3(view):
    lab = (view.get('detector_output') or {}).get('predicted_label', 'suspicious network')
    svc = view.get('flow_context', {}).get('service', '')
    return f'{lab} attack on {svc} service'

def run_condition(condition, prompt_set, seed, alert_ids, out_path, packets, gt, eng, cfg, rag,
                  stages='triage', batch_size=None, log=print):
    """Run one controlled condition (v3 packets). stages='triage' runs only the
    Triage agent (the escalation decision is fixed there, see the v2 results);
    stages='full' runs all four agents. Resumable, one JSON file per run."""
    bs = batch_size or cfg['agents']['batch_size']
    out_path = str(out_path)
    done = {r['alert_id']: r for r in json.load(open(out_path))} if os.path.exists(out_path) else {}
    todo = [a for a in alert_ids if a not in done]
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    sys_prompt = triage_prompt_v3(condition, prompt_set)
    v_full = dict(rag=True, shap=CONDITIONS[condition]['shap'] is not None, investigation=True)
    t_start = time.time()
    for b_i, ids in enumerate(batches):
        set_global_seed(seed * 100_003 + alert_ids.index(ids[0]))
        states = [{'alert_id': a, 'condition': condition, 'prompt_set': prompt_set, 'seed': seed,
                   'ground_truth': gt[a], 'alert': v3_view(packets[a], condition)} for a in ids]
        try:
            # Triage with the condition-specific prompt
            msgs = [[{'role': 'system', 'content': sys_prompt},
                     {'role': 'user', 'content': 'Flow record:\n' + json.dumps(st['alert'], indent=1)}]
                    for st in states]
            max_new = cfg['agents']['max_new_tokens']['triage']
            t0 = time.time(); outs = eng.generate(msgs, max_new); dt = time.time() - t0
            parsed = [validate_output('triage', extract_json(t)) for t, _ in outs]
            meta = [{'retries': 0, 'out_tokens': n, 'raw': t[:4000]} for t, n in outs]
            for _ in range(cfg['agents']['max_retries']):
                bad = [k for k, (o, e) in enumerate(parsed) if o is None or e]
                if not bad:
                    break
                rmsgs = [msgs[k] + [{'role': 'assistant', 'content': outs[k][0]},
                                    {'role': 'user', 'content': REPAIR.format(errors='; '.join(parsed[k][1]))}]
                         for k in bad]
                t1 = time.time(); routs = eng.generate(rmsgs, max_new); dt += time.time() - t1
                for k, (t, n) in zip(bad, routs):
                    outs[k] = (t, n); parsed[k] = validate_output('triage', extract_json(t))
                    meta[k]['retries'] += 1; meta[k]['out_tokens'] += n; meta[k]['raw'] = t[:4000]
            for st, (o, e), m in zip(states, parsed, meta):
                ok = o is not None and not e
                m.update(valid_first_try=(m['retries'] == 0 and ok), valid=ok, fallback=not ok,
                         errors=e if o is not None else ['no JSON object'])
                st['triage'] = o if ok else copy.deepcopy(FALLBACKS['triage'])
                st['triage_meta'] = m; st['triage_time'] = dt / len(states)
            if stages == 'full':
                for st in states:
                    st['rag_hits'] = rag.query(build_rag_query_v3(st['alert']))
                run_stage('investigation', states, v_full, eng, cfg)
                run_stage('response', states, v_full, eng, cfg)
                run_stage('coordinator', states, v_full, eng, cfg)
        except Exception as ex:
            log(f'batch {b_i} failed: {ex}')
            for st in states:
                st['error'] = str(ex)
        for st in states:
            st['total_time'] = sum(st.get(f'{s}_time', 0.0) for s in STAGES)
            done[st['alert_id']] = st
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(list(done.values()), f, default=str)
        os.replace(tmp, out_path)
        el = time.time() - t_start; n_done = sum(len(b) for b in batches[:b_i + 1])
        eta = el / max(n_done, 1) * (len(todo) - n_done)
        log(f'{condition}/{prompt_set}/{stages} seed={seed}: {len(done)}/{len(alert_ids)} alerts '
            f'({el / 60:.1f} min, ETA {eta / 60:.1f} min)')
    return [done[a] for a in alert_ids if a in done]

def serialize_messages(condition, prompt_set, packet):
    """Exact chat messages sent to the Triage agent (for the paper appendix)."""
    return [{'role': 'system', 'content': triage_prompt_v3(condition, prompt_set)},
            {'role': 'user', 'content': 'Flow record:\n' + json.dumps(v3_view(packet, condition), indent=1)}]
