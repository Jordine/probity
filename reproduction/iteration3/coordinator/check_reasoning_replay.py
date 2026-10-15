import json
from pathlib import Path
from transformers import AutoTokenizer
from probity_benchmark.qwen_extract import serialize
from probity_benchmark.prepare_debate_corpus import returned_reasoning
from probity_benchmark.preflight import sha256

p = Path('runs/iteration3')
records = list(map(json.loads, (p/'environment/reasoned_prepare_v2/requests.jsonl').read_text().splitlines()))
results = []
for model, revision in [('Qwen/Qwen3.5-27B','fc05daec18b0a78c049392ed2e771dde82bdf654'),('Qwen/Qwen3.8-27B','1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')]:
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
    lengths, trace_chars = [], []
    for r in records:
        if r['model'].lower() != model.lower():
            continue
        reasoning = returned_reasoning(r)
        final = json.loads(r['response_body'])['choices'][0]['message']['content']
        variants = [serialize(tokenizer, r['request']['messages'], final, arm, reasoning=reasoning) for arm in ('Yes.','No.')]
        for s in variants:
            a,b = s['returned_reasoning_range']
            assert s['text'][a:b] == reasoning
            assert s['returned_reasoning_sha256'] == sha256(reasoning)
            assert s['text'][s['response_start']:s['response_start']+len(final)] == final
            decoded = tokenizer.decode([s['ids'][i] for i in s['response_positions']], clean_up_tokenization_spaces=False, skip_special_tokens=False)
            assert decoded == final
            assert all(s['offsets'][i][0] >= b for i in s['response_positions'])
        yes,no = variants
        assert yes['ids'][:yes['followup_start']] == no['ids'][:no['followup_start']]
        lengths.append(len(yes['ids']))
        trace_chars.append(len(reasoning))
    results.append({'model':model,'revision':revision,'rows':len(lengths),'context_tokens_min':min(lengths),'context_tokens_max':max(lengths),'reasoning_characters_min':min(trace_chars),'reasoning_characters_max':max(trace_chars),'complete_reasoning_and_final_roundtrips':True,'final_pooling_excludes_reasoning':True,'dyl_original_context_unchanged':True})
out={'results':results,'rows_expected':32,'rows_checked':sum(x['rows'] for x in results),'claim_limit':'Complete returned trace under an explicit offline think-block serialization; opaque provider internals remain unverified.'}
assert out['rows_checked']==out['rows_expected']
(p/'coordinator/reasoning_replay_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
