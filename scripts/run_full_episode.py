import requests, json, time
url = 'http://127.0.0.1:7860'

def pretty(obj):
    try:
        print(json.dumps(obj, indent=2))
    except Exception:
        print(obj)

print('POST /reset (easy)')
r = requests.post(url+'/reset', json={'task':'easy'}, timeout=10)
pretty(r.json())

time.sleep(0.1)
print('\nGET /state')
r = requests.get(url+'/state', timeout=10)
state = r.json()
pretty(state)

candidates = state.get('candidates') or state.get('candidates_full') or []
# Normalize candidate list
cands = []
for c in candidates:
    if isinstance(c, dict):
        cid = c.get('candidate_id') or c.get('id')
        resume = c.get('resume_score') or c.get('resume') or 0
        cands.append({'candidate_id': cid, 'resume_score': resume})

# sort by resume_score desc
cands.sort(key=lambda x: x['resume_score'], reverse=True)

print('\nInterviewing candidates in resume order (top to bottom)')
for c in cands:
    cid = c['candidate_id']
    print(f"POST /step interview {cid}")
    try:
        r = requests.post(url+'/step', json={'action':'interview', 'candidate_id':cid}, timeout=10)
        pretty(r.json())
    except Exception as e:
        print('ERROR:', e)
    time.sleep(0.05)

print('\nGET /state after interviews')
r = requests.get(url+'/state', timeout=10)
state = r.json()
pretty(state)

# decide hire: prefer interviewed candidates' interview_score if available
interviews = state.get('interviews_done', {})
best_cid = None
best_score = -999
if interviews:
    for cid, score in interviews.items():
        if score > best_score:
            best_score = score
            best_cid = cid
else:
    # fallback to resume ordering
    if cands:
        best_cid = cands[0]['candidate_id']

if best_cid:
    print(f"\nPOST /step hire {best_cid}")
    r = requests.post(url+'/step', json={'action':'hire', 'candidate_id': best_cid}, timeout=10)
    pretty(r.json())

print('\nPOST /step finalize')
r = requests.post(url+'/step', json={'action':'finalize'}, timeout=10)
pretty(r.json())

print('\nFinal /state')
r = requests.get(url+'/state', timeout=10)
pretty(r.json())
