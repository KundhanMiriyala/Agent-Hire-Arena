import requests, json, time
url='http://127.0.0.1:7860'

def pretty(resp):
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)

print("GET /tasks")
try:
    r = requests.get(url+'/tasks', timeout=10)
    pretty(r)
except Exception as e:
    print('ERROR /tasks:', e)

print("\nPOST /reset with task=easy")
try:
    r = requests.post(url+'/reset', json={'task':'easy'}, timeout=10)
    pretty(r)
except Exception as e:
    print('ERROR /reset:', e)

time.sleep(0.2)
print("\nGET /state")
state = None
try:
    r = requests.get(url+'/state', timeout=10)
    pretty(r)
    try:
        state = r.json()
    except Exception:
        state = None
except Exception as e:
    print('ERROR /state:', e)

if state and isinstance(state.get('candidates'), list) and state['candidates']:
    first = state['candidates'][0]
    cid = None
    if isinstance(first, dict):
        cid = first.get('id') or first.get('candidate_id') or first.get('candidateId')
    elif isinstance(first, str):
        cid = first
    if cid:
        print(f"\nPOST /step interview candidate_id={cid}")
        try:
            r = requests.post(url+'/step', json={'action':'interview','candidate_id':cid}, timeout=10)
            pretty(r)
        except Exception as e:
            print('ERROR /step interview:', e)

print('\nDone')
