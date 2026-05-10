import asyncio, json, os
import httpx

async def run_checks():
    base = os.getenv("BASE_URL", "http://127.0.0.1:8001")
    async with httpx.AsyncClient(timeout=40) as c:
        er = (await c.post(f'{base}/eval/retry', json={'run_type':'failed_only'})).json()
        assert 'summary' in er and 'prompt_proposal' in er
        s = (await c.get(f'{base}/eval/summary')).json()
        assert 'total' in s
        ar = (await c.post(f'{base}/prompt/approve', json={'prompt_version_id': er['prompt_proposal']['id'], 'decision':'approved'})).json()
        assert ar['decision'] == 'approved'

        job_id = None
        event_count = 0
        tool_events = 0
        async with c.stream('POST', f'{base}/query', json={'query':'Explain eventual consistency'}) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if not line.startswith('data: '):
                    continue
                event_count += 1
                payload = json.loads(line[6:])
                if 'tool_call' in payload:
                    tool_events += 1
                if payload.get('job_id'):
                    job_id = payload['job_id']
                    break
        assert job_id, 'No final job_id seen in SSE stream'
        tr = (await c.get(f'{base}/trace/{job_id}')).json()
        assert tr['job_id'] == job_id
        assert len(tr['logs']) >= 2
        assert len(tr['tool_calls']) >= 2
        print('OK', {'event_count': event_count, 'tool_events': tool_events, 'job_id': job_id, 'trace_logs': len(tr['logs']), 'trace_tools': len(tr['tool_calls'])})

asyncio.run(run_checks())
