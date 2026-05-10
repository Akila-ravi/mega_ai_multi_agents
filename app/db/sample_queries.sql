-- Sample operational queries
SELECT id, status, created_at FROM jobs ORDER BY created_at DESC LIMIT 20;
SELECT job_id, agent_id, event_type, latency_ms FROM agent_logs ORDER BY id DESC LIMIT 50;
SELECT job_id, tool_name, accepted, retry_index FROM tool_calls ORDER BY id DESC LIMIT 50;
SELECT id, run_type, summary FROM eval_runs ORDER BY created_at DESC LIMIT 5;
SELECT id, prompt_key, decision, created_at FROM prompt_versions ORDER BY created_at DESC LIMIT 20;
