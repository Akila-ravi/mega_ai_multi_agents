# Architecture

## Text Diagram

Client (SSE)
  -> API `/query`
  -> DB queue `jobs(status=queued)`
  -> Worker polls queue
  -> Orchestrator runtime route decision
      -> Decomposition Agent (task graph)
      -> Retrieval Agent (multi-hop over 2+ chunks via web tool)
      -> Critique Agent (claim confidence + disagree spans)
      -> Synthesis Agent (contradiction resolution + provenance)
      -> Tool calls (web/python/sql/reflection) with retry/failure contracts
  -> Persist traces (`agent_logs`, `tool_calls`)
  -> API streams progress/events and final answer

Eval loop
  -> `/eval/retry`
  -> 15-case harness, multidim scores
  -> store `eval_runs`
  -> meta-agent prompt rewrite proposal -> `prompt_versions (pending)`
  -> `/prompt/approve` approve/reject

## Module map
- `app/agents`: sub-agent logic
- `app/orchestrator`: routing + handoff mediation
- `app/tools`: tool contracts + execution
- `app/context_manager`: budget control + compression
- `app/evaluation`: test harness + scoring + prompt proposal
- `app/api`: (integrated in main)
- `app/worker`: async queue processing
- `app/db`: models/session/init
