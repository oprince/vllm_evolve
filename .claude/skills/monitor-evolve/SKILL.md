---
name: monitor-evolve
description: Monitor an evolve (skydiscover/adaevolve) batch job running on CCC (IBM LSF cluster). Asks for hostname and job ID, then reports progress every 30 seconds — job status, current iteration, best score, errors, and resource usage.
---

# Monitor Evolve Process on CCC

Monitor a running skydiscover/adaevolve evolution job on the CCC IBM LSF cluster.

## Step 1: Collect parameters

Ask the user (via AskUserQuestion) for:
1. **CCC login host** — e.g. `ccc-login3` or `ccc-login5` (just the short name; the domain is `.pok.ibm.com`)
2. **LSF Job ID** — the numeric batch job ID from `bsub` output

Defaults:
- Host: `ccc-login3`
- SSH key: `/Users/oritp/.ssh/ccc_rsa`
- User: `oprince9`
- Working directory: `/proj/vlt-evolve/work/orit/vllm`

## Step 2: Start monitoring

Use the Monitor tool with a polling script that runs every 30 seconds. The script should SSH to the CCC host and gather:

### Job status check
```bash
ssh -i /Users/oritp/.ssh/ccc_rsa -o StrictHostKeyChecking=no oprince9@<HOST>.pok.ibm.com \
  "bjobs -o 'jobid name stat run_time exec_host' -noheader <JOBID> 2>&1"
```

### Live output (last progress lines from bpeek)
```bash
ssh -i /Users/oritp/.ssh/ccc_rsa -o StrictHostKeyChecking=no oprince9@<HOST>.pok.ibm.com \
  "bpeek <JOBID> 2>&1 | grep -E '(Evaluated program|New global best|iteration|Iteration|WARNING|ERROR|Evolution complete|best_program)' | tail -10"
```

### Iteration stats (if available)
```bash
ssh -i /Users/oritp/.ssh/ccc_rsa -o StrictHostKeyChecking=no oprince9@<HOST>.pok.ibm.com \
  "cd /proj/vlt-evolve/work/orit/vllm && tail -1 skydiscover_output/adaevolve_iteration_stats_*.jsonl 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print(f'Iter {d.get(\\\"iteration\\\",\\\"?\\\")}/{d.get(\\\"total_iterations\\\",\\\"?\\\")}: best={d.get(\\\"global_best_score\\\",0):.4f}, improved={d.get(\\\"improved\\\",False)}')\" 2>/dev/null"
```

## Step 3: Monitor script template

Use the Monitor tool with `persistent: true` and a combined polling script:

```bash
while true; do
  STATUS=$(ssh -i /Users/oritp/.ssh/ccc_rsa -o StrictHostKeyChecking=no -o ConnectTimeout=10 oprince9@<HOST>.pok.ibm.com "bjobs -o 'stat run_time' -noheader <JOBID>" 2>/dev/null || echo "SSH_FAILED")
  
  if echo "$STATUS" | grep -q "not found\|is not found\|SSH_FAILED"; then
    echo "[$(date +%H:%M:%S)] Job <JOBID>: NOT FOUND or SSH failed — $STATUS"
    break
  fi
  
  if echo "$STATUS" | grep -q "EXIT\|DONE"; then
    echo "[$(date +%H:%M:%S)] Job <JOBID>: FINISHED — $STATUS"
    break
  fi

  PROGRESS=$(ssh -i /Users/oritp/.ssh/ccc_rsa -o StrictHostKeyChecking=no -o ConnectTimeout=10 oprince9@<HOST>.pok.ibm.com "bpeek <JOBID> 2>&1 | grep -E '(Evaluated program|New global best|Iteration|iteration|WARNING.*Iteration|ERROR|Evolution complete)' | tail -5" 2>/dev/null || echo "bpeek failed")

  echo "[$(date +%H:%M:%S)] Job <JOBID> | $STATUS | $PROGRESS"
  
  sleep 30
done
```

## Key patterns to watch for in output

- `[evaluator] Evaluated program <ID> in <N>s: combined_score=X.XXXX` — a new candidate was evaluated
- `New global best: <ID> with fitness X.XXXX` — improvement found
- `WARNING [search.adaevolve] Iteration N: All 3 attempts failed` — iteration failed (LLM errors)
- `ERROR [llm] All 4 attempts failed` — LLM backend is down
- `Evolution complete!` — run finished
- `Graceful shutdown requested` — job was killed (SIGINT/SIGTERM)

## Step 4: Report format

Each 30-second report should be a single concise line:
```
[HH:MM:SS] Job <ID> | STAT: RUN | Runtime: 1:23 | Last eval: score=0.8270 | Best: 0.8450 (iter 3/40)
```

If errors are detected, flag them prominently. If the job ends (DONE/EXIT), report the final status and stop the monitor.
