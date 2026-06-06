# benchmark results

aggregated from `results/benchmarks/<suite>/<format>/summary.json` envelopes.
regenerate: `telos eval-aggregate-results` (reads `/home/kaarl/Developer/talos/results/benchmarks`).

## matrix

| suite | format | model | n | primary | secondary | avg_tokens | tok/success | avg_wall_sec |
|-------|--------|-------|---|---------|-----------|------------|-------------|--------------|
| bfcl | telos | telos-llama3.1-8b-lora-merged | 5 | 60.0% (accuracy) | 8.8 (avg_retry_count) | 36570 | 216 | 281.2 |
| bfcl | chatml | chatml-llama3.1-8b-lora-merged | 5 | 60.0% (accuracy) | 1.6 (avg_retry_count) | 20139 | 603 | 192.0 |
| toolbench | telos | telos-llama3.1-8b-lora-merged | 2 | 0.0% (pass_rate) | 12.0 (avg_steps) | 30063 | — | 735.8 |
| toolbench | chatml | chatml-llama3.1-8b-lora-merged | 2 | 0.0% (pass_rate) | 12.0 (avg_steps) | 40350 | — | 848.2 |
| swe | telos | telos-llama3.1-8b-lora-merged | 2 | 0.0% (resolved_rate) | 3.5 (avg_iterations) | 10884 | — | 221.2 |
| format_validity | telos | telos-llama3.1-8b-lora-merged | 3 | 100.0% (valid_rate) | 100.0% (parse_rate) | 139 | 139 | 31.0 |

## coverage

missing cells (not run yet):
- `swe/chatml`
- `format_validity/chatml`

## per-suite metrics

| suite | primary | secondary |
|-------|---------|-----------|
| format_validity | valid_rate | parse_rate |
| bfcl | accuracy | avg_retry_count |
| toolbench | pass_rate (structural) | avg_steps |
| swe | resolved_rate | avg_iterations |

## sources

- `bfcl` / `telos`: `results/benchmarks/bfcl/telos/summary.json`
- `bfcl` / `chatml`: `results/benchmarks/bfcl/chatml/summary.json`
- `toolbench` / `telos`: `results/benchmarks/toolbench/telos/summary.json`
- `toolbench` / `chatml`: `results/benchmarks/toolbench/chatml/summary.json`
- `swe` / `telos`: `results/benchmarks/swe/telos/summary.json`
- `format_validity` / `telos`: `results/benchmarks/format_validity/telos/summary.json`
