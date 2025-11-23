EhrAgent_TracLLM pairs the EHR QA agent with TracLLM-based context attribution to rerank memory when answers look wrong. Autogen runs generate execution logs that feed TracLLM, which scores each memory record to guide replacements. Configure `.env` (see `.env.sample`), install `pip install -r requirements.txt`, and run `python3 EhrAgent/ehragent/main.py`.

Model selection:
- `LLM_MODEL` controls the EhrAgent generation model.
- TracLLM can be configured with a JSON file via `TRACLLM_MODEL_CONFIG` or directly with `TRACLLM_MODEL` (uses `OPENAI_API_KEY` by default).
- `TRACLLM_MODE` sets attribution granularity: `overall` scores the whole memory record; `partition` scores question/knowledge/solution separately and uses the max per record.

Example `.env` snippet:
```env
LLM_MODEL=gpt-4o
TRACLLM_MODEL_CONFIG=TracLLM/model_configs/gpt4o-mini_config.json
# or TRACLLM_MODEL=gpt-4o-mini
# TRACLLM_MODE=partition
```
