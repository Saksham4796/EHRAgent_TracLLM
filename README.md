EhrAgent_TracLLM pairs the EHR QA agent with TracLLM-based context attribution to rerank memory when answers look wrong.  
Autogen runs generate execution logs that feed TracLLM, which scores each memory record to guide replacements.  
Configure `.env` (see `.env.sample`), install `pip install -r requirements.txt`, and run `python3 EhrAgent/ehragent/main.py`.
