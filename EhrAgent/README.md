# Enhancing EHR-Agent with TracLLM

This project now uses TracLLM for memory attribution and reranking. Autogen execution logs are fed into TracLLM to score each memory record and iteratively swap out unhelpful items when the agent answers a question incorrectly.
