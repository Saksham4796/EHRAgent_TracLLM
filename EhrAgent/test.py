from ehragent.tracllm_wrapper import TracLLMWrapper


def main():
    query = "How should I calculate average ICU stay length and debug missing values?"
    autogen_log = """[autogen]
user: Please write SQL to get the average ICU stay length.
assistant: Loaded admissions and icustays tables.
assistant: Solution:
SELECT AVG(strftime('%s', dischtime) - strftime('%s', intime)) / 86400.0 AS avg_los
FROM icustays;
assistant: TERMINATE"""

    memory_records = [
        {
            "question": "Compute ICU length of stay",
            "knowledge": "Length of stay is discharge minus admit time; prefer hours.",
            "code": "SELECT AVG(julianday(dischtime) - julianday(intime)) * 24 AS avg_hours FROM icustays;",
        },
        {
            "question": "Handle null timestamps",
            "knowledge": "Drop rows missing intime/dischtime before aggregation.",
            "code": "SELECT AVG(...) FROM icustays WHERE intime IS NOT NULL AND dischtime IS NOT NULL;",
        },
    ]

    wrapper = TracLLMWrapper(
        config_path="TracLLM/model_configs/gpt4o_config.json",
        device="cpu",
        explanation_level="sentence",
        top_k=10,
        score_funcs=("stc", "loo", "denoised_shapley"),
        shapley_samples=3,
        loo_weight=2,
        beta=0.2,
        verbose=0,
    )

    scores = wrapper.score_memory(query=query, response=autogen_log, recs=memory_records)
    for score, rec, idx in scores:
        print(f"Memory idx={idx} score={score:.4f} | question='{rec['question']}'")


if __name__ == "__main__":
    main()
