"""Local HuggingFace model as a deepeval judge.

Avoids OpenAI API costs for prototype evaluation. Scores will be noisier
than GPT-4o but provide directional signal for iterating on the agent.

Usage:
    from clinical_trial_agent.eval.local_judge import LocalJudge
    judge = LocalJudge()
    metric = SomeMetric(model=judge)
"""

from __future__ import annotations

from deepeval.models.base_model import DeepEvalBaseLLM

DEFAULT_JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 2048


class LocalJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL):
        self._model_name = model_name
        self._pipeline = None

    def load_model(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self._model_name,
                device_map="auto",
                torch_dtype="auto",
            )
        return self._pipeline

    def generate(self, prompt: str, **kwargs) -> str:
        pipe = self.load_model()
        messages = [{"role": "user", "content": prompt}]
        outputs = pipe(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_full_text=False,
        )
        return outputs[0]["generated_text"]

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def get_model_name(self) -> str:
        return self._model_name
