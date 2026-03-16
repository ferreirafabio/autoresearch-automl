"""Discriminative surrogate model — adapted from original LLAMBO paper code.

Models p(y|x) using LLM few-shot prompting with actual metric values.
Key advantage over OptunaHub port: uses real performance values (e.g., ## 0.970 ##)
instead of binary 0/1 classification.

Adapted from: tennisonliu/LLAMBO (discriminative_sm.py + discriminative_sm_utils.py)
Changes: synchronous OpenAI client, f-string prompts (no langchain), thinking trace capture.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def _count_decimal_places(n: float) -> int:
    """Count the number of decimal places in a number."""
    s = format(n, ".10f")
    if "." not in s:
        return 0
    return len(s.split(".")[1].rstrip("0"))


def prepare_configurations(
    hyperparameter_constraints: dict,
    observed_configs: pd.DataFrame,
    observed_fvals: pd.DataFrame | None = None,
    seed: int | None = None,
    use_feature_semantics: bool = True,
    shuffle_features: bool = False,
) -> list[dict]:
    """Serialize configurations into natural language for SM few-shot prompts.

    Each config becomes e.g. "DEPTH is 8, ASPECT_RATIO is 64, HEAD_DIM is 128"
    with optional performance label "## 0.970000 ##".
    """
    examples = []
    observed_configs = observed_configs.copy()
    hyperparameter_names = observed_configs.columns

    # Shuffle rows to reduce permutation sensitivity
    if seed is not None:
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(observed_configs.index)
        observed_configs = observed_configs.loc[shuffled_indices]
        if observed_fvals is not None:
            observed_fvals = observed_fvals.loc[shuffled_indices]

    # Shuffle columns
    if shuffle_features:
        rng = np.random.RandomState(0)
        shuffled_cols = rng.permutation(len(hyperparameter_names))
        observed_configs = observed_configs[hyperparameter_names[shuffled_cols]]

    observed_configs = observed_configs.reset_index(drop=True)
    if observed_fvals is not None:
        observed_fvals = observed_fvals.reset_index(drop=True)

    for index, row in observed_configs.iterrows():
        row_string = ""
        cols = observed_configs.columns
        for i in range(len(row)):
            hp_name = cols[i]
            constraint = hyperparameter_constraints[hp_name]
            hyp_type = constraint[0]

            if hyp_type in ["int", "float"]:
                lower_bound = constraint[2][0]
            else:
                lower_bound = constraint[2][1] if len(constraint[2]) > 1 else constraint[2][0]
            n_dp = _count_decimal_places(lower_bound)

            if use_feature_semantics:
                row_string += f"{hp_name} is "
            else:
                row_string += f"X{i + 1} is "

            if hyp_type == "int":
                row_string += str(int(row.iloc[i]))
            elif hyp_type == "float":
                n_dp = max(n_dp, 2)  # ensure at least 2 decimal places for floats
                row_string += f"{row.iloc[i]:.{n_dp}f}"
            elif hyp_type == "ordinal":
                row_string += f"{int(row.iloc[i])}"
            else:
                row_string += str(row.iloc[i])

            if i != len(row) - 1:
                row_string += ", "

        example = {"Q": row_string}
        if observed_fvals is not None:
            row_index = observed_fvals.index.get_loc(index)
            perf = f"## {observed_fvals.values[row_index][0]:.6f} ##"
            example["A"] = perf
        examples.append(example)

    return examples


def gen_prompt_templates(
    task_context: dict,
    observed_configs: pd.DataFrame,
    observed_fvals: pd.DataFrame,
    candidate_configs: pd.DataFrame,
    n_prompts: int = 1,
    use_context: str = "full_context",
    use_feature_semantics: bool = True,
    shuffle_features: bool = False,
) -> tuple[list[str], list[dict]]:
    """Generate few-shot prompt templates for discriminative SM.

    Returns (prompt_templates, query_examples) where each template is an
    f-string with {Q} placeholder for the query configuration.
    """
    model = task_context["model"]
    task = task_context["task"]
    tot_feats = task_context["tot_feats"]
    cat_feats = task_context["cat_feats"]
    num_feats = task_context["num_feats"]
    n_classes = task_context["n_classes"]
    n_samples = task_context["num_samples"]
    metric = task_context["metric"]
    custom_desc = task_context.get("custom_task_description", "")

    if metric == "neg_mean_squared_error":
        metric = "mean squared error"
    if use_context == "no_context" or not use_feature_semantics:
        metric = "a metric"

    all_prompt_templates = []
    for i in range(n_prompts):
        few_shot_examples = prepare_configurations(
            task_context["hyperparameter_constraints"],
            observed_configs,
            observed_fvals,
            seed=i,
            use_feature_semantics=use_feature_semantics,
            shuffle_features=shuffle_features,
        )

        # Build prefix
        prefix = ""
        if custom_desc:
            prefix = custom_desc.strip() + "\n\n"

        prefix += (
            f"The following are hyperparameter configurations for a {model} "
            f"and the corresponding performance measured in {metric}."
        )
        if use_context == "full_context" and not custom_desc:
            # Only emit tabular-dataset context when no custom description
            # (custom desc already provides domain context)
            if task == "classification":
                prefix += (
                    f" The model is evaluated on a tabular {task} task "
                    f"and the label contains {n_classes} classes."
                )
            elif task == "regression":
                prefix += f" The model is evaluated on a tabular {task} task."
            prefix += (
                f" The tabular dataset contains {n_samples} samples and "
                f"{tot_feats} features ({cat_feats} categorical, {num_feats} numerical)."
            )
        prefix += (
            f" Your response should only contain the predicted {metric} "
            f"in the format ## performance ##."
        )

        # Build few-shot section
        few_shot_str = ""
        for ex in few_shot_examples:
            few_shot_str += (
                f"\nHyperparameter configuration: {ex['Q']}"
                f"\nPerformance: {ex['A']}"
            )

        # Template with {Q} placeholder for query
        template = (
            prefix
            + few_shot_str
            + "\nHyperparameter configuration: {Q}\nPerformance: "
        )
        all_prompt_templates.append(template)

    # Query examples (candidate configs serialized without labels)
    query_examples = prepare_configurations(
        task_context["hyperparameter_constraints"],
        candidate_configs,
        seed=None,
        use_feature_semantics=use_feature_semantics,
        shuffle_features=shuffle_features,
    )

    return all_prompt_templates, query_examples


class LLM_DIS_SM:
    """Discriminative surrogate model — models p(y|x) with actual metric values.

    For each candidate configuration, prompts the LLM with few-shot examples
    showing (config → performance) and asks it to predict the performance.
    Uses Expected Improvement to select the best candidate.
    """

    def __init__(
        self,
        task_context: dict,
        n_gens: int,
        lower_is_better: bool,
        n_templates: int = 1,
        chat_engine: str | None = None,
        prompt_setting: str | None = None,
        shuffle_features: bool = False,
        client: Any = None,
        max_tokens: int = 500,
        timeout: float = 600.0,
        llm_call_logger: Any = None,
    ):
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.n_templates = n_templates
        self.chat_engine = chat_engine
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features
        self.client = client
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.llm_call_logger = llm_call_logger

    def _llm_call(
        self, user_message: str, n: int = 1
    ) -> tuple[list[str], list[str]] | None:
        """Make a synchronous LLM call. Returns (texts, thinking_traces) or None."""
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            },
            {"role": "user", "content": user_message},
        ]

        MAX_RETRIES = 3
        for retry in range(MAX_RETRIES):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.chat_engine,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=self.max_tokens,
                    top_p=0.95,
                    n=n,
                    timeout=self.timeout,
                )
                elapsed = time.time() - t0

                texts = []
                thinking_traces = []
                for choice in resp.choices:
                    msg = choice.message
                    content = (msg.content or "").strip()
                    extra = (getattr(msg, "model_extra", {}) or {})
                    thinking = extra.get("reasoning") or getattr(
                        msg, "reasoning_content", None
                    )
                    if not content and thinking:
                        content = thinking.strip()
                        thinking = None
                    texts.append(content)
                    if thinking:
                        thinking_traces.append(thinking)

                if self.llm_call_logger:
                    self.llm_call_logger.log_call(
                        component="surrogate",
                        messages=messages,
                        responses=texts,
                        thinking=thinking_traces,
                        elapsed=elapsed,
                    )

                return texts, thinking_traces

            except Exception as e:
                logger.warning(
                    "[SM] LLM call retry %d/%d: %s", retry + 1, MAX_RETRIES, e
                )
                if retry == MAX_RETRIES - 1:
                    logger.error("[SM] LLM call failed after %d retries", MAX_RETRIES)
                    return None
                time.sleep(2 ** retry)

        return None

    def _predict(
        self, all_prompt_templates: list[str], query_examples: list[dict]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Predict performance for candidate configs.

        Returns (y_mean, y_std, time_taken).
        """
        start = time.time()
        all_preds = []

        # Process candidates in chunks of 5 (like original)
        for i in range(0, len(query_examples), 5):
            chunk = query_examples[i : i + 5]

            for query_example in chunk:
                sample_preds = []
                for template in all_prompt_templates:
                    prompt = template.format(Q=query_example["Q"])
                    n_per_template = self.n_gens

                    result = self._llm_call(prompt, n=n_per_template)
                    if result is None:
                        sample_preds.extend([np.nan] * n_per_template)
                        continue

                    texts, _ = result
                    for text in texts:
                        gen_pred = re.findall(r"## (-?[\d.]+) ##", text)
                        if len(gen_pred) == 1:
                            sample_preds.append(float(gen_pred[0]))
                        else:
                            sample_preds.append(np.nan)

                while len(sample_preds) < self.n_gens:
                    sample_preds.append(np.nan)
                all_preds.append(sample_preds)

        elapsed = time.time() - start

        all_preds = np.array(all_preds).astype(float)
        y_mean = np.nanmean(all_preds, axis=1)
        y_std = np.nanstd(all_preds, axis=1)

        # Handle all-NaN case: fall back to uniform predictions (degrades to random)
        if np.all(np.isnan(y_mean)):
            logger.error("[SM] All predictions are NaN — LLM returned no parseable results")
            y_mean = np.zeros(len(query_examples))
            y_std = np.ones(len(query_examples))
        else:
            # Impute NaN with average predictions
            y_mean[np.isnan(y_mean)] = np.nanmean(y_mean)
            y_std[np.isnan(y_std)] = np.nanmean(y_std)
        y_std[y_std < 1e-5] = 1e-5  # avoid division by zero

        return y_mean, y_std, elapsed

    def select_query_point(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        candidate_configs: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float]:
        """Select the best candidate using Expected Improvement."""
        use_context = self.prompt_setting or "full_context"

        all_prompt_templates, query_examples = gen_prompt_templates(
            self.task_context,
            observed_configs,
            observed_fvals,
            candidate_configs,
            n_prompts=self.n_templates,
            use_context=use_context,
            shuffle_features=self.shuffle_features,
        )

        logger.info(
            "[SM] %d templates, %d candidates",
            len(all_prompt_templates),
            len(query_examples),
        )

        y_mean, y_std, time_taken = self._predict(
            all_prompt_templates, query_examples
        )

        # Compute Expected Improvement
        if self.lower_is_better:
            best_fval = np.min(observed_fvals.to_numpy())
            delta = -1 * (y_mean - best_fval)
        else:
            best_fval = np.max(observed_fvals.to_numpy())
            delta = y_mean - best_fval

        with np.errstate(divide="ignore"):
            Z = delta / y_std

        ei = np.where(y_std > 0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)

        best_point_index = np.argmax(ei)
        best_point = candidate_configs.iloc[[best_point_index], :]

        logger.info(
            "[SM] Best EI=%.6f at index %d (mean=%.4f, std=%.4f)",
            ei[best_point_index],
            best_point_index,
            y_mean[best_point_index],
            y_std[best_point_index],
        )

        return best_point, time_taken
