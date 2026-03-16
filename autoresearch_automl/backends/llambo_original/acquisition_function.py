"""LLM acquisition function — adapted from original LLAMBO paper code.

Generates candidate configurations by asking the LLM "what config would achieve
target performance X?" then filters and validates the suggestions.

Adapted from: tennisonliu/LLAMBO (acquisition_function.py)
Changes: synchronous OpenAI client, f-string prompts (no langchain), thinking trace capture.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LLM_ACQ:
    """LLM-based acquisition function for candidate generation.

    Prompts the LLM with observed (performance → config) examples and asks it
    to suggest configurations that could achieve a target performance level.
    """

    def __init__(
        self,
        task_context: dict,
        n_candidates: int,
        n_templates: int,
        lower_is_better: bool,
        chat_engine: str | None = None,
        prompt_setting: str | None = None,
        shuffle_features: bool = False,
        client: Any = None,
        max_tokens: int = 500,
        timeout: float = 600.0,
        llm_call_logger: Any = None,
    ):
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = int(n_candidates / n_templates)
        self.lower_is_better = lower_is_better
        self.chat_engine = chat_engine
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features
        self.client = client
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.llm_call_logger = llm_call_logger

    @staticmethod
    def _count_decimal_places(n: float) -> int:
        s = format(n, ".10f")
        if "." not in s:
            return 0
        return len(s.split(".")[1].rstrip("0"))

    def _prepare_configurations_acquisition(
        self,
        observed_configs: pd.DataFrame | None = None,
        observed_fvals: pd.DataFrame | float | None = None,
        seed: int | None = None,
        use_feature_semantics: bool = True,
        shuffle_features: bool = False,
    ) -> list[dict]:
        """Prepare few-shot examples for acquisition prompts.

        Format: performance → ## config ## (reversed from SM prompts).
        """
        examples = []

        if seed is not None and observed_configs is not None:
            rng = np.random.RandomState(seed)
            shuffled_indices = rng.permutation(observed_configs.index)
            observed_configs = observed_configs.loc[shuffled_indices]
            if isinstance(observed_fvals, pd.DataFrame):
                observed_fvals = observed_fvals.loc[shuffled_indices]
        elif observed_configs is not None and isinstance(observed_fvals, pd.DataFrame):
            # Sort by fvals: worst first, best last (recency bias)
            if self.lower_is_better:
                observed_fvals = observed_fvals.sort_values(
                    by=observed_fvals.columns[0], ascending=False
                )
            else:
                observed_fvals = observed_fvals.sort_values(
                    by=observed_fvals.columns[0], ascending=True
                )
            observed_configs = observed_configs.loc[observed_fvals.index]

        if shuffle_features and observed_configs is not None:
            rng = np.random.RandomState(0)
            shuffled_columns = rng.permutation(observed_configs.columns)
            observed_configs = observed_configs[shuffled_columns]

        if observed_configs is not None:
            hyperparameter_names = observed_configs.columns
            for index, row in observed_configs.iterrows():
                row_string = "## "
                for i in range(len(row)):
                    hp_name = hyperparameter_names[i]
                    constraint = self.task_context["hyperparameter_constraints"][hp_name]
                    hyp_type = constraint[0]

                    if use_feature_semantics:
                        row_string += f"{hp_name}: "
                    else:
                        row_string += f"X{i + 1}: "

                    if hyp_type in ["int", "float"]:
                        lower_bound = constraint[2][0]
                    else:
                        lower_bound = (
                            constraint[2][1]
                            if len(constraint[2]) > 1
                            else constraint[2][0]
                        )
                    n_dp = self._count_decimal_places(lower_bound)
                    value = row.iloc[i]

                    if hyp_type == "int":
                        row_string += str(int(value))
                    elif hyp_type == "float":
                        n_dp = max(n_dp, 2)  # ensure at least 2 decimal places
                        row_string += f"{value:.{n_dp}f}"
                    elif hyp_type == "ordinal":
                        row_string += f"{int(value)}"
                    else:
                        row_string += str(value)

                    if i != len(row) - 1:
                        row_string += ", "
                row_string += " ##"

                example = {"Q": row_string}
                if isinstance(observed_fvals, pd.DataFrame):
                    row_index = observed_fvals.index.get_loc(index)
                    perf = f"{observed_fvals.values[row_index][0]:.6f}"
                    example["A"] = perf
                examples.append(example)
        elif observed_fvals is not None:
            examples = [{"A": f"{observed_fvals:.6f}"}]
        else:
            raise ValueError("Need either observed_configs or observed_fvals")

        return examples

    def _gen_prompt_templates_acquisitions(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        desired_fval: float,
        n_prompts: int = 1,
        use_context: str = "full_context",
        use_feature_semantics: bool = True,
        shuffle_features: bool = False,
    ) -> tuple[list[str], list[list[dict]]]:
        """Generate prompt templates for acquisition function.

        Returns (prompt_templates, query_templates) using f-strings instead of langchain.
        """
        all_prompt_templates = []
        all_query_templates = []

        for i in range(n_prompts):
            few_shot_examples = self._prepare_configurations_acquisition(
                observed_configs,
                observed_fvals,
                seed=i,
                use_feature_semantics=use_feature_semantics,
            )

            # Task context
            tc = self.task_context
            model = tc["model"]
            task = tc["task"]
            tot_feats = tc["tot_feats"]
            cat_feats = tc["cat_feats"]
            num_feats = tc["num_feats"]
            n_classes = tc["n_classes"]
            metric = tc["metric"]
            if metric == "neg_mean_squared_error":
                metric = "mean squared error"
            num_samples = tc["num_samples"]
            hyperparameter_constraints = tc["hyperparameter_constraints"]
            custom_desc = tc.get("custom_task_description", "")

            # Build prefix
            prefix = ""
            if custom_desc:
                prefix = custom_desc.strip() + "\n\n"

            prefix += (
                f"The following are examples of performance of a {model} "
                f"measured in {metric} and the corresponding model "
                f"hyperparameter configurations."
            )
            if use_context == "full_context" and not custom_desc:
                if task == "classification":
                    prefix += (
                        f" The model is evaluated on a tabular {task} task "
                        f"containing {n_classes} classes."
                    )
                elif task == "regression":
                    prefix += f" The model is evaluated on a tabular {task} task."
                prefix += (
                    f" The tabular dataset contains {num_samples} samples and "
                    f"{tot_feats} features ({cat_feats} categorical, {num_feats} numerical)."
                )

            prefix += " The allowable ranges for the hyperparameters are:\n"

            for j, (hp_name, constraint) in enumerate(
                hyperparameter_constraints.items()
            ):
                hp_type, transform, search_range = constraint

                if hp_type == "float":
                    n_dp = max(self._count_decimal_places(search_range[0]), 2)
                    lower_bound = search_range[0]
                    upper_bound = search_range[1]
                    if use_feature_semantics:
                        prefix += f"- {hp_name}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                    else:
                        prefix += f"- X{j + 1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                    prefix += f" (float, precise to {n_dp} decimals)"

                elif hp_type == "int":
                    lower_bound = search_range[0]
                    upper_bound = search_range[1]
                    if use_feature_semantics:
                        prefix += f"- {hp_name}: [{lower_bound}, {upper_bound}]"
                    else:
                        prefix += f"- X{j + 1}: [{lower_bound}, {upper_bound}]"
                    prefix += " (int)"

                elif hp_type == "ordinal":
                    if use_feature_semantics:
                        prefix += f"- {hp_name}: "
                    else:
                        prefix += f"- X{j + 1}: "
                    prefix += f" (ordinal, must take value in {search_range})"

                else:
                    raise ValueError(f"Unknown hyperparameter type: {hp_type}")

                prefix += "\n"

            prefix += (
                f"Recommend a configuration that can achieve the target "
                f"performance of {desired_fval:.6f}. "
            )
            if use_context in ["partial_context", "full_context"]:
                prefix += (
                    "Do not recommend values at the minimum or maximum of "
                    "allowable range, do not recommend rounded values. "
                    "Recommend values with highest possible precision, "
                    "as requested by the allowed ranges. "
                )
            prefix += (
                "Your response must only contain the predicted configuration, "
                "in the format ## configuration ##.\n"
            )

            # Build few-shot section
            few_shot_str = ""
            for ex in few_shot_examples:
                few_shot_str += (
                    f"\nPerformance: {ex['A']}"
                    f"\nHyperparameter configuration: {ex['Q']}"
                )

            # Suffix with query
            query_examples = self._prepare_configurations_acquisition(
                observed_fvals=desired_fval, seed=None, shuffle_features=shuffle_features
            )

            template = (
                prefix
                + few_shot_str
                + f"\nPerformance: {query_examples[0]['A']}"
                + "\nHyperparameter configuration:"
            )
            all_prompt_templates.append(template)
            all_query_templates.append(query_examples)

        return all_prompt_templates, all_query_templates

    def _llm_call(self, user_message: str, n: int = 1) -> list[str] | None:
        """Make a synchronous LLM call. Returns list of response texts or None."""
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
                    temperature=0.8,
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
                        component="acquisition",
                        messages=messages,
                        responses=texts,
                        thinking=thinking_traces,
                        elapsed=elapsed,
                    )

                return texts

            except Exception as e:
                logger.warning(
                    "[ACQ] LLM call retry %d/%d: %s", retry + 1, MAX_RETRIES, e
                )
                if retry == MAX_RETRIES - 1:
                    logger.error(
                        "[ACQ] LLM call failed after %d retries", MAX_RETRIES
                    )
                    return None
                time.sleep(2 ** retry)

        return None

    @staticmethod
    def _convert_to_json(response_str: str) -> dict[str, float]:
        """Parse LLM response '## key: value, key: value ##' into dict."""
        pairs = response_str.split(",")
        result = {}
        for pair in pairs:
            parts = [x.strip() for x in pair.split(":")]
            if len(parts) == 2:
                key, value = parts
                try:
                    result[key] = float(value)
                except ValueError:
                    logger.debug("Cannot parse value for key %s: %s", key, value)
                    continue
        return result

    def _filter_candidate_points(
        self,
        observed_points: list[dict],
        candidate_points: list[dict],
        precision: int = 8,
    ) -> pd.DataFrame:
        """Filter candidates: remove duplicates, out-of-bounds, already observed."""
        rounded_observed = [
            {k: round(v, precision) for k, v in d.items()} for d in observed_points
        ]
        rounded_candidate = [
            {k: round(v, precision) for k, v in d.items()} for d in candidate_points
        ]
        filtered = [
            x
            for i, x in enumerate(candidate_points)
            if rounded_candidate[i] not in rounded_observed
        ]

        hyperparameter_constraints = self.task_context["hyperparameter_constraints"]

        def is_within_range(value: float, constraint: tuple) -> bool:
            value_type, transform, search_range = constraint
            if value_type == "int":
                return (
                    search_range[0] <= value <= search_range[1]
                    and int(value) == value
                )
            elif value_type == "float":
                return search_range[0] <= value <= search_range[1]
            elif value_type == "ordinal":
                return any(
                    math.isclose(value, x, abs_tol=1e-2) for x in search_range
                )
            else:
                raise ValueError(f"Unknown type: {value_type}")

        expected_keys = set(hyperparameter_constraints.keys())
        filtered = [
            d
            for d in filtered
            if set(d.keys()) == expected_keys
            and all(
                is_within_range(v, hyperparameter_constraints[k])
                for k, v in d.items()
            )
        ]

        if not filtered:
            return pd.DataFrame()

        df = pd.DataFrame(filtered)
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    def get_candidate_points(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        use_feature_semantics: bool = True,
        use_context: str = "full_context",
        alpha: float = -0.2,
    ) -> tuple[pd.DataFrame, float]:
        """Generate candidate points for the acquisition function.

        Returns (candidate_configs DataFrame, time_taken).
        """
        assert -1 <= alpha <= 1, "alpha must be between -1 and 1"
        if alpha == 0:
            alpha = -1e-3

        if self.prompt_setting is not None:
            use_context = self.prompt_setting

        start_time = time.time()

        # Compute desired target performance
        fval_range = abs(
            np.max(observed_fvals.values) - np.min(observed_fvals.values)
        )
        if fval_range == 0:
            fval_range = max(0.1 * abs(np.max(observed_fvals.values)), 1e-6)

        if self.lower_is_better:
            observed_best = np.min(observed_fvals.values)
            desired_fval = observed_best - alpha * fval_range
        else:
            observed_best = np.max(observed_fvals.values)
            desired_fval = observed_best + alpha * fval_range

        logger.info(
            "[ACQ] alpha=%.3f, best=%.6f, range=%.6f, desired=%.6f",
            alpha,
            observed_best,
            fval_range,
            desired_fval,
        )

        prompt_templates, query_templates = self._gen_prompt_templates_acquisitions(
            observed_configs,
            observed_fvals,
            desired_fval,
            n_prompts=self.n_templates,
            use_context=use_context,
            use_feature_semantics=use_feature_semantics,
            shuffle_features=self.shuffle_features,
        )

        logger.info(
            "[ACQ] %d templates, n_gens=%d per template",
            len(prompt_templates),
            self.n_gens,
        )

        filtered_candidate_points = pd.DataFrame()
        retry = 0

        while filtered_candidate_points.shape[0] < 5:
            candidate_points = []

            for prompt in prompt_templates:
                texts = self._llm_call(prompt, n=self.n_gens)
                if texts is None:
                    continue

                for text in texts:
                    try:
                        content = text.split("##")[1].strip()
                        candidate_points.append(self._convert_to_json(content))
                    except (IndexError, ValueError) as e:
                        logger.debug("[ACQ] Parse failed: %s | text: %s", e, text[:200])
                        continue

            proposed = self._filter_candidate_points(
                observed_configs.to_dict(orient="records"), candidate_points
            )
            filtered_candidate_points = pd.concat(
                [filtered_candidate_points, proposed], ignore_index=True
            )

            logger.info(
                "[ACQ] Attempt %d: %d proposed, %d accepted (total: %d)",
                retry,
                len(candidate_points),
                proposed.shape[0] if not proposed.empty else 0,
                filtered_candidate_points.shape[0],
            )

            retry += 1
            if retry > 3:
                if len(candidate_points) > 5:
                    # Use unfiltered candidates as fallback
                    filtered_candidate_points = pd.DataFrame(candidate_points)
                    logger.warning(
                        "[ACQ] Using %d unfiltered candidates after %d retries",
                        len(candidate_points),
                        retry,
                    )
                    break
                else:
                    logger.error(
                        "[ACQ] Failed to generate candidates after %d retries", retry
                    )
                    raise RuntimeError("LLM failed to generate candidate points")

        # Deduplicate across retry attempts
        if not filtered_candidate_points.empty:
            filtered_candidate_points = filtered_candidate_points.drop_duplicates().reset_index(drop=True)

        time_taken = time.time() - start_time
        return filtered_candidate_points, time_taken
