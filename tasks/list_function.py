import logging

from prompts.list_function import (
    example_prompt,
    feedback_prompt,
    io_prompt,
    io_prompt_with_format,
    noisy_rule_prompt,
    nosiy_rule_with_feedback_prompt,
    python_rule_prompt,
    rule_prompt,
    rule_to_output_prompt,
    rule_to_output_prompt_with_format,
    rule_to_python_prompt,
    rule_with_feedback_prompt,
)
from tasks.base import PythonTask
from utils.format_utils import str_to_list
from utils.query_utils import CLAUDE_MODELS

logger = logging.getLogger(__name__)


class ListFunction(PythonTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_name in CLAUDE_MODELS:
            self.io_prompt = io_prompt_with_format
            self.rule_to_output_prompt = rule_to_output_prompt_with_format
        else:
            self.io_prompt = io_prompt
            self.rule_to_output_prompt = rule_to_output_prompt

        if self.rule_type == "python":
            self.rule_prompt = python_rule_prompt
            self.rule_with_feedback_prompt = rule_with_feedback_prompt
        elif self.rule_type == "noisy":
            self.rule_prompt = noisy_rule_prompt
            self.rule_with_feedback_prompt = nosiy_rule_with_feedback_prompt
        else:
            self.rule_prompt = rule_prompt
            self.rule_with_feedback_prompt = rule_with_feedback_prompt
        self.example_prompt = example_prompt
        self.feedback_prompt = feedback_prompt
        self.rule_to_python_prompt = rule_to_python_prompt

    def format_input(self, input):
        return str_to_list(input)

    def format_output(self, output):
        if isinstance(output, list):
            try:
                return [int(c) for c in output]
            except:
                return output
        if isinstance(output, str):
            try:
                return [int(c) for c in str_to_list(output)]
            except:
                return output
        return output

    def extract_prediction(self, x):
        x = super().extract_prediction(x)
        return str_to_list(x)
