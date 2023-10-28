example_prompt = """Input: {input}
Output: {output}"""


io_prompt = """Generate an output corresponding to the given input. Each output is generated by applying the same function to the respective inputs.

{examples}
Input: {test_input}
Output:"""


io_prompt_with_format = """Generate an output corresponding to the given input. Each output is generated by applying the same function to the respective inputs.

{examples}

Your input:
Input: {test_input}

Please format your output as follows:

Output: <Your output>"""


rule_prompt = """Generate a rule that maps the following inputs to their corresponding outputs.

{examples}

Please format your rule as follows:

Rule: <Your rule>"""


noisy_rule_prompt = """Generate a rule that maps the following inputs to their corresponding outputs. Note that some examples may be noisy, and you should take this into account when proposing the rule.

{examples}

Please format your rule as follows:

Rule: <Your rule>"""


python_rule_prompt = """Generate a Python function `fn` that maps the following inputs to their corresponding outputs.

{examples}

Please format your Python function as follows:

```python
def fn(x):
    # Your code here
```"""


rule_to_python_prompt = """You are an expert Python programmer. Write a Python function `fn` for the following rule. The input is a list of integers. The output is also a list of integers.

Rule: {rule}"""


feedback_prompt = """Input: {input}
Expected output: {target}
Actual output: {output}"""


rule_with_feedback_prompt = """Your rule: {rule}

This rule does not work for the following examples.

{feedback}

Generate a new rule that maps the given inputs to their corresponding outputs. Please format your rule as follows:

Rule: <Your rule>"""


nosiy_rule_with_feedback_prompt = """Your rule: {rule}

This rule does not work for the following examples.

{feedback}

Generate a new rule that maps the given inputs to their corresponding outputs. Note that some examples may be noisy, and you should take this into account when proposing the rule. Please format your rule as follows:

Rule: <Your rule>"""


rule_to_output_prompt = """Generate an output corresponding to the given input based on the rule. The input is a list of integers. The output is also a list of integers.

Rule: {rule}

Input: {test_input}
Output:"""


rule_to_output_prompt_with_format = """Generate an output corresponding to the given input based on the rule. The input is a list of integers. The output is also a list of integers.

Rule: {rule}

Your input:
Input: {test_input}

Please format your output as follows:

Output: <Your output>"""