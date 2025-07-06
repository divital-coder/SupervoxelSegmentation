import vllm 

gbnf_grammar="""root ::= statement
statement ::= assign-var ws "=" ws expression ws ";" ws
expression ::= term (ws add-op ws term)*
term ::= factor (ws mul-op ws factor)*
factor ::= power (ws pow-op ws factor)?
power ::= func-call | variable | number | "(" ws expression ws ")"
func-call ::= func-name ws "(" ws arg-list ws ")"
arg-list ::= expression (ws "," ws expression)*
assign-var ::= "a"
variable ::= "b" | "c" | "d" | "e"
func-name ::= "sin" | "cos"
number ::= [0-9]+ ("." [0-9]+)?
add-op ::= "+" | "-"
mul-op ::= "*" | "/"
pow-op ::= "^"
ws ::= ([ \t]+)?"""

prompt = """You are a mathematical expression generator.
Your task is to generate a single assignment statement according to a strict GBNF grammar.
The statement must assign to the variable 'a'.
The expression on the right-hand side must be the sine of the arithmetic mean (average) of the variables 'b', 'c', 'd', and 'e'.
Use only the variables 'a', 'b', 'c', 'd', 'e'.
Use the function 'sin'. For division, use '/'.
The statement must end with a semicolon.

Allowed variables: a, b, c, d, e
Allowed functions: sin, cos
Allowed operators: +, -, *, ^

Output format: variable = expression;

Generate the assignment statement:"""

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


guided_params = GuidedDecodingParams(grammar=gbnf_grammar)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1000,
    guided_decoding=guided_params
)


llm = LLM(model="Qwen/Qwen2-7B-Instruct")
outputs = llm.generate([prompt], sampling_params)
generated_text = outputs[0].outputs[0].text
print("LLM output:", generated_text)

import math

# Example output from LLM
generated_text = "a = sin((b+c+d+e)/4);"

# Preprocess for Python
processed = generated_text.replace("^", "**")
processed = processed.replace("sin(", "math.sin(")
processed = processed.replace("cos(", "math.cos(")
if processed.strip().endswith(";"):
    processed = processed.strip()[:-1]

print("Python code:", processed)
# Define test values
b, c, d, e = 1.0, 2.0, 3.0, 4.0
a = None
local_vars = {'b': b, 'c': c, 'd': d, 'e': e, 'math': math}

exec(processed, {}, local_vars)
result = local_vars['a']

# Manual calculation
expected = math.sin((b + c + d + e) / 4)

print("LLM result:", result)
print("Expected:", expected)
print("Match:", abs(result - expected) < 1e-9)