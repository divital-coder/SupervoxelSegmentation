from outlines import models
from outlines import models, generate
import outlines
import os
arithmetic_grammar = outlines.grammars.arithmetic
os.environ["VLLM_USE_V1"] = "0"

model = models.vllm(
    "microsoft/Phi-3-mini-4k-instruct"
    # tensor_parallel_size=2
)

generator = generate.cfg(model, arithmetic_grammar)
sequence = generator(
  "Alice had 4 apples and Bob ate 2. "
  + "Write an expression for Alice's apples:"
)

print(sequence)


