from lark import Lark, UnexpectedInput, UnexpectedToken, UnexpectedCharacters, UnexpectedEOF
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from grammar import GEOMETRIC_ALGORITHM_GRAMMAR, GEOMETRIC_ALGORITHM_GRAMMAR_WITH_GBNF

def is_ebnf_compatible(text, ebnf_grammar, start_rule=None, parser_type='earley'):
    """
    Verifies if the given string 'text' is compatible with the provided EBNF grammar.
    Args:
        text (str): The string to verify.
        ebnf_grammar (str): The EBNF grammar as a string.
        start_rule (str, optional): The start rule for the grammar. If None, uses the first rule.
        parser_type (str): 'earley' (default, robust) or 'lalr' (faster if grammar is LALR(1)).
    Returns:
        tuple: (is_success: bool, parse_tree_or_None, error_details_or_None)
    """
    try:
        parser = Lark(ebnf_grammar, parser=parser_type, start=start_rule) if start_rule else Lark(ebnf_grammar, parser=parser_type)
        parse_tree = parser.parse(text)
        return True, parse_tree, None
    except UnexpectedInput as e:
        error_details = {
            "message": str(e),
            "line": getattr(e, 'line', None),
            "column": getattr(e, 'column', None),
            "context": e.get_context(text) if hasattr(e, 'get_context') else None,
            "type": type(e).__name__
        }
        if isinstance(e, UnexpectedToken):
            error_details["expected_tokens"] = list(e.expected) if hasattr(e, 'expected') and e.expected else None
            error_details["unexpected_token_type"] = e.token.type if hasattr(e, 'token') else None
            error_details["unexpected_token_value"] = e.token.value if hasattr(e, 'token') else None
        elif isinstance(e, UnexpectedCharacters):
            error_details["unexpected_sequence"] = e.seq if hasattr(e, 'seq') else None
        elif isinstance(e, UnexpectedEOF):
            error_details["expected_at_eof"] = list(e.expected) if hasattr(e, 'expected') and e.expected else None
        return False, None, error_details


guided_params = GuidedDecodingParams(grammar=GEOMETRIC_ALGORITHM_GRAMMAR_WITH_GBNF)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1000,
    guided_decoding=guided_params
)

prompt=f""" get some random but complex geometric algorithm that would meet criteria of gbnf grammar {GEOMETRIC_ALGORITHM_GRAMMAR_WITH_GBNF}"""

llm = LLM(model="Qwen/Qwen3-14B-FP8",tensor_parallel_size=2)
for i in range(5):
    outputs = llm.generate([prompt])#, sampling_params
    generated_text = outputs[0].outputs[0].text
    print("LLM output:", generated_text)
    iss=is_ebnf_compatible(generated_text, GEOMETRIC_ALGORITHM_GRAMMAR_WITH_GBNF)
    print(f"Is LLM output compatible with EBNF grammar? {iss[0]}")

#python3 is_EBNF_grammar_compatible.py