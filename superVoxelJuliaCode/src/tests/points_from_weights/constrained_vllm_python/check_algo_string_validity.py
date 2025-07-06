import re
from algo_validation_utils import check_tags_presence, check_s5_point_usage, check_statement_termination, extract_step_blocks
from verify_is_string_ok_with_grammar import validate_algorithm_syntax


def validate_algorithm(algorithm_string: str) -> tuple[bool, str]:
    """
    Runs all validation checks on the provided algorithm string, including per-step grammar validation.

    Args:
        algorithm_string: The LLM-generated algorithm.

    Returns:
        (is_valid, correction_prompt)
    """
    # 1. Check tags presence
    tags_ok, tags_msg = check_tags_presence(algorithm_string)
    if not tags_ok:
        return False, f"Missing required tags: {tags_msg}\nAdd all <S1> through <S5> blocks, each with code inside."

    # 2. Extract step blocks
    step_blocks = extract_step_blocks(algorithm_string)
    all_steps_present = all(code is not None for code in step_blocks.values())
    all_steps_nonempty = all(code and any(l.strip() and not l.strip().startswith('#') for l in code.split('\n')) for code in step_blocks.values() if code is not None)
    correction_messages = []
    is_valid = True

    # 3. Per-step grammar validation (with step-specific checks)
    for step, code in step_blocks.items():
        if code is None:
            correction_messages.append(f"Step <{step}> is missing. Add this block with the required code as per the design pattern.")
            is_valid = False
            continue
        code_lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        if not code_lines:
            correction_messages.append(f"Step <{step}> is empty. Fill it with valid code following the grammar and design pattern.")
            is_valid = False
            continue
        # Step number as int (S1->1, S2->2, ...)
        step_num = int(step[1]) if step.startswith('S') and step[1:].isdigit() else 0
        grammar_msg = validate_algorithm_syntax(code, step=step_num)
        if grammar_msg:
            correction_messages.append(f"Grammar error in <{step}>: {grammar_msg}")
            is_valid = False

    # 4. S5 point usage check
    s5_code = step_blocks.get('S5')
    s5_ok, s5_msg = check_s5_point_usage(s5_code if s5_code is not None else '')
    if not s5_ok:
        is_valid = False
        correction_messages.append(f"<S5> block error: {s5_msg}\nEnsure all required points are used in triangle definitions as per the Negative Corner pattern.")

    # 5. Semicolon check
    semi_ok, semi_msg = check_statement_termination(algorithm_string)
    if not semi_ok:
        is_valid = False
        correction_messages.append(f"Semicolon error: {semi_msg}\nEvery statement inside <S1> to <S5> must end with a semicolon.")

    # 6. Compose correction prompt
    if is_valid:
        return True, "Algorithm is valid and follows all grammar and design rules."
    else:
        prompt = ("Your algorithm has the following issues:\n" +
                  '\n'.join(f"- {msg}" for msg in correction_messages) +
                  "\n\nTo fix:\n" +
                  "- Carefully review the EBNF grammar and the Negative Corner design pattern.\n" +
                  "- For each error above, rewrite the affected step(s) using only allowed syntax, variable naming, and functions.\n" +
                  "- Ensure all required points are defined and used, and every statement ends with a semicolon.\n" +
                  "- Do not use any prohibited constructs (conditionals, loops, arrays, etc).\n" +
                  "- Each <S*> block must be present and non-empty.\n" +
                  "- When correcting, keep comments for clarity, but ensure all code lines are valid.\n")
        return False, prompt

# --- Example Usage ---

# Example of a valid algorithm string for testing purposes
valid_algorithm = """
<S1>
        svCenter = originalSvCenter_(0,0,0) + vector(range(0.001, 0.002, w1) * rx - rx * 0.0005, range(0.001, 0.002, w2) * ry - ry * 0.0005, range(0.001, 0.002, w3) * rz - rz * 0.0005);
</S1>
<S2>
        linX = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), range(0.49, 0.51, w4));
        linY = lin_p(svCenter_(0,0,0), svCenter_(0,-1,0), range(0.49, 0.51, w5));
        linZ = lin_p(svCenter_(0,0,0), svCenter_(0,0,-1), range(0.49, 0.51, w6));
        temp1 = lin_p(svCenter_(-1,-1,0), svCenter_(0,-1,0), range(0.49, 0.51, w7));
        temp2 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.49, 0.51, w7));
        edgeXY = lin_p(temp1, temp2, range(0.49, 0.51, w8));
        temp3 = lin_p(svCenter_(0,-1,-1), svCenter_(0,0,-1), range(0.49, 0.51, w9));
        temp4 = lin_p(svCenter_(0,-1,0), svCenter_(0,0,0), range(0.49, 0.51, w9));
        edgeYZ = lin_p(temp3, temp4, range(0.49, 0.51, w10));
        temp5 = lin_p(svCenter_(-1,0,-1), svCenter_(0,0,-1), range(0.49, 0.51, w11));
        temp6 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.49, 0.51, w11));
        edgeXZ = lin_p(temp5, temp6, range(0.49, 0.51, w12));
</S2>
<S3>
        tempX00 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.49, 0.51, w13));
        tempX10 = lin_p(svCenter_(-1,-1,0), svCenter_(0,-1,0), range(0.49, 0.51, w13));
        tempX01 = lin_p(svCenter_(-1,0,-1), svCenter_(0,0,-1), range(0.49, 0.51, w13));
        tempX11 = lin_p(svCenter_(-1,-1,-1), svCenter_(0,-1,-1), range(0.49, 0.51, w13));
        tempY0 = lin_p(tempX10, tempX00, range(0.49, 0.51, w14));
        tempY1 = lin_p(tempX11, tempX01, range(0.49, 0.51, w14));
        mob = lin_p(tempY1, tempY0, range(0.49, 0.51, w15));</S3>
<S4>
        int1 = lin_p(mob_(0,0,0), mob_(1,0,0), 0.5);
        int2 = lin_p(int1, mob_(1,0,0), 0.5);
        int3 = lin_p(int2, mob_(1,0,0), 0.5);
        int4 = lin_p(int3, mob_(1,0,0), 0.5);

        int5 = lin_p(mob_(0,0,0), mob_(0,1,0), 0.5);
        int6 = lin_p(int5, mob_(0,1,0), 0.5);
        int7 = lin_p(int6, mob_(0,1,0), 0.5);
        int8 = lin_p(int7, mob_(0,1,0), 0.5);
        
        int9 = lin_p(mob_(0,0,0), mob_(0,0,1), 0.5);
        int10 = lin_p(int9, mob_(0,0,1), 0.5);
        int11 = lin_p(int10, mob_(0,0,1), 0.5);
        int12 = lin_p(int11, mob_(0,0,1), 0.5);
        
        
        </S4>
<S5>

# Triangles on the -Z face
# Original: defineTriangle(mob_(0,0,0), mob_(1,0,0), linZ_(0,0,0));
defineTriangle(mob_(0,0,0), int1_(0,0,0), linZ_(0,0,0));
defineTriangle(int1_(0,0,0), int2_(0,0,0), linZ_(0,0,0));
defineTriangle(int2_(0,0,0), int3_(0,0,0), linZ_(0,0,0));
defineTriangle(int3_(0,0,0), int4_(0,0,0), linZ_(0,0,0));
defineTriangle(int4_(0,0,0), mob_(1,0,0), linZ_(0,0,0));

# Original: defineTriangle(mob_(0,0,0), mob_(0,1,0), linZ_(0,0,0));
defineTriangle(mob_(0,0,0), int5_(0,0,0), linZ_(0,0,0));
defineTriangle(int5_(0,0,0), int6_(0,0,0), linZ_(0,0,0));
defineTriangle(int6_(0,0,0), int7_(0,0,0), linZ_(0,0,0));
defineTriangle(int7_(0,0,0), int8_(0,0,0), linZ_(0,0,0));
defineTriangle(int8_(0,0,0), mob_(0,1,0), linZ_(0,0,0));

# Triangles on the -Y face
# Original: defineTriangle(mob_(0,0,0), mob_(0,0,1), linY_(0,0,0));
defineTriangle(mob_(0,0,0), int9_(0,0,0), linY_(0,0,0));
defineTriangle(int9_(0,0,0), int10_(0,0,0), linY_(0,0,0));
defineTriangle(int10_(0,0,0), int11_(0,0,0), linY_(0,0,0));
defineTriangle(int11_(0,0,0), int12_(0,0,0), linY_(0,0,0));
defineTriangle(int12_(0,0,0), mob_(0,0,1), linY_(0,0,0));

# Original: defineTriangle(mob_(0,0,0), mob_(1,0,0), linY_(0,0,0));
defineTriangle(mob_(0,0,0), int1_(0,0,0), linY_(0,0,0));
defineTriangle(int1_(0,0,0), int2_(0,0,0), linY_(0,0,0));
defineTriangle(int2_(0,0,0), int3_(0,0,0), linY_(0,0,0));
defineTriangle(int3_(0,0,0), int4_(0,0,0), linY_(0,0,0));
defineTriangle(int4_(0,0,0), mob_(1,0,0), linY_(0,0,0));

# Triangles on the -X face
# Original: defineTriangle(mob_(0,0,0), mob_(0,1,0), linX_(0,0,0));
defineTriangle(mob_(0,0,0), int5_(0,0,0), linX_(0,0,0));
defineTriangle(int5_(0,0,0), int6_(0,0,0), linX_(0,0,0));
defineTriangle(int6_(0,0,0), int7_(0,0,0), linX_(0,0,0));
defineTriangle(int7_(0,0,0), int8_(0,0,0), linX_(0,0,0));
defineTriangle(int8_(0,0,0), mob_(0,1,0), linX_(0,0,0));

# Original: defineTriangle(mob_(0,0,0), mob_(0,0,1), linX_(0,0,0));
defineTriangle(mob_(0,0,0), int9_(0,0,0), linX_(0,0,0));
defineTriangle(int9_(0,0,0), int10_(0,0,0), linX_(0,0,0));
defineTriangle(int10_(0,0,0), int11_(0,0,0), linX_(0,0,0));
defineTriangle(int11_(0,0,0), int12_(0,0,0), linX_(0,0,0));
defineTriangle(int12_(0,0,0), mob_(0,0,1), linX_(0,0,0));


# --- Triangles on the shared face between (0,0,0) and (0,1,0) ---
# Edge mob_(1,1,0) -> mob_(1,0,0) is a -Y edge, negative corner is (1,0,0). Use int5-8_(1,0,0) in reverse.
# Original: defineTriangle(mob_(1,1,0), mob_(1,0,0), linZ_(0,0,0));
defineTriangle(mob_(1,1,0), int8_(1,0,0), linZ_(0,0,0));
defineTriangle(int8_(1,0,0), int7_(1,0,0), linZ_(0,0,0));
defineTriangle(int7_(1,0,0), int6_(1,0,0), linZ_(0,0,0));
defineTriangle(int6_(1,0,0), int5_(1,0,0), linZ_(0,0,0));
defineTriangle(int5_(1,0,0), mob_(1,0,0), linZ_(0,0,0));

# Edge mob_(1,1,0) -> mob_(0,1,0) is a -X edge, negative corner is (0,1,0). Use int1-4_(0,1,0) in reverse.
# Original: defineTriangle(mob_(1,1,0), mob_(0,1,0), linZ_(0,0,0));
defineTriangle(mob_(1,1,0), int4_(0,1,0), linZ_(0,0,0));
defineTriangle(int4_(0,1,0), int3_(0,1,0), linZ_(0,0,0));
defineTriangle(int3_(0,1,0), int2_(0,1,0), linZ_(0,0,0));
defineTriangle(int2_(0,1,0), int1_(0,1,0), linZ_(0,0,0));
defineTriangle(int1_(0,1,0), mob_(0,1,0), linZ_(0,0,0));

# Edge mob_(1,1,0) -> mob_(0,1,0) is a -X edge, negative corner is (0,1,0). Use int1-4_(0,1,0) in reverse.
# Original: defineTriangle(mob_(1,1,0), mob_(0,1,0), linY_(0,1,0));
defineTriangle(mob_(1,1,0), int4_(0,1,0), linY_(0,1,0));
defineTriangle(int4_(0,1,0), int3_(0,1,0), linY_(0,1,0));
defineTriangle(int3_(0,1,0), int2_(0,1,0), linY_(0,1,0));
defineTriangle(int2_(0,1,0), int1_(0,1,0), linY_(0,1,0));
defineTriangle(int1_(0,1,0), mob_(0,1,0), linY_(0,1,0));

# Edge mob_(1,1,0) -> mob_(1,1,1) is a +Z edge, negative corner is (1,1,0). Use int9-12_(1,1,0).
# Original: defineTriangle(mob_(1,1,0), mob_(1,1,1), linY_(0,1,0));
defineTriangle(mob_(1,1,0), int9_(1,1,0), linY_(0,1,0));
defineTriangle(int9_(1,1,0), int10_(1,1,0), linY_(0,1,0));
defineTriangle(int10_(1,1,0), int11_(1,1,0), linY_(0,1,0));
defineTriangle(int11_(1,1,0), int12_(1,1,0), linY_(0,1,0));
defineTriangle(int12_(1,1,0), mob_(1,1,1), linY_(0,1,0));

# Edge mob_(1,1,0) -> mob_(1,1,1) is a +Z edge, negative corner is (1,1,0). Use int9-12_(1,1,0).
# Original: defineTriangle(mob_(1,1,0), mob_(1,1,1), linX_(1,0,0));
defineTriangle(mob_(1,1,0), int9_(1,1,0), linX_(1,0,0));
defineTriangle(int9_(1,1,0), int10_(1,1,0), linX_(1,0,0));
defineTriangle(int10_(1,1,0), int11_(1,1,0), linX_(1,0,0));
defineTriangle(int11_(1,1,0), int12_(1,1,0), linX_(1,0,0));
defineTriangle(int12_(1,1,0), mob_(1,1,1), linX_(1,0,0));

# Edge mob_(1,1,0) -> mob_(1,0,0) is a -Y edge, negative corner is (1,0,0). Use int5-8_(1,0,0) in reverse.
# Original: defineTriangle(mob_(1,1,0), mob_(1,0,0), linX_(1,0,0));
defineTriangle(mob_(1,1,0), int8_(1,0,0), linX_(1,0,0));
defineTriangle(int8_(1,0,0), int7_(1,0,0), linX_(1,0,0));
defineTriangle(int7_(1,0,0), int6_(1,0,0), linX_(1,0,0));
defineTriangle(int6_(1,0,0), int5_(1,0,0), linX_(1,0,0));
defineTriangle(int5_(1,0,0), mob_(1,0,0), linX_(1,0,0));


# --- Triangles on the shared face between (0,0,0) and (0,0,1) ---
# Edge mob_(1,0,1) -> mob_(1,1,1) is a +Y edge, negative corner is (1,0,1). Use int5-8_(1,0,1).
# Original: defineTriangle(mob_(1,0,1), mob_(1,1,1), linZ_(0,0,1));
defineTriangle(mob_(1,0,1), int5_(1,0,1), linZ_(0,0,1));
defineTriangle(int5_(1,0,1), int6_(1,0,1), linZ_(0,0,1));
defineTriangle(int6_(1,0,1), int7_(1,0,1), linZ_(0,0,1));
defineTriangle(int7_(1,0,1), int8_(1,0,1), linZ_(0,0,1));
defineTriangle(int8_(1,0,1), mob_(1,1,1), linZ_(0,0,1));

# Edge mob_(1,0,1) -> mob_(0,0,1) is a -X edge, negative corner is (0,0,1). Use int1-4_(0,0,1) in reverse.
# Original: defineTriangle(mob_(1,0,1), mob_(0,0,1), linZ_(0,0,1));
defineTriangle(mob_(1,0,1), int4_(0,0,1), linZ_(0,0,1));
defineTriangle(int4_(0,0,1), int3_(0,0,1), linZ_(0,0,1));
defineTriangle(int3_(0,0,1), int2_(0,0,1), linZ_(0,0,1));
defineTriangle(int2_(0,0,1), int1_(0,0,1), linZ_(0,0,1));
defineTriangle(int1_(0,0,1), mob_(0,0,1), linZ_(0,0,1));

# Edge mob_(1,0,1) -> mob_(0,0,1) is a -X edge, negative corner is (0,0,1). Use int1-4_(0,0,1) in reverse.
# Original: defineTriangle(mob_(1,0,1), mob_(0,0,1), linY_(0,0,0));
defineTriangle(mob_(1,0,1), int4_(0,0,1), linY_(0,0,0));
defineTriangle(int4_(0,0,1), int3_(0,0,1), linY_(0,0,0));
defineTriangle(int3_(0,0,1), int2_(0,0,1), linY_(0,0,0));
defineTriangle(int2_(0,0,1), int1_(0,0,1), linY_(0,0,0));
defineTriangle(int1_(0,0,1), mob_(0,0,1), linY_(0,0,0));

# Edge mob_(1,0,1) -> mob_(1,0,0) is a -Z edge, negative corner is (1,0,0). Use int9-12_(1,0,0) in reverse.
# Original: defineTriangle(mob_(1,0,1), mob_(1,0,0), linY_(0,0,0));
defineTriangle(mob_(1,0,1), int12_(1,0,0), linY_(0,0,0));
defineTriangle(int12_(1,0,0), int11_(1,0,0), linY_(0,0,0));
defineTriangle(int11_(1,0,0), int10_(1,0,0), linY_(0,0,0));
defineTriangle(int10_(1,0,0), int9_(1,0,0), linY_(0,0,0));
defineTriangle(int9_(1,0,0), mob_(1,0,0), linY_(0,0,0));

# Edge mob_(1,0,1) -> mob_(1,1,1) is a +Y edge, negative corner is (1,0,1). Use int5-8_(1,0,1).
# Original: defineTriangle(mob_(1,0,1), mob_(1,1,1), linX_(1,0,0));
defineTriangle(mob_(1,0,1), int5_(1,0,1), linX_(1,0,0));
defineTriangle(int5_(1,0,1), int6_(1,0,1), linX_(1,0,0));
defineTriangle(int6_(1,0,1), int7_(1,0,1), linX_(1,0,0));
defineTriangle(int7_(1,0,1), int8_(1,0,1), linX_(1,0,0));
defineTriangle(int8_(1,0,1), mob_(1,1,1), linX_(1,0,0));

# Edge mob_(1,0,1) -> mob_(1,1,1) is a +Y edge, negative corner is (1,0,1). Use int5-8_(1,0,1).
# Original: defineTriangle(mob_(1,0,1), mob_(1,1,1), linX_(1,0,0));
defineTriangle(mob_(1,0,1), int9_(1,0,1), linX_(1,0,0));
defineTriangle(int9_(1,0,1), int10_(1,0,1), linX_(1,0,0));
defineTriangle(int10_(1,0,1), int11_(1,0,1), linX_(1,0,0));
defineTriangle(int11_(1,0,1), int12_(1,0,1), linX_(1,0,0));
defineTriangle(int12_(1,0,1), mob_(1,1,1), linX_(1,0,0));


# --- Triangles on the shared face between (0,0,0) and (1,0,0) ---
# Edge mob_(0,1,1) -> mob_(1,1,1) is a +X edge, negative corner is (0,1,1). Use int1-4_(0,1,1).
# Original: defineTriangle(mob_(0,1,1), mob_(1,1,1), linZ_(0,0,1));
defineTriangle(mob_(0,1,1), int1_(0,1,1), linZ_(0,0,1));
defineTriangle(int1_(0,1,1), int2_(0,1,1), linZ_(0,0,1));
defineTriangle(int2_(0,1,1), int3_(0,1,1), linZ_(0,0,1));
defineTriangle(int3_(0,1,1), int4_(0,1,1), linZ_(0,0,1));
defineTriangle(int4_(0,1,1), mob_(1,1,1), linZ_(0,0,1));

# Edge mob_(0,1,1) -> mob_(0,0,1) is a -Y edge, negative corner is (0,0,1). Use int5-8_(0,0,1) in reverse.
# Original: defineTriangle(mob_(0,1,1), mob_(0,0,1), linZ_(0,0,1));
defineTriangle(mob_(0,1,1), int8_(0,0,1), linZ_(0,0,1));
defineTriangle(int8_(0,0,1), int7_(0,0,1), linZ_(0,0,1));
defineTriangle(int7_(0,0,1), int6_(0,0,1), linZ_(0,0,1));
defineTriangle(int6_(0,0,1), int5_(0,0,1), linZ_(0,0,1));
defineTriangle(int5_(0,0,1), mob_(0,0,1), linZ_(0,0,1));

# Edge mob_(0,1,1) -> mob_(1,1,1) is a +X edge, negative corner is (0,1,1). Use int1-4_(0,1,1).
# Original: defineTriangle(mob_(0,1,1), mob_(1,1,1), linY_(0,1,0));
defineTriangle(mob_(0,1,1), int1_(0,1,1), linY_(0,1,0));
defineTriangle(int1_(0,1,1), int2_(0,1,1), linY_(0,1,0));
defineTriangle(int2_(0,1,1), int3_(0,1,1), linY_(0,1,0));
defineTriangle(int3_(0,1,1), int4_(0,1,1), linY_(0,1,0));
defineTriangle(int4_(0,1,1), mob_(1,1,1), linY_(0,1,0));

# Edge mob_(0,1,1) -> mob_(0,1,0) is a -Z edge, negative corner is (0,1,0). Use int9-12_(0,1,0) in reverse.
# Original: defineTriangle(mob_(0,1,1), mob_(0,1,0), linY_(0,1,0));
defineTriangle(mob_(0,1,1), int12_(0,1,0), linY_(0,1,0));
defineTriangle(int12_(0,1,0), int11_(0,1,0), linY_(0,1,0));
defineTriangle(int11_(0,1,0), int10_(0,1,0), linY_(0,1,0));
defineTriangle(int10_(0,1,0), int9_(0,1,0), linY_(0,1,0));
defineTriangle(int9_(0,1,0), mob_(0,1,0), linY_(0,1,0));

# Edge mob_(0,1,1) -> mob_(0,0,1) is a -Y edge, negative corner is (0,0,1). Use int5-8_(0,0,1) in reverse.
# Original: defineTriangle(mob_(0,1,1), mob_(0,0,1), linX_(0,0,0));
defineTriangle(mob_(0,1,1), int8_(0,0,1), linX_(0,0,0));
defineTriangle(int8_(0,0,1), int7_(0,0,1), linX_(0,0,0));
defineTriangle(int7_(0,0,1), int6_(0,0,1), linX_(0,0,0));
defineTriangle(int6_(0,0,1), int5_(0,0,1), linX_(0,0,0));
defineTriangle(int5_(0,0,1), mob_(0,0,1), linX_(0,0,0));

# Edge mob_(0,1,1) -> mob_(0,1,0) is a -Z edge, negative corner is (0,1,0). Use int9-12_(0,1,0) in reverse.
# Original: defineTriangle(mob_(0,1,1), mob_(0,1,0), linX_(0,0,0));
defineTriangle(mob_(0,1,1), int12_(0,1,0), linX_(0,0,0));
defineTriangle(int12_(0,1,0), int11_(0,1,0), linX_(0,0,0));
defineTriangle(int11_(0,1,0), int10_(0,1,0), linX_(0,0,0));
defineTriangle(int10_(0,1,0), int9_(0,1,0), linX_(0,0,0));
defineTriangle(int9_(0,1,0), mob_(0,1,0), linX_(0,0,0));

</S5>
"""



if __name__ == "__main__":
    print("\n--- TEST: All steps correct ---")
    is_valid, prompt = validate_algorithm(valid_algorithm)
    print(f"Valid: {is_valid}")
    if not is_valid:
        print(prompt)
    else:
        print("All checks passed!")

# --- Additional Test Cases ---

def test_missing_semicolons():
    algo = valid_algorithm.replace('svCenter =', 'svCenter =')\
        .replace(';', '', 2)  # Remove first two semicolons
    ok, msg = validate_algorithm(algo)
    print('Test missing semicolons:', 'PASS' if not ok and 'semicolon' in msg else 'FAIL')

def test_missing_tags():
    # Remove <S2> and </S2>
    algo = valid_algorithm.replace('<S2>', '').replace('</S2>', '')
    ok, msg = validate_algorithm(algo)
    print('Test missing tags:', 'PASS' if not ok and 'Missing required tags' in msg else 'FAIL')

def test_missing_linX_in_S5():
    # Remove all linX from S5
    import re
    blocks = valid_algorithm.split('<S5>')
    before, after = blocks[0], blocks[1]
    s5, rest = after.split('</S5>')
    s5 = re.sub(r'linX_\([^)]*\)', 'REMOVED_LINX', s5)
    algo = before + '<S5>' + s5 + '</S5>' + rest
    ok, msg = validate_algorithm(algo)
    print('Test missing linX in S5:', 'PASS' if not ok and 'linX' in msg else 'FAIL')

def test_mob_defined_in_S1_S2():
    # Add mob definition to S1 and S2
    algo = valid_algorithm.replace('<S1>', '<S1>\n    mob = (1,2,3);')
    algo = algo.replace('<S2>', '<S2>\n    mob = (4,5,6);')
    ok, msg = validate_algorithm(algo)
    print('Test mob defined in S1 and S2:', 'PASS' if not ok and 'mob' in msg else 'FAIL')

def test_missing_linX_linY_linZ_in_S2():
    # Remove linX, linY, linZ from S2
    import re
    def remove_lines_with_vars(text, vars):
        lines = text.split('\n')
        return '\n'.join([l for l in lines if not any(v+' =' in l for v in vars)])
    blocks = valid_algorithm.split('<S2>')
    before, after = blocks[0], blocks[1]
    s2, rest = after.split('</S2>')
    s2 = remove_lines_with_vars(s2, ['linX', 'linY', 'linZ'])
    algo = before + '<S2>' + s2 + '</S2>' + rest
    ok, msg = validate_algorithm(algo)
    print('Test missing linX, linY, linZ in S2:', 'PASS' if not ok and all(v in msg for v in ['linX','linY','linZ']) else 'FAIL')

if __name__ == '__main__':
    print('Running additional test cases...')
    test_missing_semicolons()
    test_missing_tags()
    test_missing_linX_in_S5()
    test_mob_defined_in_S1_S2()
    test_missing_linX_linY_linZ_in_S2()


