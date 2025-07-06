import re

def check_tags_presence(algorithm_string: str) -> tuple[bool, str]:
    """
    Checks for the presence of all required <Sx> and </Sx> tags in the algorithm string.
    Returns (True, "") if all are present, else (False, message).
    """
    tags = [f"<S{i}>" for i in range(1, 6)] + [f"</S{i}>" for i in range(1, 6)]
    missing_tags = [tag for tag in tags if tag not in algorithm_string]
    if not missing_tags:
        return (True, "")
    else:
        return (False, f"Missing required tags: {', '.join(missing_tags)}")


def check_s5_point_usage(algorithm_string: str) -> tuple[bool, str]:
    """
    Checks that all required points are present anywhere in the algorithm string.
    Returns (True, "") if all are present, else (False, message).
    """
    required_points = ["linX", "linY", "linZ", "mob"] + [f"int{i}" for i in range(1, 13)]
    unused_points = [point for point in required_points if point not in algorithm_string]
    if not unused_points:
        return (True, "")
    else:
        return (False, f"The following points were not found in the algorithm string: {', '.join(unused_points)}. All points must be present.")

def check_statement_termination(algorithm_string: str) -> tuple[bool, str]:
    """
    Checks that each non-empty, non-comment line in each <Sx> block ends with a semicolon.
    Returns (True, "") if all are correct, else (False, message).
    """
    s_blocks = re.findall(r"<S\d>(.*?)</S\d>", algorithm_string, re.DOTALL)
    error_messages = []
    for i, block in enumerate(s_blocks, 1):
        lines = block.strip().split('\n')
        for line_num, line in enumerate(lines, 1):
            cleaned_line = line.strip()
            if not cleaned_line or cleaned_line.startswith('#'):
                continue
            if not cleaned_line.endswith(';'):
                error_messages.append(f"Error in <S{i}>, line {line_num}: Statement does not end with a semicolon. -> '{line}'")
    if not error_messages:
        return (True, "")
    else:
        return (False, "\n".join(error_messages))

def extract_step_blocks(algorithm_string):
    """
    Extracts the contents of each <Sx> block as a dictionary {Sx: content or None}.
    """
    blocks = {}
    for i in range(1, 6):
        match = re.search(rf"<S{i}>(.*?)</S{i}>", algorithm_string, re.DOTALL)
        if match:
            blocks[f"S{i}"] = match.group(1).strip()
        else:
            blocks[f"S{i}"] = None
    return blocks

def check_vars_present(code: str, required_vars: list[str]) -> bool:
    """
    Returns True if all required_vars are present as substrings in code.
    This is a simplified check using the 'in' operator.
    """
    for var in required_vars:
        if var not in code:
            return False
    return True

def check_vars_absent(code: str, forbidden_vars: list[str]) -> bool:
    """
    Returns True if none of the forbidden_vars are present as substrings in code.
    This is a simplified check using the 'in' operator.
    """
    for var in forbidden_vars:
        if var in code:
            return False
    return True
