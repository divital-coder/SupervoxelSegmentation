from grammar import GEOMETRIC_ALGORITHM_GRAMMAR
from geometric_utils import lin_p, orthogonal_projection_onto_plane, line_plane_intersection
from lark_transformer import GeometricAlgorithmTransformer
from execute_algorithm_on_data import get_highest_w_number,execute_grid_algorithm, initialize_cell, process_step_for_cell,UNINITIALIZED_POINT
from is_line_intersect import check_star_convexity,find_star_convexity_intersections
from is_water_tight import is_mesh_watertight,check_watertight
from visualize_mesh import visualize_geometry
from algo_validation_utils import check_tags_presence, check_s5_point_usage, check_statement_termination, extract_step_blocks
from algo_validation_utils import check_vars_present, check_vars_absent
import re
from lark import Lark, Transformer, v_args, LarkError


def validate_algorithm_syntax(script_string, step=0):
    """
    Validates a geometric algorithm script against the language grammar, with step-specific variable checks.

    Args:
        script_string (str): The algorithm script to validate.
        step (int): Step number (0 for normal, 1-5 for step-specific checks).

    Returns:
        str or None: None if the script is valid, otherwise a string
                     containing the error message.
    """
    
    # Step-specific variable checks
    if step == 1:
        # S1: Must contain both 'svCenter' and 'originalSvCenter', but none of the required_points
        required = ['svCenter', 'originalSvCenter']
        forbidden = ["linX", "linY", "linZ", "mob"] + [f"int{i}" for i in range(1, 13)]
        if not check_vars_present(script_string, required):
            return "Step 1 must contain both 'svCenter' and 'originalSvCenter'."
        if not check_vars_absent(script_string, forbidden):
            return "Step 1 must not contain any of: linX, linY, linZ, mob, int1-int12."
    elif step == 2:
        script_string=""" 
            svCenter = (0.0,0.0,0.0);
            """+script_string

        # S2: Must contain linX, linY, linZ, but none of the other required_points
        required = ["linX", "linY", "linZ"]
        forbidden = ["mob"] + [f"int{i}" for i in range(1, 13)]
        if not check_vars_present(script_string, required):
            return "Step 2 must contain linX, linY, linZ."
        if not check_vars_absent(script_string, forbidden):
            return "Step 2 must not contain mob or int1-int12."
    elif step == 3:
        script_string=""" 
            svCenter = (0.0,0.0,0.0);
            """+script_string


        # S3: Must contain linX, linY, linZ, mob, but none of the int1-int12
        required = [ "mob"]
        forbidden = [f"int{i}" for i in range(1, 13)]
        if not check_vars_present(script_string, required):
            return "Step 3 must contain mob."
        if not check_vars_absent(script_string, forbidden):
            return "Step 3 must not contain int1-int12."
    elif step == 4:
        script_string=""" 
        svCenter = (0.0,0.0,0.0);
        linX = (0.0,0.0,0.0);
        linY = (0.0,0.0,0.0);
        linZ = (0.0,0.0,0.0);
        mob = (0.0,0.0,0.0);

            """+script_string


        # S4: All required points must be defined
        required = [f"int{i}" for i in range(1, 13)]
        if not check_vars_present(script_string, required):
            return "Step 4 must contain all of  int1-int12."
    elif step == 5:
                script_string=""" 
        svCenter = (0.0,0.0,0.0);
        linX = (0.0,0.0,0.0);
        linY = (0.0,0.0,0.0);
        linZ = (0.0,0.0,0.0);
        mob = (0.0,0.0,0.0);
        int1 = (0.0,0.0,0.0);
        int2 = (0.0,0.0,0.0);
        int3 = (0.0,0.0,0.0);
        int4 = (0.0,0.0,0.0);
        int5 = (0.0,0.0,0.0);
        int6 = (0.0,0.0,0.0);
        int7 = (0.0,0.0,0.0);
        int8 = (0.0,0.0,0.0);
        int9 = (0.0,0.0,0.0);
        int10 = (0.0,0.0,0.0);
        int11 = (0.0,0.0,0.0);
        int12 = (0.0,0.0,0.0);
        
            """+script_string


    # --- Original grammar validation logic ---
    # Use a minimal mock context for validation purposes.
    # This allows the transformer to instantiate without a full grid.
    mock_context = {
        'current_indices': (0, 0, 0),
        'grid_spacing': (1.0, 1.0, 1.0),
        'read_grid_data': {
            (0,0,0): {'control_points': {}}
        },
        'control_points': {}
    }

    
    # Mock weights, as they are required by the transformer.
    mock_weights = {f'w{i}': 0.5 for i in range(151)}
    
    transformer = GeometricAlgorithmTransformer(mock_context, mock_weights)
    parser = Lark(GEOMETRIC_ALGORITHM_GRAMMAR, parser='lalr', transformer=transformer)

    try:
        parser.parse(script_string)
        # If parsing and transforming succeed, the syntax is valid.
        return None
    except LarkError as e:
        # This catches syntax errors found by the parser itself.
        return f"Syntax Error: {e}"
    except (NameError, TypeError, ValueError, SyntaxError) as e:
        # This catches semantic errors raised by the transformer.
        return f"Semantic Error: {e}"
    except Exception as e:
        # Catch any other unexpected errors.
        return f"An unexpected error occurred during validation: {e}"
    
def extract_step_blocks(algorithm_string):
    """
    Extracts the code for each <S1> ... <S5> block as a dict {step: code}.
    """
    blocks = {}
    for i in range(1, 6):
        match = re.search(rf"<S{i}>(.*?)</S{i}>", algorithm_string, re.DOTALL)
        if match:
            blocks[f"S{i}"] = match.group(1).strip()
        else:
            blocks[f"S{i}"] = None
    return blocks


def validate_algorithm_syntax_main(algorithm_string: str) -> tuple[bool, str]:
    """
    Comprehensive validation: grammar for each step, tags, S5 usage, semicolons, etc.
    Returns (is_valid, correction_prompt) for LLM correction.
    """
    correction_messages = []
    is_valid = True

    # 1. Check tags presence
    tags_ok, tags_msg = check_tags_presence(algorithm_string)
    if not tags_ok:
        is_valid = False
        correction_messages.append(f"Missing required step tags: {tags_msg}.\nAdd all <S1> through <S5> blocks, each with code inside.")

    # 2. Extract step blocks
    step_blocks = extract_step_blocks(algorithm_string)

    # 3. Grammar check for the full script (cumulative, not per step)
    # This avoids false negatives due to missing context between steps.
    # Only check grammar if all steps are present and non-empty.
    all_steps_present = all(code is not None for code in step_blocks.values())
    all_steps_nonempty = all(code and any(l.strip() and not l.strip().startswith('#') for l in code.split('\n')) for code in step_blocks.values() if code is not None)
    if all_steps_present and all_steps_nonempty:
        # Concatenate all step codes in order
        full_script = '\n'.join(step_blocks[f'S{i}'] for i in range(1, 6))
        grammar_error = validate_algorithm_syntax(full_script)
        if grammar_error:
            is_valid = False
            correction_messages.append(f"Grammar error in full algorithm: {grammar_error}\nReview the EBNF grammar and allowed constructs. Only use permitted functions, variable naming, and syntax. End every statement with a semicolon.")
    else:
        # If any step is missing or empty, keep the per-step checks for user guidance
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

    # 4. S5 point usage check
    s5_ok, s5_msg = check_s5_point_usage(algorithm_string)
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
    
if __name__ == "__main__":
    # Example usage
    script = """
# Define the modified supervoxel center by applying a weighted offset.
# Weights w1, w2, w3 control the displacement along each axis.
# The offset is scaled by half the grid spacing (rx, ry, rz) and centered.
svCenter = originalSvCenter_(0,0,0) + vector(w1 * rx * 0.5 - rx * 0.25, w2 * ry * 0.5 - ry * 0.25, w3 * rz * 0.5 - rz * 0.25);


# Define points along the negative axes using linear interpolation.
# w4, w5, w6 control the position along the segments connecting to neighbor centers.
linX = lin_p(originalSvCenter_(0,0,0), originalSvCenter_(-1,0,0), w4);
linY = lin_p(originalSvCenter_(0,0,0), originalSvCenter_(0,-1,0), w5);
linZ = lin_p(originalSvCenter_(0,0,0), originalSvCenter_(0,0,-1), w6);

# Define the shared edge points of the negative corner.
# These points are formed by a two-step linear interpolation between the four
# supervoxel centers that meet at that edge, ensuring a shared position.
p1 = lin_p(originalSvCenter_(0,0,0), originalSvCenter_(-1,0,0), w7);
p2 = lin_p(originalSvCenter_(0,-1,0), originalSvCenter_(-1,-1,0), w7);
edgeXY = lin_p(p1, p2, w8);

p3 = lin_p(originalSvCenter_(0,0,0), originalSvCenter_(0,0,-1), w9);
p4 = lin_p(originalSvCenter_(0,-1,0), originalSvCenter_(0,-1,-1), w9);
edgeYZ = lin_p(p3, p4, w10);

p5 = lin_p(originalSvCenter_(0,0,0), originalSvCenter_(-1,0,0), w11);
p6 = lin_p(originalSvCenter_(0,0,-1), originalSvCenter_(-1,0,-1), w11);
edgeXZ = lin_p(p5, p6, w12);

# Define the main oblique point (mob) as a weighted blend of the 8 centers
# forming the negative corner cube. This creates a trilinearly interpolated point.
# First, interpolate along the X-axis for the four parallel edges.
pXY0 = lin_p(originalSvCenter_(0,0,0), originalSvCenter_(-1,0,0), w13);
pXY1 = lin_p(originalSvCenter_(0,-1,0), originalSvCenter_(-1,-1,0), w13);
pXY2 = lin_p(originalSvCenter_(0,0,-1), originalSvCenter_(-1,0,-1), w13);
pXY3 = lin_p(originalSvCenter_(0,-1,-1), originalSvCenter_(-1,-1,-1), w13);

# Second, interpolate the results along the Y-axis.
pXYZ0 = lin_p(pXY0, pXY1, w14);
pXYZ1 = lin_p(pXY2, pXY3, w14);

# Finally, interpolate the last two points along the Z-axis to get the final mob point.
mob = lin_p(pXYZ0, pXYZ1, w15);

# Define intermediate points to add flexibility to the final mesh faces.
# int1, int2, int3 are corner points blended between the edge and the mob point.

# Define alternative face centers by projecting mob onto the main planes.
# The plane is defined by the original center and its two relevant neighbors.
projYZ = ortho_proj(mob_(0,0,0), originalSvCenter_(0,0,0), originalSvCenter_(0,-1,0), originalSvCenter_(0,0,-1));
projXZ = ortho_proj(mob_(0,0,0), originalSvCenter_(0,0,0), originalSvCenter_(-1,0,0), originalSvCenter_(0,0,-1));
projXY = ortho_proj(mob_(0,0,0), originalSvCenter_(0,0,0), originalSvCenter_(-1,0,0), originalSvCenter_(0,-1,0));

# Blend the projected points with the linear points to create final face centers.
int4 = lin_p(linX_(0,0,0), projYZ, w19);
int5 = lin_p(linY_(0,0,0), projXZ, w20);
int6 = lin_p(linZ_(0,0,0), projXY, w21);


# Define the 24 triangles forming the six faces of the watertight polyhedron.
# All triangles are the bases of tetrahedrons with svCenter as their common apex.

# 1. Negative-X Face: Centered on int4_(0,0,0), with corners from int1 points.
defineTriangle(int4_(0,0,0), int1_(0,0,0), int1_(0,0,-1));
defineTriangle(int4_(0,0,0), int1_(0,0,-1), int1_(0,-1,-1));
defineTriangle(int4_(0,0,0), int1_(0,-1,-1), int1_(0,-1,0));
defineTriangle(int4_(0,0,0), int1_(0,-1,0), int1_(0,0,0));

# 2. Negative-Y Face: Centered on int5_(0,0,0), with corners from int2 points.
defineTriangle(int5_(0,0,0), int2_(0,0,0), int2_(-1,0,0));
defineTriangle(int5_(0,0,0), int2_(-1,0,0), int2_(-1,0,-1));
defineTriangle(int5_(0,0,0), int2_(-1,0,-1), int2_(0,0,-1));
defineTriangle(int5_(0,0,0), int2_(0,0,-1), int2_(0,0,0));

# 3. Negative-Z Face: Centered on int6_(0,0,0), with corners from int3 points.
defineTriangle(int6_(0,0,0), int3_(0,0,0), int3_(0,-1,0));
defineTriangle(int6_(0,0,0), int3_(0,-1,0), int3_(-1,-1,0));
defineTriangle(int6_(0,0,0), int3_(-1,-1,0), int3_(-1,0,0));
defineTriangle(int6_(0,0,0), int3_(-1,0,0), int3_(0,0,0));

# 4. Positive-X Face: This is the Negative-X face of the neighbor at (1,0,0).
defineTriangle(int4_(1,0,0), int1_(1,0,0), int1_(1,0,-1));
defineTriangle(int4_(1,0,0), int1_(1,0,-1), int1_(1,-1,-1));
defineTriangle(int4_(1,0,0), int1_(1,-1,-1), int1_(1,-1,0));
defineTriangle(int4_(1,0,0), int1_(1,-1,0), int1_(1,0,0));

# 5. Positive-Y Face: This is the Negative-Y face of the neighbor at (0,1,0).
defineTriangle(int5_(0,1,0), int2_(0,1,0), int2_(-1,1,0));
defineTriangle(int5_(0,1,0), int2_(-1,1,0), int2_(-1,1,-1));
defineTriangle(int5_(0,1,0), int2_(-1,1,-1), int2_(0,1,-1));
defineTriangle(int5_(0,1,0), int2_(0,1,-1), int2_(0,1,0));

# 6. Positive-Z Face: This is the Negative-Z face of the neighbor at (0,0,1).
defineTriangle(int6_(0,0,1), int3_(0,0,1), int3_(0,-1,1));
defineTriangle(int6_(0,0,1), int3_(0,-1,1), int3_(-1,-1,1));
defineTriangle(int6_(0,0,1), int3_(-1,-1,1), int3_(-1,0,1));
defineTriangle(int6_(0,0,1), int3_(-1,0,1), int3_(0,0,1));

  """
    
    result = validate_algorithm_syntax_main(script)
    
    if result is None:
        print("The script is valid.")
    else:
        print(f"Validation failed: {result}")
