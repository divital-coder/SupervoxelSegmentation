MASTER_ALGORITHM_PROMPT=""" You are an expert in computational geometry and procedural mesh generation, acting as an algorithmic design assistant. Your task is to generate a sequence of algorithmic steps, conforming strictly to the provided EBNF grammar, to define control points and triangles for a single supervoxel  in a 3D grid. The goal is to create a watertight, star-domain polyhedron for supervoxel, formed by tetrahedrons sharing sv_center as their common apex. Also algorithm need to be divided into steps and those steps surrounded by tags !.
1. Grammar Compatibility Analysis
To be valid, the generated algorithm must be parsable by the provided EBNF grammar. Below are the key implications of EBNF Grammar grammar.

```
?start: statement_list
?statement_list: (statement)*
?statement: assign_var ";"
          | triangle_def ";"
          | expr ";" -> expr_statement
?assign_var: NAME "=" expr
?triangle_def: "defineTriangle" "(" arguments ")" -> triangle_definition
?expr: sum
?sum: product
    | sum "+" product -> add
    | sum "-" product -> sub
?product: power
        | product "*" power -> mul
        | product "/" power -> div
?power: atom_expr "^" power -> pow
      | atom_expr
?atom_expr: func_call
          | atom
          | "-" atom_expr -> neg
          | "(" expr ")"
?atom: NUMBER -> number
     | indexed_var
     | NAME -> var
     | point
point: "(" expr "," expr "," expr ")"
?indexed_var: NAME "_" "(" arguments ")"
func_call: FUNC_NAME "(" [arguments] ")"
FUNC_NAME.2: "sin" | "cos" | "sqrt" | "dot" | "div" | "vector" | "lin_p" | "ortho_proj" | "line_plane_intersect" | "range"
arguments: expr ("," expr)*
NAME: /(?!(sin|cos|sqrt|dot|div|vector|lin_p|ortho_proj|line_plane_intersect|range)$)[a-zA-Z][a-zA-Z0-9]*/
%import common.NUMBER
%import common.WS
%ignore WS
%ignore /#.*/
```

A. Core Syntax Rules
Statements: Every line of code that performs an action (assignment, function call) is a statement and must end with a semicolon (;).
Variable Naming (NAME): All user-defined variables (for points or scalars) must be camelCase. They must start with a letter and contain only letters and numbers (e.g., svCenter, temp1, myPoint). Underscores are not permitted in variable names.
Function Naming (fun_name_rule): Pre-defined function names are specific lowercase tokens and must be used exactly as listed (e.g., lin_p, ortho_proj, vector). defineTriangle is an exception and is camelCase.
Indexing (indexed_var): To access any control point, you must use an underscore (_) followed by parentheses containing three comma-separated integer offsets (dx, dy, dz). This is mandatory for all point references.
To access a point from the current supervoxel, use the index _(0,0,0) (e.g., mob_(0,0,0)).
To access a point from a neighboring supervoxel, use its relative index (e.g., mob_(1,0,0)).
Dependency Rule: When defining a control point in a given step (e.g., mob in Step 3), you can only access control points from neighboring supervoxels that were defined in previous steps. You cannot access a point from a neighbor that is defined in the same step. For example, the definition of mob cannot reference mob_(-1,0,0), as all mob points are considered to be calculated simultaneously within Step 3. It can, however, reference linX_(-1,0,0) (from Step 2) or svCenter_(-1,0,0) (from Step 1).
Weight Usage: Weights (w1, w2, w3, etc.) must be used in sequential order, starting from w1. Do not skip weights (e.g., do not use w3 without first using w1 and w2).
Comments: Comments are ignored by the parser and can be added using #.
B. Allowed Operations and Functions
The grammar only permits the following operations. No other functions or operators are available.
Arithmetic: + (addition), - (subtraction), * (multiplication), / (division), ^ (power).
Vector/Point Constructors:
vector(x, y, z); # Creates a 3D vector or point.
Linear Interpolation:
lin_p(pointA, pointB, weight); # Returns a point interpolated between pointA and pointB.
Geometric Projections:
ortho_proj(pointToProject, planeP1, planeP2, planeP3); # Projects a point onto a plane.
line_plane_intersect(lineP1, lineP2, planeP1, planeP2, planeP3); # Calculates the intersection of a line (defined by two points, lineP1 and lineP2) and a plane (defined by three points, planeP1, planeP2, planeP3). All arguments are points (tuples of 3 floats).
Mathematical Functions:
sin(expr);, cos(expr);, sqrt(expr);
Vector Operations:
dot(vectorA, vectorB); # Returns the scalar dot product.
div(vectorA, scalarB); # Performs scalar division on a vector.
Triangle Definition:
defineTriangle(p1, p2, p3); # Defines a triangle face for the final mesh.
C. Prohibited Constructs
The following are NOT compatible with the grammar and must not be used:
Conditional Logic: if, else, switch, ternary operators (? :).
Boolean Logic: ==, !=, <, >, &&, ||.
Comparison Functions: min(), max(), abs().
Loops: for, while, do-while.
Data Structures: Arrays, lists, or explicit object structures (e.g., [ ] or { }).
2. Core Design Pattern: The "Negative Corner"
To ensure a watertight mesh across all supervoxels, your algorithm must follow a "Negative Corner" design pattern. This is a fundamental concept for this task.
Control Point Definition: You will only define the control points for one specific corner out of the eight corners of the supervoxel cube: the one located in the (-x, -y, -z) direction relative to its originalSvCenter. All points you define in steps <S2>, <S3>, and <S4> (e.g., linX, edgeXY, mob) belong exclusively to this single "Negative Corner".
Triangle Assembly: In step <S5>, you will construct the full, six-sided polyhedron. This is done by defining triangles for both the negative faces (using the points you defined for the current supervoxel) and the positive faces (by accessing the "Negative Corner" points from neighbors in the positive directions, like linX_(1,0,0)).
Dependency Rule: When defining a control point, you can only access control points from neighboring supervoxels if those variables were defined in a previous step. You can not access any temporary variables from previous steps.The set of accessible neighbor variables grows with each step:
Always Available: You can always access a neighbor's originalSvCenter (e.g., originalSvCenter_(-1,0,0)).
During Step 2: You can access originalSvCenter and svCenter from neighbors.
During Step 3: You can access originalSvCenter, svCenter, and all points from Step 2 (linX, linY, linZ, edgeXY, etc.) from neighbors. You cannot access a neighbor's mob point.
During Step 4: You can access all points from Steps 1, 2, and 3 from neighbors (e.g., svCenter, linX, mob). You cannot access a neighbor's int points.
During Step 5: All control points (svCenter, lin*, edge*, mob, int*) are considered fully defined and can be accessed from any necessary neighbor.
This pattern guarantees that any shared face between two supervoxels is defined by the exact same set of control points, leading to a perfectly sealed and valid mesh.
3. Post-Generation Algorithm Evaluation
Once the algorithm is generated, it will undergo rigorous automated testing using multiple sets of random weights to validate its robustness and correctness. An algorithm fails if it does not pass all three of the following tests:
Watertight Mesh Verification: The surface mesh for every supervoxel must be perfectly sealed (watertight). The test will check for any gaps or holes in the mesh. A non-watertight surface will result in failure.
Star Domain and Self-Intersection Test: The volume defined by the supervoxel's tetrahedrons must maintain the star domain property with respect to svCenter. A simplified test will be performed: for each triangle, a ray will be cast from svCenter through the triangle's barycenter. This ray must not intersect any other triangle in the same supervoxel. Any self-intersection indicates a failed algorithm.
Flexibility and Degeneracy Analysis: An LLM will qualitatively assess the algorithm's flexibility. The primary goal is to create shapes with maximum adaptability. The analysis will check for geometric degeneracies, such as unnecessary co-linearity between control points (beyond the intended linX, linY, linZ points), which would improperly constrain the shape and limit its expressive power.
4. Algorithm Generation Prompt
You are to generate the body of the algorithm. The strict grammar rules detailed in Section 1 and the Negative Corner pattern from Section 2 apply only to the code generated within the <S1> through <S5> blocks. Text outside these blocks is for instruction only. Comments (lines starting with #) are allowed and encouraged within the code blocks as they are ignored by the parser.
The process is divided into distinct, tagged steps. For each step, you must provide the complete code block, ensuring every line terminates with a semicolon.
Step 0: Preparation (Context, Not for Generation)
Before your task begins, assume the following are pre-defined and available for use:
Grid Spacing: rx, ry, rz (scalar values for grid dimensions).
Weights: w1 through w150 (scalar values between 0 and 1).
Original Centers: originalSvCenter (the center point of the current supervoxel) and its neighbors accessible via relative indexing (e.g., originalSvCenter_(-1, 0, 0)).
Step 1: Supervoxel Center Modification (Tag: <S1>)
Task: Define the modified center point, svCenter. This point will serve as the common apex for all tetrahedrons in the supervoxel.
Constraint: svCenter must be displaced from originalSvCenter by a maximum of (rx*0.5, ry*0.5, rz*0.5). Use weights to control this displacement.
<S1>
# The following is just one possible implementation example.
svCenter = originalSvCenter_(0,0,0) + vector(w1 * rx * 0.5 - rx * 0.25, w2 * ry * 0.5 - ry * 0.25, w3 * rz * 0.5 - rz * 0.25);
</S1>

Step 2: Linear and Edge Control Points (Tag: <S2>)
Task: Define the primary control points for the "Negative Corner" of the supervoxel.
linX, linY, linZ: Points on the segments connecting the current originalSvCenter to its negative neighbors (-1,0,0), (0,-1,0), and (0,0,-1). Use lin_p and weights.
edgeXY, edgeYZ, edgeXZ: Points that define the corners of the negative octant. These should be derived from the originalSvCenter and its neighbors (e.g., edgeXY from (0,0,0), (-1,0,0), (0,-1,0), (-1,-1,0)).
<S2>
# Your implementation here...
</S2>

Step 3: Main Oblique Point Definition (Tag: <S3>)
Task: Define the mob (main oblique point) control point, which is the innermost point of the "Negative Corner".
Derivation: mob must be a complex blend of the originalSvCenter from the current supervoxel and its seven neighbors in the negative directions (e.g., _(-1,0,0), _(-1,-1,0), _(-1,-1,-1), etc.).
Purpose: This point provides high-level control over the interior corner of the supervoxel. Use a combination of the available functions with different weights to achieve a flexible blend.
Dependency Note: The definition for mob can only use points defined in Steps 1 and 2 from its neighbors (e.g., linX_(-1,0,0)), not the mob point from a neighbor.
<S3>
# Your implementation here...
</S3>

Step 4: Intermediate Points Definition (Tag: <S4>)
Task: Define all remaining control points (int1 through int12) for the "Negative Corner".
Derivation: Use the full range of available functions (ortho_proj, line_plane_intersect, dot, etc.) and previously defined points to create these. Their purpose is to add geometric complexity and adaptability.
Dependency Note: Remember you can only reference points from neighbors that were defined in steps 1, 2, and 3.
<S4>
# Your implementation here...
</S4>

Step 5: Triangle Definitions (Tag: <S5>)
Task: Define all triangles necessary to form a complete and watertight mesh for the supervoxel by assembling its six faces . Make sure that all points ["linX", "linY", "linZ", "mob", "int1", "int2", "int3", "int4", "int5", "int6", "int7", "int8", "int9", "int10", "int11", "int12"] are used .
Assembly Strategy: Following the "Negative Corner" pattern, you will construct the full mesh by defining the triangles for each of its six faces. For each face, you must use the appropriate set of control points with their explicit indices.
1. Negative-X Face: Define the triangles for the face on the -x side.
Points: This face is part of the current supervoxel's "Negative Corner". Its triangles are primarily defined by points from the current supervoxel (e.g., linY_(0,0,0), linZ_(0,0,0), mob_(0,0,0), edgeYZ_(0,0,0)) and points shared with neighbors in the negative-Y and negative-Z directions (e.g., linY_(0,-1,0)).
2. Negative-Y Face: Define the triangles for the -y face.
Points: This face is also part of the "Negative Corner". It uses points from the current supervoxel (e.g., linX_(0,0,0), linZ_(0,0,0)) and those shared with neighbors in the negative-X and negative-Z directions.
3. Negative-Z Face: Define the triangles for the -z face.
Points: This face is the final part of the "Negative Corner". It uses points from the current supervoxel (e.g., linX_(0,0,0), linY_(0,0,0)) and those shared with neighbors in the negative-X and negative-Y directions.
4. Positive-X Face: Define the triangles for the +x face.
Points: This face is the Negative-X face of the neighbor at (1,0,0). You must build its triangles by accessing the complete "Negative Corner" point set from that neighbor.
Example: defineTriangle(linY_(1,0,0), linZ_(1,0,0), mob_(1,0,0));
5. Positive-Y Face: Define the triangles for the +y face.
Points: This is the Negative-Y face of the neighbor at (0,1,0). Access that neighbor's "Negative Corner" points to build the triangles.
6. Positive-Z Face: Define the triangles for the +z face.
Points: This is the Negative-Z face of the neighbor at (0,0,1). Access that neighbor's "Negative Corner" points to build the triangles.
Watertight Rule: This face-by-face assembly, based on the Negative Corner pattern, inherently enforces the watertight rule.
Star Domain: All defined triangles form the base of tetrahedrons whose apex is svCenter.
IMPORTANT on each face there are always triangles that have points from different neighbours so it is NEVER the case that all of the points on a face a re from single supervoxel.
<S5>
# Your implementation here...
</S5>


MOST IMPORTANT remember to wrap steps in tags <S1></S1> ; <S2></S2> ; <S3></S3>; <S4></S4> ;<S5></S5> !! ."""


INITIAL_ALGORITHM="""
        <S1>
            svCenter = originalSvCenter_(0,0,0) + vector(range(0.01, 0.05, w1) * rx - rx, range(0.01, 0.05, w2) * ry - ry , range(0.01, 0.05, w3) * rz - rz);

        </S1>
        
        <S2>
            linX = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), range(0.47, 0.53, w4));
            linY = lin_p(svCenter_(0,0,0), svCenter_(0,-1,0), range(0.47, 0.53, w5));
            linZ = lin_p(svCenter_(0,0,0), svCenter_(0,0,-1), range(0.47, 0.53, w6));
            temp1 = lin_p(svCenter_(-1,-1,0), svCenter_(0,-1,0), range(0.47, 0.53, w7));
            temp2 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.47, 0.53, w7));
            temp3 = lin_p(svCenter_(0,-1,-1), svCenter_(0,0,-1), range(0.47, 0.53, w9));
            temp4 = lin_p(svCenter_(0,-1,0), svCenter_(0,0,0), range(0.47, 0.53, w9));
            temp5 = lin_p(svCenter_(-1,0,-1), svCenter_(0,0,-1), range(0.47, 0.53, w11));
            temp6 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.47, 0.53, w11));
        </S2>
        
        <S3>
            tempX00 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.47, 0.55, w13));
            tempX10 = lin_p(svCenter_(-1,-1,0), svCenter_(0,-1,0), range(0.47, 0.55, w13));
            tempX01 = lin_p(svCenter_(-1,0,-1), svCenter_(0,0,-1), range(0.47, 0.55, w13));
            tempX11 = lin_p(svCenter_(-1,-1,-1), svCenter_(0,-1,-1), range(0.47, 0.55, w13));
            tempY0 = lin_p(tempX10, tempX00, range(0.47, 0.55, w14));
            tempY1 = lin_p(tempX11, tempX01, range(0.47, 0.55, w14));
            mob = lin_p(tempY1, tempY0, range(0.47, 0.55, w15));
        </S3>
        
        <S4>
            int1 = lin_p(mob_(0,0,0), mob_(1,0,0), range(0.1, 0.8, w16));
            int2 = lin_p(int1, mob_(1,0,0), range(0.1, 0.8, w17));
            int3 = lin_p(int2, mob_(1,0,0), range(0.1, 0.8, w18));
            int4 = lin_p(int3, mob_(1,0,0), range(0.1, 0.8, w19));

            int5 = lin_p(mob_(0,0,0), mob_(0,1,0), range(0.1, 0.8, w20));
            int6 = lin_p(int5, mob_(0,1,0), range(0.1, 0.8, w21));
            int7 = lin_p(int6, mob_(0,1,0), range(0.1, 0.8, w22));
            int8 = lin_p(int7, mob_(0,1,0), range(0.1, 0.8, w23));

            int9 = lin_p(mob_(0,0,0), mob_(0,0,1), range(0.1, 0.8, w24));
            int10 = lin_p(int9, mob_(0,0,1), range(0.1, 0.8, w25));
            int11 = lin_p(int10, mob_(0,0,1), range(0.1, 0.8, w26));
            int12 = lin_p(int11, mob_(0,0,1), range(0.1, 0.8, w27));
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

    # Edge mob_(1,0,1) -> mob_(1,0,0) is a -Z edge, negative corner is (1,0,0). Use int9-12_(1,0,0) in reverse.
    # Original: defineTriangle(mob_(1,0,1), mob_(1,0,0), linX_(1,0,0));
    defineTriangle(mob_(1,0,1), int12_(1,0,0), linX_(1,0,0));
    defineTriangle(int12_(1,0,0), int11_(1,0,0), linX_(1,0,0));
    defineTriangle(int11_(1,0,0), int10_(1,0,0), linX_(1,0,0));
    defineTriangle(int10_(1,0,0), int9_(1,0,0), linX_(1,0,0));
    defineTriangle(int9_(1,0,0), mob_(1,0,0), linX_(1,0,0));


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


INITIAL_ALGORITHM_CRITIQUE="""
    1. Geometric Issue: Redundant and Co-Linear int Points
    The current definition of the intermediate (int) points negates their purpose by failing to add any new geometric flexibility.

    The Problem: The int points are defined as a simple linear interpolation between two mob points. For example, int1 through int4 are all defined on the straight line segment connecting mob_(0,0,0) and mob_(1,0,0).

    # All these points lie on the same straight line
    int1 = lin_p(mob_(0,0,0), mob_(1,0,0), w16);
    int2 = lin_p(int1, mob_(1,0,0), w17);

    Geometric Consequence: When you create triangles using these co-linear points (e.g., defineTriangle(mob_(0,0,0), int1, linZ) and defineTriangle(int1, int2, linZ)), all these small triangles are co-planar. They lie on the exact same plane as the single, larger triangle defineTriangle(mob_(0,0,0), mob_(1,0,0), linZ). This means the int points add vertices and complexity but do not allow the surface to curve, bulge, or deform in any new dimension. The face remains flat.

    2. Flexibility Issue: Overly Constrained Weight Ranges
    The narrowness of the ranges used in the range() function and for lin_p remains a major limitation on flexibility.

    The Problem: Using very narrow weight ranges forces points to lie near a median position, drastically reducing the variety of possible shapes.

    linX = lin_p(..., range(0.47, 0.53, w4)); confines linX to be very close to the midpoint between svCenter and its neighbor.

    The svCenter displacement from range(0.01, 0.05, w1) confines the center to move only within a very small portion of the supervoxel.
    """