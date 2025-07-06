import re

def get_highest_w_number(text_string):
    """
    Extracts the highest number from variables starting with 'w' in a given string.

    Args:
        text_string: The input string containing variables like w1, w2, w27, etc.

    Returns:
        The highest integer found after a 'w'.
        Returns None if no variables starting with 'w' are found.
    """
    # Find all sequences of digits that are preceded by the letter 'w'
    w_numbers = re.findall(r'w(\d+)', text_string)

    # If no such patterns are found, return None
    if not w_numbers:
        return None

    # Convert the list of found strings to a list of integers
    w_integers = [int(num) for num in w_numbers]

    # Return the maximum value from the list
    return max(w_integers)

# Example usage with the string you provided:
step4 = """
        int1 = lin_p(mob_(0,0,0), mob_(1,0,0), range(0.49, 0.51, w16));
        int2 = lin_p(int1, mob_(1,0,0), range(0.49, 0.51, w17));
        int3 = lin_p(int2, mob_(1,0,0), range(0.49, 0.51, w18));
        int4 = lin_p(int3, mob_(1,0,0), range(0.49, 0.51, w19));

        int5 = lin_p(mob_(0,0,0), mob_(0,1,0), range(0.49, 0.51, w20));
        int6 = lin_p(int5, mob_(0,1,0), range(0.49, 0.51, w21));
        int7 = lin_p(int6, mob_(0,1,0), range(0.49, 0.51, w22));
        int8 = lin_p(int7, mob_(0,1,0), range(0.49, 0.51, w23));

        int9 = lin_p(mob_(0,0,0), mob_(0,0,1), range(0.49, 0.51, w24));
        int10 = lin_p(int9, mob_(0,0,1), range(0.49, 0.51, w25));
        int11 = lin_p(int10, mob_(0,0,1), range(0.49, 0.51, w26));
        int12 = lin_p(int11, mob_(0,0,1), range(0.49, 0.51, w21));

    """

highest_w = get_highest_w_number(step4)
print(f"The highest number next to variable 'w' is: {highest_w}")