## Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
##
## Developed by: Frederik De Ceuster - University College London & KU Leuven
## _________________________________________________________________________


def  get_length (input_file):
    """
    Return the number of lines in the input file.
    """
    with open(input_file) as file:
        for i, line in enumerate(file):
            pass
    return i + 1


def read_list (input_file):
    """
    Return the contents of the line as a list.
    """
    with open(input_file) as file:
        list = [int(line) for line in file]
    return list
