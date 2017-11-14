
# Take the Jacobian of the system of rate equations


NEQ = 32

species_index_1 = -1

with open("standard_code/jacobian_hd.txt", 'r') as header_file:
    header = header_file.readlines()


with open("../src/sundials/rate_equations.cpp", 'r') as read_file, open("../src/sundials/jacobian.cpp", 'w') as write_file:


    # Write the header for the jacobian.cpp file

    for line in header:
        write_file.write(line)


    # Write the rate_equations.cpp file and calculate the Jacobian

    while True:

        line = read_file.readline()


        # Break at end of file

        if not line: break


        line = line.split(" ")


        if "loss" in line:

            line_loss = line


            # Next line should be a "form line"

            line_form = read_file.readline()

            line_form = line_form.split(" ")


            # Check if this is indeed a "form line"

            if not "form" in line_form:

                print "ERROR INPUT DOES NOT HAVE THE RIGHT FORMAT"

                break


            species_index_1 = species_index_1 + 1


            # Extract the part containing the equation, remove first symbol at front and semicolon at end

            line_loss = line_loss[4]
            line_form = line_form[4]


            line_loss = line_loss.split(";")[0]
            line_form = line_form.split(";")[0]


            # split in terms, assuming all loss terms contribute negatively and form terms contribute positively

            line_loss = line_loss.split("-")
            line_form = line_form.split("+")


            for species_index_2 in range(NEQ):

                jacobian_entry = "  IJth(J, " + str(species_index_2) + ", " + str(species_index_1) + ") = ("

                diff_spec      = "*Ith(y," + str(species_index_2) + ")"


                # for all terms, if the term depends on diff_spec remove diff_spec once and add to Jacobian
                #

                for term in line_loss:

                    if diff_spec in term:

                        nr   = term.count(diff_spec)
                        term = term.replace(diff_spec, "", 1)

                        for i in range(nr):
                            jacobian_entry = jacobian_entry + "-" + term

                jacobian_entry = jacobian_entry + "+0.0)*Ith(y," + str(species_index_1) + ")"


                if species_index_1 == species_index_2:

                    for term in line_loss:

                        if (len(term)>0):

                            jacobian_entry = jacobian_entry + "-" + term


                for term in line_form:

                    if diff_spec in term:

                        nr   = term.count(diff_spec)
                        term = term.replace(diff_spec, "", 1)

                        for i in range(nr):
                            jacobian_entry = jacobian_entry + "+" + term


                jacobian_entry = jacobian_entry + "+0.0;\n"


                write_file.write(jacobian_entry)


            write_file.write(jacobian_entry)


    # Write the footer for the jacobian.cpp file

    write_file.write("\n\n  return(0);\n\n}\n\n/*-----------------------------------------------------------------------------------------------*/")
