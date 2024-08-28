import gzip as gz

def extract_species(species_file, uniref_file):
    with open(species_file, 'r') as file:
        species_list = [line.strip() for line in file]
    match = False
    print_line = ""
    for line in gz.open(uniref_file, "rt"):
        if line.startswith('>') and any(species.lower() in line.lower() for species in species_list):
            if print_line != "":
                print(print_line)
                print_line = ""
            print(line.strip())
            match = True
        elif match and not line.startswith('>'):
            # print(line.strip())
            print_line += line.strip()
        else:
            if print_line != "":
                print(print_line)
                print_line = ""
            match = False


if __name__ == '__main__':
    import sys
    extract_species(sys.argv[1], sys.argv[2])