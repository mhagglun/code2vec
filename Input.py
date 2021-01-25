def read_file(self, input_filename):
    with open(input_filename, 'r') as file:
        return file.readlines()
