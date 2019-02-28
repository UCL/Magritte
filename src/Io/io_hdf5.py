## Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
##
## Developed by: Frederik De Ceuster - University College London & KU Leuven
## _________________________________________________________________________


import numpy as np
import h5py  as hp


def read_length (io_file, file_name):
    """
    Return the number of lines in the input file.
    """
    with hp.File (io_file, 'r') as file:
        try:
            # Try to open the object
            object = file [file_name]
            # Check if it is a Dataset
            if isinstance (object, hp.Dataset):
                return object.len()
        except:
            # Get name of object we need to count
            object_name = file_name.split('/')[-1]
            # Get containing group
            group_name = file_name[:-len(object_name)]
            # Count occurences
            length = 0
            for key in file[group_name].keys():
                if (object_name in key):
                    length += 1
            return length
    # Error if not yet returned
    raise ValueError ('file_name is no Group or Dataset.')


def read_attribute (io_file, file_name):
    """
    Return the contents of the attribute
    """
    with hp.File (io_file, 'r') as file:
        object    = file_name.split('.')[0]
        attribute = file_name.split('.')[1]
        return file[object].attrs[attribute]


def write_attribute (io_file, file_name, data):
    """
    Write the data to the attribute
    """
    with hp.File (io_file) as file:
        object    = file_name.split('.')[0]
        attribute = file_name.split('.')[1]
        # Make sure all groups exists, if not create them
        # NOTE: ASSUMES THAT WORD IS WRITTEN TO A GROUP !
        group = ''
        for g in object.split('/'):
            group += f'/{g}'
            file.require_group (group)
        file[object].attrs[attribute] = data


def read_number (io_file, file_name):
    """
    Return the contents of the attribute
    """
    return int(read_attribute(io_file, file_name))


def write_number (io_file, file_name, data):
    """
    Write the data to the attribute
    """
    write_attribute (io_file, file_name, data)


def read_array (io_file, file_name):
    """
    Return the contents of the data array.
    """
    with hp.File (io_file, 'r') as file:
        return np.array (file.get (file_name))


def write_array (io_file, file_name, data):
    """
    Write the contents to the data array.
    """
    with hp.File (io_file) as file:
        # Delete if dataset already exists
        try:
            del file[file_name]
        except:
            pass
        # Make sure all groups exists, if not create them
        # NOTE: ASSUMES THAT NUMBER IS WRITTEN TO A DATASET !
        group = ''
        for g in file_name.split('/')[:-1]:
            group += f'/{g}'
            file.require_group (group)
        # Write dataset
        file.create_dataset (name=file_name, data=data)
