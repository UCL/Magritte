#from os         import path, makedirs, getcwd
from subprocess import Popen, PIPE
from sys        import argv


def run(command):
    # Run command pipe output
    process = Popen(command, stdout=PIPE, shell=True)
    # Continuously read output and print
    while True:
        # Extract output
        line = process.stdout.readline().rstrip()
        # Break if there is no more line
        if not line:
            break
        # Convert bytelike object to string
        line = line.decode("utf-8")
        # Output line
        print(line)


#def createFolder(folder):
#    # Try to create folder if it does not exists
#    try:
#        if not path.exists(folder):
#            makedirs(folder)
#    except OSError:
#        print ('Error: Creating directory. ' + folder)


def compile(MagritteSetupFolder, ProjectFolder):
    # Prepare command
    command  =  'bash '
    command += f'{MagritteSetupFolder}build.sh '
    command += ProjectFolder
    # Run command
    run(command)


if __name__ == '__main__':
    # Run compile with provided arguments  
    if (len(argv) != 3):
        print('ERROR: Please provide the 2 arguments:')
        print(' 1) Full path to Magritte Setup Folder')
        print(' 2) Full path to Project Folder')
    else:
        compile(argv[1],argv[2])
