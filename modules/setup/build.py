#from os         import path, makedirs, getcwd
from subprocess import check_output


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
    # Run command and capture output
    output = check_output([command], shell=True)
    # Convert bytelike object to string
    output = output.decode("utf-8")
    # Show output
    print(output)


#def __main__():
#    cwd = getcwd()
#    # Create build folder
#    createFolder(cwd + '/build')
#    compile()
