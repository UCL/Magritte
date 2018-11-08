from subprocess import call, check_output
from os         import path
from re         import search, MULTILINE
from time       import time

dir_path = path.dirname(path.realpath(__file__))


def appendToLine(line, text):
    # Remove newline character
    line = line.rstrip('\n')
    # Append text to line
    line = f'{line}{text}\n'
    # Done
    return line


class SlurmJob:
    '''
    Settings for slurm job
    '''
    def __init__(self):
        self.template   = '/slurm_templates/template_slurm_submit_CSD3_Peta4-Skylake.txt'
        self.jobName    = 'Magritte_test'
        self.slurmName  = 'slurm_submit'
        self.id         = 0
        self.nNodes     = 1
        self.nTasks     = 1
        self.nThrds     = 1
        self.totalTime  = '00:00:02'
        self.executable = '/home/dc-dece1/MagritteProjects/Lines_3D_LTE/build/examples/example_Lines_Lime1'
        self.workDir    = '/home/dc-dece1/MagritteProjects/Lines_3D_LTE/'
        self.submitTime = 0

    def create(self):
        # Read in the slurm_submit template
        with open(dir_path + self.template, 'r') as file:
            fullText = [line for line in file]
        # Fill out the required fields    
        fullText[12] = appendToLine(fullText[12], self.jobName)
        fullText[16] = appendToLine(fullText[16], self.nNodes)
        fullText[19] = appendToLine(fullText[19], self.nTasks)
        fullText[21] = appendToLine(fullText[21], self.totalTime)
        fullText[84] = appendToLine(fullText[84], f'\"{self.executable}\"')
        fullText[90] = appendToLine(fullText[90], f'\"{self.workDir}\"')
        fullText[96] = appendToLine(fullText[96], self.nThrds)
        # Write the slurm_submit file
        with open(self.workDir + self.slurmName, 'w') as file:
            file.writelines(fullText)

    def submit(self):
        # Make sure slurm job is created
        self.create()
        # Submit slurm job and catch output
        output  = check_output([f'sbatch {self.workDir + self.slurmName}'], shell=True)
        # Remove end of line character from output
        output  = output.rstrip(b'\n')
        # Extract job id (which is bytelike object)
        output  = output.split(b' ')[3]
        # Convert bytelike object to string
        self.id = output.decode("utf-8")
        # Set submitTime
        self.submitTime = time()

    def cancel(self):
	# Cancel the slurm job
        call([f'scancel {self.id}'], shell=True)

    def status(self):
        # Get slurm queue
        #output = check_output([f'squeue -j {self.id}'], shell=True)
        output = check_output([f'showq'], shell=True)
        # Convert bytelike object to string
        output = output.decode("utf-8")
        # Extract line containig job id
        output = search(f'^.*{self.id}.*$', output, MULTILINE)
        # Check for output
        if not output:
            print(f'\n---> Job {self.id} is not in the slurm queue (anymore).')
        else:
            # Extract regex output
            output = output.group(0)
            # Get time since submission
            elapsedTime = time() - self.submitTime
            elapsedTime = '{:.0f}'.format(elapsedTime)
            # print status
            print(f'Submitted {elapsedTime} seconds ago: {output}', end='\r')
        # Done
        return output

    def status_cont(self):
        # Continuously write status until job is out of queue
        while self.status():
            pass 
        # Done
        return
