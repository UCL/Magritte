from subprocess import call, check_output


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
    def __init__():
        self.template   = 'slurm_templates/template_slurm_submit_CSD3_Peta4-Skylake.txt'
        self.jobName    = 'Magritte_test',
        self.slurmName  = 'slurm_submit'
        self.slurmId    = 'slurm_submit'
        self.nNodes     = 1,
        self.nTasks     = 1,
        self.nThrds     = 1,
        self.totalTime  = '00:00:02',
        self.executable = '\"$HOME/MagritteProjects/Lines_3D_LTE/build/examples/example_Lines_Lime1.exe\"',
        self.workDir    = '\"$HOME/MagritteProjects/Lines_3D_LTE/\"'

    def write(self):
        # Read in the slurm_submit template
        with open(self.template, 'r') as file:
            fullText = [line for line in file]
        # Fill out the required fields    
        fullText[12] = appendToLine(fullText[12], self.jobName)
        fullText[16] = appendToLine(fullText[16], self.nNodes)
        fullText[19] = appendToLine(fullText[19], self.nTasks)
        fullText[21] = appendToLine(fullText[21], self.totalTime)
        fullText[84] = appendToLine(fullText[84], self.executable)
        fullText[90] = appendToLine(fullText[90], self.workDir)
        fullText[96] = appendToLine(fullText[96], self.nThrds)
        # Write the slurm_submit file
        with open(self.slurmName,     'w') as file:
            file.writelines(fullText)

    def submit(self):
        self.slurmId = check_output([f'sbatch {self.slurmName}'], shell=True)

    def delete(self):
        call([f'scancel {self.slurmId}'], shell=True)
