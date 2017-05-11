'''
Contains utility functions
'''

import shlex
from subprocess import Popen

DIGITS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def shlex_cmd(command, print_cmd=False):
    '''
    A convinience function that submits a command using shlex and
    subprocess.Popen.

    command: string
        The command to be submitted
    '''
    if print_cmd:
        print(command)
    proc = Popen(shlex.split(command))
    proc.communicate()
    