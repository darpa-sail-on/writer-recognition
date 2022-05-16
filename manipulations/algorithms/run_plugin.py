import json
import logging
import os
import subprocess
import sys
import tarfile
import traceback
import copy


def _runCustomPlugin(command, im, source, target, **kwargs):
    import copy
    commands = copy.deepcopy(command['command'])
    mapping = copy.deepcopy(command['mapping'])
    executeOk = False
    for k, command in commands.items():
        if sys.platform.startswith(k):
            executeWith(command, im, source, target, mapping, **kwargs)
            executeOk = True
            break
    if not executeOk:
        executeWith(commands['default'], im, source, target, mapping, **kwargs)
    return None, None


def executeWith(executionCommand, im, source, target, mapping, **kwargs):
    shell=False
    if executionCommand[0].startswith('s/'):
        executionCommand[0] = executionCommand[0][2:]
        shell = True
    kwargs = mapCmdArgs(kwargs, mapping)
    kwargs['inputimage'] = source
    kwargs['outputimage'] = target
    for i in range(len(executionCommand)):
        try:
            executionCommand[i] = executionCommand[i].format(**kwargs)
        except KeyError as e:
            logging.getLogger('maskgen').warn('Argument {} not provided for {}'.format(e.message,executionCommand[0]))
    ret = subprocess.call(executionCommand,shell=shell)
    if ret != 0:
        raise RuntimeError('Plugin {} failed with code {}'.format(executionCommand[0],ret))

def mapCmdArgs(args, mapping):
    import copy
    newargs = copy.copy(args)
    if mapping is not None:
        for key, val in args.iteritems():
            if key in mapping:
                if val not in mapping[key] or mapping[key][val] is None:
                    raise ValueError('Option \"' + str(val) + '\" is not permitted for this plugin.')
                newargs[key] = mapping[key][val]
    return newargs

