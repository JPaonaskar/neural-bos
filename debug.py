import torch

import pix2pix
import models

i2i_hard = pix2pix.Image2Image()
i2i_config = models.Image2Image(torch.device('cuda'))

# check optimizers
print(
    'Gen Opt:',
    str(i2i_hard.opt_gen) == str(i2i_config.opt_gen)
)
print(
    'Dis Opt:',
    str(i2i_hard.opt_dis) == str(i2i_config.opt_dis)
)

# check parameters
print(
    'Gen Params:',
    len(list(
        i2i_hard.net_gen.parameters()
    )) == len(list(
        i2i_config.gen.parameters()
    ))
)

print(
    'Dis Params:',
    len(list(
        i2i_hard.net_dis.parameters()
    )) == len(list(
        i2i_config.dis.parameters()
    ))
)

def clean_string(module:str) -> str:
    '''
    Clean module string to just the core functions

    Args:
        module (str) : module string

    Returns:
        cleaned (str) : cleaned string
    '''
    cleaned = ''

    layer = []
    contents = []
    i = 0
    for line in module.splitlines():
        line = line.strip()

        if line[-1] == '(':
            if ' x ' in line:
                layer.append(int(
                    line.split(' x ')[0].split('): ')[1].strip()
                ))
            else:
                layer.append(1)
            contents.append('')

        elif line == ')':
            reps = layer.pop()
            content = contents.pop()

            for rep in range(reps):
                for newline in content.splitlines():
                    if newline:
                        cleaned += f'({i}): ' + newline + '\n'
                        i += 1

        elif line[-1] == ')':
            contents[-1] += line.split('): ')[1] + '\n'

    return cleaned

# check strings
print(
    'Dis Modules:',
    clean_string(str(i2i_hard.net_dis)) == clean_string(str(i2i_config.dis))
)

print(
    'Gen Modules:',
    clean_string(str(i2i_hard.net_gen)) == clean_string(str(i2i_config.gen))
)

# check line by line
hard_gen_lines = clean_string(str(i2i_hard.net_gen)).splitlines()
config_gen_lines = clean_string(str(i2i_config.gen)).splitlines()
i = 0
while True:
    if (i >= len(hard_gen_lines) - 1) or (i >= len(config_gen_lines) - 1):
        break

    if hard_gen_lines[i] != config_gen_lines[i]:
        print('False:', hard_gen_lines[i], '!=', config_gen_lines[i])

    i += 1