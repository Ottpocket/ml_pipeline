"""
Utilities for record keeper classes
"""

def print_block(heading, metric=None, time= None, size = 'small'):
    """ prints out a block of text enclosed in *s 
    
    ARGUMENTS
    -----------------
    heading: (str) title of heading
    body: (None, `metric`) body of block
    size: (['big', 'small']) how many ***s to add
    """
    if size == 'big':
        line = '******************************************'+\
               '******************************************'
        mini_line = '******************************************'
    elif size == 'small':
        line = '******************'
        mini_line = '********'
    else:
        msg = f"""
        ERROR: `size` arg must be either `big` or `small`. Received:
        `{size}`
        """
        raise Exception(msg)

    print(line)
    print(heading)
    if metric is not None:
        print(mini_line)
        print(metric)
    if time is not None:
        print(mini_line)
        print(f'Time: {time}')
    print(line)
    
