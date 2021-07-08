"""
This file consists of functions that classify samples
on the basis of heuristical rules

NOTE: These functions were used at the start to aid in modelling
as explained in the writeup.
"""

import config

def slur_exists(text):
    """ 
    True if the text contains one of these slur words
    The slurs obtained via data exploration
    """
    slurs = ['nigga', 'faggot', 'fuck', 'aids']
    for slur in slurs:
        if slur in text:
            return 1
            break
    return 0

def potentially_problematic(df):
    """
    Find potentially problematic nouns or verbs
    """

    attack_nvs = ' '.join(df[df.attack == 1]['noun_comp'].sum()).split() + ' '.join(df[df.attack == 1]['verb_comp'].sum()).split()
    non_attack_nvs = ' '.join(df[df.attack == 0]['noun_comp'].sum()).split() + ' '.join(df[df.attack == 0]['verb_comp'].sum()).split()
    potentially_problematic = list(set(attack_nvs) - set(non_attack_nvs))

    return potentially_problematic

def is_profanity(text):
    """
    Does the text contain a profane word?
    """

    profane_list = open(config.PROFANITY_LIST, 'r').readlines()
    profane_list = [w.replace('\n', '') for w in profane_list]

    for profane_word in profane_list:
        if profane_word in text:
            return 1
            break
    return 0