import logging
import sys

logging.basicConfig(stream=sys.stdout,
    level=logging.DEBUG)
'''
root = logging.getLogger()
root.setlevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
root.addHandler(handler)
'''

class ANSIColor:
    red = "\u001b[31m"
    black = "\u001b[30m"
    green = "\u001b[32m"
    yellow = "\u001b[33m"
    blue = "\u001b[34m"
    magenta = "\u001b[35m"
    cyan = "\u001b[36m"
    white = "\u001b[37m"
    reset = "\u001b[0m"

def comma_concat(argv) :
    return "".join([str(arg) for arg in argv])

def star(num_stars, color = ANSIColor.red) :
    starstring = color
    for _ in range(num_stars) :
        starstring += "*"
    starstring += ANSIColor.reset
    return starstring

def title(*stringv) :
    string = comma_concat(stringv)
    logging.info(
        (
        star(5) + "[" +
        ANSIColor.red + 
        "{}" + 
        ANSIColor.reset +
        "]" + star(5)).format(string)
        )


def section(*stringv) :
    #print("\n\n")
    string = comma_concat(stringv)
    logging.info(
        (
        star(1) + "[" +
        ANSIColor.green + 
        "{}" + 
        ANSIColor.reset +
        "]").format(string)
        )

def subsection(*stringv) :
    string = comma_concat(stringv)
    #print("\n")
    logging.info((
        star(2) + "[" + 
        ANSIColor.yellow +
        "{}" + 
        ANSIColor.reset +
        "]"
        ).format(string)
        )

def debug(*stringv) :
    string = comma_concat(stringv)
    #print("\n")
    logging.debug("{}".format(string))

def result(*stringv) :
    string = comma_concat(stringv)
    logging.info(
        (ANSIColor.red +
        "[Result]"+
        ANSIColor.reset +
        "{}").format(string)
        )

