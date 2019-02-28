


class Routine :

    def __init__(self, src=None):
        if src :
            copy_constr(self, src)
        else :
            self.name = None #name of the gate
            self.params = [] # parameters
            self.args = []  # qbit arguments
            self.glist = [] # list of gates
    def copy_constr(self, src):
        self.name = src.name
        self.params = src.params
        self.args = src.args
        self.glist = src.glist

class Gate :
    def __init__(self):
        self.name = None #name of the gate
        self.params = [] #parameters
        self.qblist  = [] #qbit arguments
class Element:
    def __init__(self):
        self.name = None
        self.index = None
        self.value = None
