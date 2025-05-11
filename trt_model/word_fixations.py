class WordFixations:
    """
    Class to store all fixations for a specific word.
    """

    def __init__(self, fixations=[], TRT=0):
        self.fixations = fixations
        self.TRT = TRT

    def __str__(self):
        return f"Fixations: {self.fixations}, TRT: {self.TRT}"
