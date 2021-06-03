class Params:
    def __init__(self,
                 file_in,
                 novelty_method,
                 random_seed
                 ):
        self.file_in = file_in  # type: dict
        self.random_seed = random_seed  # type: int
        self.novelty_method = novelty_method  # type: str
