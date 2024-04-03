class transform_functions:
    def __init__(self) -> None:
        pass

    @staticmethod
    def hardlim(input :int) -> int:
        if input >= 0:
            return 1
        else:
            return 0
        
    @staticmethod
    def hardlims(input :int) -> int:
        if input >= 0:
            return 1
        else:
            return -1