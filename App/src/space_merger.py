
from input_pipeline import InputPipeline


class SpaceMerger:
    def __init__(self, input_pipelines:list[InputPipeline])->None:
        self.__input_pipelines_list = input_pipelines
        self.__stream_results = [] # a list of tuples containing the current results 

    # stream1 = 0-30%; stream2 = 31-70%; stream3 = 71-100%
    def merge(self, stream1:tuple, stream2:tuple, stream3:tuple)->dict:
        result = dict()


        return result