# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

class ModelOverview:
    """
    A class to provide an overview of a PyTorch model.
    It includes the model's architecture, number of parameters, and a summary.
    """
    def __init__(self):
        pass

    def get_model_summary(self, model) -> str:
        """
        Generate a summary of the model architecture and number of parameters.
        """
        model_summary = str(model)
        return model_summary