from typing import ClassVar

class PipelineState:
    ''' Global state of the whole pipeline.
    Contains process-wide state needed by modules that are configured indirectly.
    
    Args:
        train: Whether the module is in training mode or not.
            Note: Validation is either train or test, depending on the print_meta_val flagg. See module's validation step.
    '''
    train: ClassVar[bool] = True
