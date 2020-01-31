from transformers import BertConfig


class HyperParameters(object):
    """
    Hyper parameter of the model
    """

    def __init__(
            self,
            max_length: int = 128,
            epochs=4,
            batch_size=32,
            learning_rate=2e-5,
            fp16=True,
            fp16_opt_level="O1",
            max_grad_norm=1.0,
            warmup_steps=0.1,
    ) -> None:
        self.max_length = max_length
        """Max length of sentence"""
        self.epochs = epochs
        """Num of epochs"""
        self.batch_size = batch_size
        """Size of mini batch"""
        self.learning_rate = learning_rate
        """Learning rate"""
        self.fp16 = fp16
        """Enable FP16 mixed-precision training"""
        self.fp16_opt_level = fp16_opt_level
        """NVIDIA APEX Level, ['O0', 'O1', 'O2', and 'O3'], see: https://nvidia.github.io/apex/amp.html"""
        self.max_grad_norm = max_grad_norm
        """Max gradient normalization"""
        self.warmup_steps = warmup_steps
        """Steps of warm-up for learning rate"""

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class Config(BertConfig):
    """
    Config of MatchModel
    """

    def __init__(self, max_len=512, algorithm="BertForSimMatchModel", **kwargs):
        super(Config, self).__init__(**kwargs)
        self.max_len = max_len
        self.algorithm = algorithm
