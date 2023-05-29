from .train_options import TrainOptions


class Valptions(TrainOptions):
    """This class includes validation options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        self.isTrain = False

        parser = TrainOptions.initialize(self, parser)
        # loading parameters
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='val')

        return parser
