from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.set_defaults(serial_batches=True)
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        parser.add_argument('--file_time', type=str, default='', help='start training time, this is designd to distinguish the folder to store checkpoints')

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # rewrite devalue values
        self.isTrain = False
        return parser