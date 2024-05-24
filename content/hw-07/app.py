from model import Model


class App:
    def __init__(self, args):
        self._args = args
        self._model = Model()

    def run(self):
        if self._args.mode == 'train':
            test_size = None
            if self._args.split:
                test_size = self._args.split
            self._model.train(self._args.data, test_size)
            self._model.upload(self._args.model)
            if self._args.test:
                print(self._model.get_stats(self._args.test))
            elif self._args.split:
                print(self._model.get_stats(self._args.data, test_size))
        elif self._args.mode == 'predict':
            self._model = Model(self._args.model)
            for nums in self._model.get_predict(self._args.data):
                print(nums)
        else:
            raise RuntimeError('mode not found')
