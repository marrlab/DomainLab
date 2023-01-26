import torch

from train.train_jigsaw import TrainerJigsaw, TrainerJigsawCalmDown


class FactoryJigsaw():
    """
    """
   def create_model(self):
        model = ModelJigsawAlex(y_dim=y_dim, jigsaw_classes=self.__class__.jigsaw_n_classes + 1)
        self.model = model.to(self.device)
        return self.model

    def create_trainer(self, model_path=None):
        if not hasattr(self, "model"):
            self.create_model()
        self.perf = self.create_perf(model_path, self._scenario)
        self.algo = TrainerJigsaw(self.model, self.perf, self.conf, self.device)
        return self.algo

class FactoryJigsawCaffe(FactoryJigsaw):
    def create_model(self):
        # model = AlexNetCaffe(jigsaw_classes=self.__class__.jigsaw_n_classes + 1, n_classes=self._scenario.y_dim)
        model = caffenet(jigsaw_classes=self.__class__.jigsaw_n_classes + 1, classes=self._scenario.y_dim)   # FIXME: the initialization plays a vital role!
        self.model = model.to(self.device)
        return self.model

    def create_trainer(self, model_path=None):
        if not hasattr(self, "model"):
            self.create_model()
        self.perf = self.create_perf(model_path, self._scenario)
        self.algo = TrainerJigsawCalmDown(self.model, self.perf, self.conf, self.device)
        return self.algo
