from domainlab.algos.observers.a_observer import AObVisitor


class ObVisitorCleanUp(AObVisitor):
    """
    decorator of observer, instead of using if and else to decide clean up or not, we use decorator
    """

    def __init__(self, observer):
        super().__init__()
        self.observer = observer

    def after_all(self):
        self.observer.after_all()
        self.observer.clean_up()  # FIXME should  be self.clean_up???

    def accept(self, trainer):
        self.observer.accept(trainer)

    def update(self, epoch, flag_info=False):
        return self.observer.update(epoch, flag_info)

    def clean_up(self):
        self.observer.clean_up()

    @property
    def model_sel(self):
        return self.observer.model_sel

    @model_sel.setter
    def model_sel(self, model_sel):
        self.observer.model_sel = model_sel

    @property
    def metric_te(self):
        return self.observer.metric_te

    @property
    def metric_val(self):
        return self.observer.metric_val
