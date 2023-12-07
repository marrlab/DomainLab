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
        self.observer.clean_up()

    def accept(self, trainer):
        self.observer.accept(trainer)

    def update(self, epoch):
        return self.observer.update(epoch)

    def clean_up(self):
        self.observer.clean_up()
