from domainlab.algos.observers.a_observer import AObVisitor


class ObVisitorCleanUp(AObVisitor):
    """
    decorator of observer
    """
    def __init__(self, observer):
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
