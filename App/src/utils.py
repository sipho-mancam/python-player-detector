from threading import Thread


class CThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=None, daemon:bool=None, **kwargs)->None:
        super().__init__(group=group,target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self.return_value = None
        self.target = target
        self.args = args
        self.kwargs = kwargs


    def run(self):

        try:
            if self.target is not None and self.args is not None:
                self.return_value = self.target(*self.args, **self.kwargs)
            else:
                self.return_value = self.target()
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self.target, self.args, self.kwargs


        # self.return_value = self.target(self.args[0], self.args[1])
    
    def get_return(self):
        return self.return_value