from src.Variable import Variable


class Function:
    '''
    __call__方法是一个特殊的python方法，定义了这个方法后，
    当f=Function()时，就可以通过编写f(...)来调用__call__方法了
    '''
    def __call__(self, input):
        x = input.data
        output = Variable(self.forward(x))
        return output

    def forward(self, x):
        raise NotImplementedError