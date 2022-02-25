"""Library with layers for Technotrack task #1"""
import numpy as np


class Linear:
    """ Линейный слой """
    def __init__(self, input_size, output_size, no_b=False):
        """
        Инициализация собственных значений весов
        :param input_size: число признаков на входе
        :param output_size: число признаков на выходе
        :param no_b: флаг на отсутсвие смещения
        """
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=[input_size, output_size])
        self.no_b = no_b
        self.x = None
        self.dldw = None
        if not no_b:
            self.biases = np.random.normal(loc=0.0, scale=0.01, size=[1, output_size])
            self.dldb = None

    def forward(self, x):
        """
        Прямой проход линейного слоя
        :param x: входной тензор
        :return: выходной тензор y = xW + b
        """
        self.x = x  # сохраняем значения для обратного прохода
        y = x @ self.weights
        if not self.no_b:
            y += self.biases
        return y

    def backward(self, dldy):
        """
        Обратный проход
        :param dldy: производная функции ошибок по выходу y линейного слоя dL/dy
        :return: производная функции оибок по входу х линейного слоя dL/dx
        """
        self.dldw = self.x.transpose()@dldy/self.x.shape[0]  # расчитываем производную по весам
        dldx = dldy@self.weights.transpose()  # производня по входным фичам
        if not self.no_b:
            self.dldb = np.sum(dldy)/dldy.shape[0]
        return dldx

    def step(self, learning_rate):
        """
        Шаг в сторону антиградиента
        :param learning_rate: скорость обучения
        """
        self.weights = self.weights - learning_rate * self.dldw
        if not self.no_b:
            self.biases = self.biases - learning_rate * self.dldb


class Sigmoid:
    """Слой активации: сигмоида"""
    def __init__(self):
        self.y = None
    
    def forward(self, x):
        """
        Прямой проход
        :param x: входные признаки np.array size (N, d)
        :return: результат применениия функции активации
        """
        y = 1/(1+np.exp(-x))
        self.y = y  # сохраняем результат для обратного прохода
        return y

    def backward(self, dldy):
        """
        Обратный проход
        :param dldy: производная функции ошибок по выходу y слоя активации dL/dy
        :return: производная функции оибок по входу х слоя активации dL/dx
        """
        dldx = dldy * (self.y*(1-self.y))
        return dldx

    def step(self, learning_rate):
        pass


class ELU:
    """Слой активации: ELU"""
    def __init__(self, alpha):
        self.alpha = alpha
        self.x = None

    def forward(self, x):
        """
        Прямой проход
        :param x: входные признаки
        :return: результат активации
        """
        self.x = x
        return np.where(self.x < 0, self.alpha * (np.exp(x) - 1), x)

    def backward(self, dldy):
        """
        Обратный проход
        :param dldy: производная по выходу
        :return: производная по входу
        """
        return dldy*np.where(self.x < 0, self.alpha * np.exp(self.x), 1)

    def step(self, learning_rate):
        pass


class ReLU:
    """Функция активации ReLU"""
    def __init__(self, a):
        self.a = a
        self.x = None

    def forward(self, x):
        """Прямой проход"""
        self.x = x
        return np.where(x < 0, self.a * x, x)
      
    def backward(self, dldy):
        """Обратный проход"""
        return dldy*np.where(self.x < 0, self.a, 1)

    def step(self, learning_rate):
        pass


class Tanh:
    """Функция активации Tanh"""
    def __init__(self):
        self.y = None

    def forward(self, x):
        """Прямой проход"""
        self.y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return self.y

    def backward(self, dldy):
        """Обратный проход"""
        return dldy*(1-self.y**2)

    def step(self, learning_rate):
        pass


class SoftMax_NLLLoss:
    """Функция ошибки: Softmax и NLL loss"""
    def __init__(self):
        self.y = None

    def forward(self, x):
        """Прямой проход, возвращает SoftMax"""
        self.y = np.exp(x)/np.expand_dims(np.sum(np.exp(x), axis=1), axis=1)
        return self.y

    def get_loss(self, y, numb):
        """Расчет функции ошибки"""
        dldy = np.zeros((y.size, numb))
        for i in range(y.size):
            dldy[i][y[i]] = 1
        eps = 10**(-9)
        loss = - dldy * np.log(self.y+eps) - (1 - dldy) * np.log(1 - self.y+eps)
        return np.sum(loss)/dldy.shape[0]
        
    def backward(self, y):
        """Обратный проход"""
        dldy = np.zeros((y.size, self.y.shape[1]))
        for i in range(y.size):
            dldy[i][y[i]] = 1
        dldx = self.y-dldy
        return dldx


class MSE_Error:
    """Функция ошибок MSE"""
    def __init__(self):
        self.x = None

    def forward(self, x):
        """Прямой проход"""
        self.x = x
        return x

    def backward(self, y):
        """Обратный проход"""
        dldy = np.zeros((y.size, y.max()+1))
        for i in range(y.size):
            dldy[i][y[i]] = 1
        dldx = 2*self.x-2*dldy
        return dldx

    def get_loss(self, y):
        dldy = np.zeros((y.size, y.max()+1))
        for i in range(y.size):
            dldy[i][y[i]] = 1
        loss = (self.x-dldy)**2
        return np.sum(loss)/dldy.shape[0]


class NeuralNetwork:
    """Основной класс"""
    def __init__(self, modules):
        self.layers = modules
        
    def forward(self, x):
        """
        Прямой проход
        :param x: входные данные
        """
        current_value = x
        for i in range(len(self.layers)):
            current_value = self.layers[i].forward(current_value)
        return current_value

    def backward(self, y):
        """
        Обратный проход
        :param y: правильная разметка
        """
        dldy = y
        for i in range(len(self.layers)):
            dldy = self.layers[len(self.layers)-i-1].backward(dldy)
        return dldy

    def step(self, learning_rate):
        """Шаг обучения"""
        for i in range(len(self.layers)-1):
            self.layers[i].step(learning_rate)

    def get_loss(self, y, numb):
        """Расчет ошибки"""
        return self.layers[len(self.layers)-1].get_loss(y, numb)
