import matplotlib.pyplot as plt
import random
import math
import numpy as np
from functools import partial
from IPython.display import Image



print('Data Science')
print("\n Data Science é uma area relativamente nova de estudos, mas que se ")
print("aproveita de  varias outras mais antigas, como a Matematica, Negocios")
print(" e Programacao. Nesse conjunto de talks vamos nos focar nos dois ")
print("primeiros pilares Sobre programacao, usaremos python que e")
print("razoavelmente facil de entender e deixaremos as partes mais complicadas")
print("como melhorar a eficiencia dos codigo deixaremos para uma proxima")
print("oportunidade. O escopo aqui e ter um bom overview sobre uma gama ")
print("de tecnicas e com isso poder nao so entender como enxergar")
print("possibilidades em seus proprios negocios. \n")

#####################################################################
print("Nivelando Matematica Vetorial Basica")


print("vetores")

v1 = [1, 2]
v2 = [2, 1]

plot_arrows(v1, v2)


def vector_subtract(v, w):
    return [v_i - w_i
            for v_i, w_i in zip(v, w)]


subtract = (vector_subtract(v1, v2))

print(subtract)

plot_subtract(v1, v2, subtract)


def scalar_multiply(c, v):
    return [c * v_i for v_i in v]


print(scalar_multiply(2, v1))

plot_scalar(2, v1)


def dot(v, w):
    return sum(v_i * w_i
               for v_i, w_i in zip(v, w))


print(" 2*1 + 1*2 ")
print(dot(v1, v2))


def sum_of_squares(v):
    return dot(v, v)


print("1*1 + 2*2")
print(sum_of_squares(v1))


def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))


print("-1*-1 + 1*1")
print(squared_distance(v1, v2))


def distance(v, w):
    return math.sqrt(squared_distance(v, w))


print("Raiz(2), é a diagonal de um quadrado de lado 1")
print(distance(v1, v2))


def sum_of_squares(v):
    return sum(v_i ** 2 for v_i in v)


print(sum_of_squares(v1))


def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h


print("A ideia de Limite de uma funcao quando o intervalo h tende a zero")


def derive(f, a, h=0.001, epsilon=1e-7):
    f1 = (f(a + h) - f(a)) / h
    while True:  # DO-WHILE
        h /= 2.
        f2 = (f(a + h) - f(a)) / h
        diff = abs(f2 - f1)
        f1 = f2
        if diff < epsilon:
            break
    return f2


print("∂f/∂x(x, y, z) = (f(x+epsilon,y,z) - f(x-epsilon, y, z))/(epsilon * 2)")

print("derivatives in x=0")
print("x^2: \t\t %.6f" % derive(lambda x: x**2, 0))
print("x:\t\t %.6f" % derive(lambda x: x, 0))
print("(x-1)^2:\t %.6f" % derive(lambda x: (x - 1)**2, 0))

print("\n\nReal values:")
print(derive(lambda x: x**2, 0))
print(derive(lambda x: x, 0))
print(derive(lambda x: (x - 1)**2, 0))


#####################################################################
print("Nivelando Estatistica e Probabilidade Basica")
print("\n")
print("Tendencias Centrais: Média, Mediana e Moda")


def mean(x):
    return sum(x) / len(x)


print("Dispersão")


def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]
# a ideia aqui, é quanto esse dados estao distantes da media, ou seja, podemos
# ter medias iguais mas dispersoes bastante diferentes.


def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)
# Como a soma de de_mean pode se anular com dados + e -, calculamos a var.
# o n-1 vem da variancia amostral que perde um grau de liberdade.


print("Correlacao")


def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)
# Covariancia nos diz se a tendencia das variancoes estao caminhando juntas.
# A correlacao da variavel com ela mesmo é sua variancia.


def standard_deviation(x):
    return math.sqrt(variance(x))
# Como elevamos os desvios ao quadrado para que se tornem positivos e nao se
# cancelem, precisamos voltar para a escala original, para isso tiramos a raiz
# da variancia e a chamamos de desvio padrao.


def correlation():
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x != 0 and stedv_y != 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0
# A ideia da correlacao é parecida com a covariance, ou seja, como a
# variancia dos
# dados se comporta uma relacao a outra. Com a diferença de
# que fazemos uma especie
# de ponderacao pelo seus desvios padrao, ou seja, se a cov
#  é alta mas seu stdev
# é alto, nao teremos uma corr tao alta quanto com um desvio baixo.


print("Paradoxo de Simpson")
print("Correlacao e Causalidade")
print("Nivelando Probabilidade")
print("Dependencia e Independencia")
print("Probabilidade Condicional")
print("Teorema de Bayes")
print("Variaveis Aleatorias")
print("Distribuicoes Continuas")


def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0


def uniform_cdf(x):
    if x < 0:
        return 0  # uniform random is never less than 0
    elif x < 1:
        return x  # e.g. P(X <= 0.4) = 0.4
    else:
        return 1  # uniform random is always less than


def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

normal_pdf_graph()


print("Quando a media=0 e o stdev=1 chamamos essa curva de normal padrao.")


def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(3) / sigma)) / 2

normal_cdf_graph()

#####################################################################
print("Regressao Linear Simples")



def predict(alpha, beta, x_i):
    return beta * x_i + alpha


def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)


def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))


def least_squares_fit(x, y):
    beta = correlation(x, y) * stantard_deviation(y) / stantard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def total_sum_of_squares(y):
    return sum(v**2 for v in de_mean(y))


def r_squared(alpha, beta, x, y):
    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) /
                  total_sum_of_squares(y))


#####################################################################
print("Regressao Linear Multipla")


def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)


#####################################################################
print("Gradiente Descendente")

x = range(-10, 10)
plt.title("Actual Derivates vs Estimates")
plt.plot(x, map(derivative, x), 'rx', label='Actual')  # red x
plt.plot(x, map(derivative_estimate, x), 'b+', label='Estimate')  # blue +
plt.legend(loc=9)
plt.show()

# compute the ith partial difference quotient of f at v


def partial_difference_quotient(f, v, i, h):
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
            for i, _ in enumerate(v)]

# using the gradient


def step(v, direction, step_size):
    # move step_size in the direction from v
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]


def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


# pick a random starting point
v = [random.randint(-10, 10) for i in range(3)]

tolerance = 0.0000001

while True:
    gradient = sum_of_squares_gradient(v)
    next_v = step(v, gradient, -0.01)
    if distance(next_v, v) < tolerance:
        break
    v = next_v


def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')  # this means "infinity" in python
        return safe_f


step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]


def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    theta = theta_0
    target_fn = safe(target_fn)
    value = target_fn(theta)

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value


print("Indo Além! Gradiente Descendente Estocastico")


def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]


def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = zip(x, y)
    theta = theta_0  # initial guess
    alpha = alpha_0  # initial step size
    min_theta, min_value = None, float("inf")  # infinity
    iteractions_with_no_improvemnt = 0

    while iteractions_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            min_theta, min_value = theta, value
            iteractions_with_no_improvement = 0
            alpha = alpha_0
        else:
            iteractions_with_no_improvement += 1
            alpha *= 0.9

        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
        return min_theta


###############################################################################
# Funcoes de Suporte


def plot_arrows(v1, v2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver((0, 0), (0, 0), v1, v2, units='xy', scale=1)
    plt.axis('equal')
    plt.xticks(range(-2, 5))
    plt.yticks(range(-2, 5))
    plt.grid()
    return plt.show()


def plot_subtract(v1, v2, v3):
    v0 = [0, 0]
    soa = np.array([v0 + v1, v0 + v2, v2 + v3])
    X, Y, U, V = zip(*soa)
    fig_sub = plt.figure()
    ax = fig_sub.add_subplot(111)
    ax.quiver([0, 0, 2], [0, 0, 1], U, V,
              angles='xy', scale_units='xy', scale=1)
    plt.axis('equal')
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    ax.set_xlim([-2, 5])
    ax.set_ylim([-2, 5])
    plt.grid()
    plt.draw()
    return plt.show()


def plot_scalar(c, v1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.quiver((1, 1), (2, 0), v1, scalar_multiply(c, v1), units='xy', scale=1)
    plt.axis('equal')
    plt.xticks(range(-2, 7))
    plt.yticks(range(-2, 7))
    return plt.show()

def normal_pdf_graph():
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0,sigma=1')
    plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0,sigma=2')
    plt.plot(xs, [normal_pdf(x, sigma=0.5)
                  for x in xs], ':', label='mu=0,sigma=0.5')
    plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.', label='mu=-1,sigma=1')
    plt.legend()
    plt.title("Various Normal pdfs")
    return plt.show()

def normal_cdf_graph():
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-', label='mu=0,sigma=1')
    plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label='mu=0,sigma=2')
    plt.plot(xs, [normal_cdf(x, sigma=0.5)
                  for x in xs], ':', label='mu=0,sigma=0.5')
    plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.', label='mu=-1,sigma=1')
    plt.legend(loc=4)  # bottom right
    plt.title("Diferentes cdfs Normais")
    return plt.show()
