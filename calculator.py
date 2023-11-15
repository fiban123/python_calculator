import decimal
from decimal import Decimal as d
import pygame
from sys import argv
from time import perf_counter_ns as t

# --- to do ---:
# fix calculate() function
# fix arccosh and arctanh and arctan
# add support for complex numbers everywhere (except integral ranges)
# add fourier and laplace transform
# add matrix calculations
# add indefinite integrals
# add summations and products


# constants


# global variables
pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679  # first 100 digits of pi
e = 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274  # first 100 digits of e
root15 = 3.8729833462074168851792653997823996108329217052915908265875737661134830919369790335192873768586735179  # first 100 digits or root15

decimal.getcontext().prec = 20


def argparse(somefunction):
    somefunction = somefunction.replace('[', '¬')
    somefunction = somefunction.replace(',', '¬')
    somefunction = somefunction.replace(']', '')
    somefunction = somefunction.split('¬')[0:]
    return list(somefunction[1:])


def parse(somefunction):
    somefunction = str(somefunction)
    somefunction = somefunction.split('.')
    value = (somefunction[1:len(somefunction)])
    value = '.'.join(value)
    if not 'j' in str(somefunction):
        return float(value)
    else:
        try:
            value = complex(value)
            return complex(value)
        except Exception as e:
            return 'Invalid syntax'


# from here everything is working and compatible with complex numbers

def polyLogarithm(function):
    args = argparse(function)
    x = complex(args[0])
    base = complex(args[1])
    res = x
    for i in range(1, 750):
        res += x ** i / i ** base
    return res


def factorial(function):
    x = parse(function)
    if int(x) == x:
        # regular factorial
        if x.real < 0:
            return 'result does not converge'
        else:
            res = 1
            for i in range(int(x)):
                res *= x - i
    elif x.real > 0:
        res = 0
        for i in range(1, int(x * 1000)):
            # 3.3264089
            res += e ** -i * i ** x
    else:
        return 'result does not converge'

    return res


def summation(function):
    args = argparse(function)
    start = int(args[0])
    end = int(args[1])
    function = args[2]
    res = 0
    cFunc = function
    for i in range(start, end + 1):
        cFunc = function.replace('x', str(i))
        cRes = float(evaluate(cFunc))
        res += cRes
    return res


def product(function):
    args = argparse(function)
    start = int(args[0])
    end = int(args[1])
    function = args[2]
    res = 1
    cFunc = function
    for i in range(start, end + 1):
        cFunc = function.replace('x', str(i))
        cRes = float(evaluate(cFunc))
        res *= cRes
    return res


def hyperbolic_arccosine(function):
    x = parse(function)
    if x.imag == 0:
        root1 = root(f'root,2,{x + 1}')


def hyperbolic_arcsecant(function):
    x = parse(function)
    return hyperbolic_arccosine(f'arccosh.{1 / x}')


def hyperbolic_arccotangent(function):
    x = parse(function)
    root1 = hyperbolic_arctangent(f'arctanh.{1 / x}')
    if x.real > 1 and x.real > -1:
        return root1
    else:
        if x.imag > 0:
            return (-root1.real) + (root1.imag * 1j)
        else:
            return (-root1.real) - (root1.imag * 1j)


def hyperbolic_arctangent(function):
    x = parse(function)
    if x.imag == 0:
        if x.real in range(-1, 1):
            pass
        else:
            root1 = 1 / root(f'root[2,{1 - (x ** 2)}]')
            hyperbolic_arccosine(f'arccosh.{root1}')


def hyperbolic_arccosecant(function):
    x = parse(function)
    root1 = root(f'root[2,{(1 / (x ** 2)) + 1}]')
    return natural_logarithm(f'ln.{(1 / x) + root1}')


def hyperbolic_arcsine(function):
    x = parse(function)
    if x.imag == 0:
        root1 = root(f'root[2,{x ** 2 + 1}]')
        return natural_logarithm(f'ln.{x + root1}')
    else:
        root1 = 1 + (x ** 2)
        return natural_logarithm(f'ln.{root1}')


def hyperbolic_cosecant(function):
    x = parse(function)
    return 1 / (hyperbolic_sine(f'sinh.{x}'))


def hyperbolic_secant(function):
    x = parse(function)
    return 1 / (hyperbolic_cosine(f'coth.{x}'))


def hyperbolic_cotangent(function):
    x = parse(function)
    return 1 / (hyperbolic_tangent(f'tanh.{x}'))


def hyperbolic_tangent(function):
    x = parse(function)
    return hyperbolic_sine(f'sinh.{x}') / hyperbolic_cosine(f'cosh.{x}')


def hyperbolic_sine(function):
    x = parse(function)
    return ((e ** x) - (e ** -x)) / 2


def hyperbolic_cosine(function):
    x = parse(function)
    return ((e ** x) + (e ** -x)) / 2


def arccotangent(function):
    x = parse(function)
    return arctangent(f'arctan.{1 / x}')


def arccosecant(function):
    x = parse(function)
    if x.imag == 0:
        return arcsine(f'arcsin.{1 / x}').real
    else:
        return -(arcsine(f'arcsin.{1 / x}'))


def arcsecant(function):
    x = parse(function)
    return arccosine(f'arccos.{1 / x}')


def arctangent(function):
    x = parse(function)
    print(x)
    root1 = natural_logarithm(f'ln.{1 - (x * 1j)}')
    root2 = natural_logarithm(f'ln.{1 + (x * 1j)}')
    root3 = ((root1 - root2) * 1j) / 2
    return root3


def arccosine(function):
    x = parse(function)
    if x.imag == 0:
        if x.real < 1 and x.real > -1:
            return pi / 2 + (hyperbolic_arcsine(f'arcsinh.{x * 1j}') * 1j)
        else:
            return (pi / 2 - (-arcsine(f'arcsin.{x}'))) - pi
    else:
        return pi / 2 + (hyperbolic_arcsine(f'arcsinh.{x * 1j}') * 1j)


def arcsine(function):
    x = parse(function)
    if x.imag == 0:
        root1 = (x + root(f'root[2,{1 - (x ** 2)}]')) * 1j
        return natural_logarithm(f'ln.{root1}')
    else:
        if x.real == 0 and x.imag != 0:
            return (hyperbolic_arcsine(f'arcsinh.{x * 1j}')) * 1j
        else:
            root1 = hyperbolic_arcsine(f'arcsinh.{-(x * 1j)}')


def cotangent(function):
    x = parse(function)
    return (cosine(f'cos.{x}')) / (sine(f'sin.{x}'))


def cosecant(function):
    x = parse(function)
    return 1 / (sine(f'sin.{x}'))


def secant(function):
    x = parse(function)
    return 1 / (cosine(f'cos.{x}'))


def tangent(function):
    x = parse(function)
    return (sine(f'sin.{x}')) / (cosine(f'cos.{x}'))


def root(function):
    function = argparse(function)
    base = complex(function[0])
    value = complex(function[1])
    if value.imag == 0:
        res = value ** (1 / base)
        if (value.real) > 0:
            return res
        else:
            return res * 1j
    else:
        if base == 2:
            root1 = abs(root(f'root[2,{(abs(value) + value.real) / 2}]'))
            root2 = value.imag * 1j / abs(value.imag)
            root3 = root(f'root[2,{(abs(value) - value.real) / 2}]').imag
            return root1 + (root2 * root3)


def natural_logarithm(function):
    x = parse(function)
    result = logarithm(f'log[{e},{x}]')
    print((end - start) / 1000000000)
    return result


def logarithm(function: str):
    function = argparse(function)
    base = function[0]
    result = function[1]
    base = complex(base.replace(' ', ''))
    result = complex(result.replace(' ', ''))

    # algorithm for complex and regular logs
    range2: float = max([abs(result.real), abs(result.imag)])
    rstart: float = -range2
    rend: float = range2
    cstart: float = -range2
    cend: float = range2
    resultlist: list = []
    d_resultlist: list = []
    realv: float = rstart
    complexv: float = cstart
    step: float = 50
    prec: int = 50
    for i in range(prec):
        # approximate
        while complexv < cend:
            realv = rstart
            while realv < rend:
                difference = abs((base ** (realv * 0.1 + (complexv * 0.01 * 1j))) - result)
                resultlist.append(difference)
                d_resultlist.append([realv, complexv])
                realv += step
            complexv += step
        # update values
        smallest_value = min(resultlist)
        smallest_value_index = resultlist.index(smallest_value)
        realpart = d_resultlist[smallest_value_index][0]
        complexpart = d_resultlist[smallest_value_index][1]
        range2 = step * 2
        step /= 2
        rstart = realpart - range2
        rend = realpart + range2
        cstart = complexpart - range2
        cend = complexpart + range2
        complexv = cstart
        resultlist = []
        d_resultlist = []
        realv = rstart
    return realpart * 0.1 + (complexpart * 0.01 * 1j)


def sine(function):
    x = parse(function)
    return (-e ** ((-x) * 1j) + (e ** (x * 1j))) / 2j


def cosine(function):
    x = parse(function)
    return sine(f'sin.{(pi / 2) - x}')


def indefinite_integral(funtion):
    pass


def definite_integral(function):
    function = str(function)
    function = function.split(',')
    a = function[1]
    b = function[2]
    calculationbase = function[3]
    exactness = 5000
    a = int(a) * exactness
    b = int(b) * exactness
    ans = 0
    while a != b:
        calculation = calculationbase.replace('x', str(a / exactness))
        calculation = calculation.replace(';', ' ')
        value = evaluate(calculation)
        ans += complex(value)
        if a > b:
            a -= 1
        elif a < b:
            a += 1
    return ans / exactness


def exponentiation(a, b):
    return complex(a) ** complex(b)


def division(a, b):
    return complex(a) / complex(b)


def multiplication(a, b):
    return complex(a) * complex(b)


def substraction(a, b):
    return complex(a) - complex(b)


def addition(a, b):
    return complex(a) + complex(b)


FUNCTIONS = [
    'acsch', 'acoth', 'asech', 'atanh', 'acosh', 'asinh', 'csch', 'coth', 'sech', 'tanh',
    'cosh', 'sinh', 'acsc', 'acot', 'asec', 'atan', 'acos', 'asin', 'csc', 'cot', 'sec',
    'tan', 'cos', 'sin', 'log', 'ln', 'root', 'sum', 'prod', 'di', 'integral', 'fact', 'Li'
]
BASICFUNCTIONS = ['^', '/', '*', '+', '-']
FUNCTIONFUNCTIONS = [
    hyperbolic_arccosecant, hyperbolic_arccotangent, hyperbolic_arcsecant, hyperbolic_arctangent,
    hyperbolic_arccosine, hyperbolic_arcsine, hyperbolic_cosecant, hyperbolic_cotangent, hyperbolic_secant,
    hyperbolic_tangent, hyperbolic_cosine, hyperbolic_sine, arccosecant, arccotangent, arcsecant, arctangent,
    arccosine, arcsine, cosecant, cotangent, secant, tangent, cosine, sine, logarithm, natural_logarithm,
    root, summation, product, definite_integral, indefinite_integral, factorial, polyLogarithm
]
BASICFUNCTIONFUNCTIONS = [
    exponentiation, division, multiplication, addition, substraction
]


def calculate(expression):
    result2 = expression
    values = []
    # splitting expression
    expression = str(expression)
    split = True
    pSpaceIndex = -1
    values = []
    for i, char in enumerate(expression):
        if char == '[':
            split = False
        elif char == ']':
            split = True
        if split:
            if char == ' ':
                values.append(expression[pSpaceIndex + 1:i])
                pSpaceIndex = i
    values.append(expression[pSpaceIndex + 1:])
    stoploop = False
    # calculate expression
    while not stoploop:
        for i in range(len(values)):
            updatevalues = False
            char = str(values[i])
            # find numbers
            try:
                n1 = values[i - 1]
                n2 = values[i + 1]
                n1 = complex(n1)
                n2 = complex(n2)
            except:
                pass
            for i2, function in enumerate(FUNCTIONS):
                if char[0:len(function)] == function:
                    result2 = FUNCTIONFUNCTIONS[i2](char)
                    updatevalues = True
            for i2, basicfunction in enumerate(BASICFUNCTIONS):
                if char[0:len(basicfunction)] == basicfunction:
                    n1 = complex(values[i - 1])
                    n2 = complex(values[i + 1])
                    result2 = BASICFUNCTIONFUNCTIONS[i2](n1, n2)
                    updatevalues = True

            # update expression
            if updatevalues:
                del values[i - 1:i + 2]
                values.insert(i - 1, result2)
                break
            if len(values) == 1 or len(values) < 1:
                stoploop = True
    return result2


def evaluate(inpt):
    result = ''
    inpt2 = ' ' + inpt + ' '
    inpt = str(inpt)
    startindex = 0
    endindex = 0
    inparenthesis = ''
    if ' e ' in inpt2:
        inpt = inpt2.replace(' e ', f' {e} ')[1:-1]
    if ' pi ' in inpt2:
        inpt = inpt2.replace(' pi ', f' {pi} ')[1:-1]
    while True:
        if '(' and ')' in inpt:
            for i in range(len(inpt)):
                char = inpt[i]
                if char == '(':
                    startindex = i
                elif char == ')':
                    endindex = i
                    break
            inparenthesis = inpt[startindex + 1:endindex]
            ebp = inpt[0:startindex]  # everything before the stuf in parenthesis
            eap = inpt[endindex + 1:len(inpt)]  # everything after
            evaluation = str(evaluate(inparenthesis))
            inpt = ebp + evaluation + eap
        else:
            result = calculate(inpt)
            break
    try:
        result = complex(result)
    except:
        return result
    # make sure that this function doesnt return any rounding errors
    if result.imag > -1e-11 and result.imag < 1e-11:
        return result.real
    elif result.real > -1e-11 and result.real < 1e-11:
        return result.imag * 1j
    else:
        return result


#####-- G U I --#####


# essantial setup
pygame.init()
pygame.font.init()
path0 = argv[0]
path1 = len(path0.split('/')[-1])
path = path0[0:-path1] + "assets/"

# constants
RES = [600, 750]
BGCOLOR = '#00203F'
TEXTCOLOR = '#AEEFD1'
CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.'
FPS = 127
SCROLLLIMITN = -450
SCROLLLIMITP = 0
SHIFT = 1073742049
ALT_GR = 1073742054
KEYNUMS = [1073742049, 1073742054, 51, 49, 252, 168, 56, 57, 94, 55, 8, 13, 45]
KEYCOMBS = [
    [55, SHIFT], [252, ALT_GR], [168, ALT_GR], [51, SHIFT], [49, SHIFT],
    [56, SHIFT], [57, SHIFT], [8, SHIFT]
]
KEYCOMBSVALS = [' / ', '[', ']', ' * ', ' + ', '(', ')', '?CLS']
SUBSTRINGLIST = [
    'acsch', 'acoth', 'asech', 'atanh', 'acosh', 'asinh', 'csch', 'coth', 'sech', 'tanh',
    'cosh', 'sinh', 'acsc', 'acot', 'asec', 'atan', 'acos', 'asin', 'csc', 'cot', 'sec',
    'tan', 'cos', 'sin', 'ln', 'log', 'root', 'di'
]
REPLACELIST = SUBSTRINGLIST
SINGLEKEYS = [45, 94, 8, 13]
SINGLEKEYREPLACELIST = ['-', '^', '?BCK', '=']
SEPERATIONINDICATORS = ['0', ' + ', 'di[', '(', 'sin.']
# variables
keypressed = {
    1073742049: False, 1073742054: False, 51: False, 49: False, 252: False,
    168: False, 56: False, 57: False, 94: False, 55: False, 8: False, 13: False,
    45: False
}
run = True
buttons = []
screen = pygame.display.set_mode(RES)
pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
clock = pygame.time.Clock()
string = []
new_press = False
p_new_press = False
pos = 0
pos0 = 0
ppos = 0
end = 0
start = 0
calibri40 = pygame.font.SysFont("Calibri", 40)
calibri30 = pygame.font.SysFont("Calibri", 30)
calibri25 = pygame.font.SysFont("Calibri", 25)
mmt5 = False
entryRect = pygame.Rect([0, 0], [600, 145])
enterKeys = True
addBasicKey = True

# images
images = []
imagenames = ['plus', 'minus', 'times', 'over', 'pow', 'sqrt', 'root', 'logarithm', 'natural_logarithm',
              'polyLogarithm', 'pi', 'e', 'factorial',
              'opening_bracket', 'closing_bracket', 'dot', 'comma', 'opening_curly_bracket', 'closing_curly_bracket',
              'clear', 'backspace', 'equals', 'trig/sin',
              'trig/cos', 'trig/tan', 'trig/sec', 'trig/csc', 'trig/cot', 'trig/asin', 'trig/acos', 'trig/asec',
              'trig/acsc', 'trig/acot',
              'trig/sinh', 'trig/cosh', 'trig/sech', 'trig/csch', 'trig/coth', 'trig/asinh', 'trig/acosh', 'trig/atanh',
              'trig/asech',
              'trig/acsch', 'trig/acoth', 'cal/definite_integral', 'cal/integral', 'cal/summation', 'cal/product'
              ]
for i in range(10):
    images.append(pygame.image.load(f'{path}{i}.png').convert_alpha())
for imagename in imagenames:
    images.append(pygame.image.load(f'{path}{imagename}.png').convert_alpha())
functions = [str(i) for i in range(10)]
functions += ' + ', ' - ', ' * ', ' / ', ' ^ ', 'root[2,', 'root[', 'log[', 'ln.', 'Li[', 'pi', 'e', 'fact.', '(', ')', '.', ',', '[', ']', '?CLS', '?BCK', '=', 'sin.', 'cos.', 'tan.', 'sec', 'csc'
functions += 'cot.', 'arcsin.', 'arccos.', 'arcsec.', 'arccsc.', 'arccot.', 'sinh.', 'cosh.', 'tanh.', 'sech.', 'csch.', 'coth.', 'arcsinh.', 'arccosh.', 'arcsech.',
functions += 'arccsch.', 'arccoth.', 'di[', 'integral[', 'sum[', 'prod['
seperationMessages = ['Numbers:', 'Basic Math:', 'Calculator Specific:', 'Trigonometric functions:', 'Calculus:']
entryimg = pygame.image.load(f'{path}entry.png').convert_alpha()


# classes
class btn:
    def __init__(self, x_pos, y_pos, image, width, height):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.image = image
        self.width = width
        self.height = height

    def draw(self):
        screen.blit(self.image, [self.x_pos, self.y_pos])

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        button_rect = pygame.rect.Rect([self.x_pos, self.y_pos], [self.width, self.height])
        if button_rect.collidepoint(mouse_pos):
            return True
        else:
            return False


# functions
def makestring(n):
    global string
    add = True
    if n == '=':
        string = list(getOutput(string))
        add = False
    elif n == '?CLS':
        string = []
        add = False
    elif n == '?BCK':
        try:
            del string[-1]
        except:
            pass
        add = False
    if add:
        for char in n:
            string.append(char)


def entry():
    global string
    string2 = ''.join(string)
    # display intersectiob rec
    pygame.draw.rect(screen, BGCOLOR, entryRect)
    # display entry image
    screen.blit(entryimg, [10, 10])
    # display string
    if len(string2) < 27:
        renderedString = [calibri40.render(string2, 1, BGCOLOR)]
    elif len(string2) < 80:
        renderedString = [calibri25.render(string2[0:40], 1, BGCOLOR), calibri25.render(string2[40:], 1, BGCOLOR)]
    else:
        renderedString = [calibri25.render(string2[0:40], 1, BGCOLOR), calibri25.render(string2[40:80], 1, BGCOLOR),
                          calibri25.render(string2[80:], 1, BGCOLOR)]
    pos = [20, 57]
    for string3 in renderedString:
        screen.blit(string3, pos)
        pos[1] += 26


def scroll(ypos, new_press, p_new_press, btnpress):
    stepsize = 75
    y2 = ypos + 180
    x2 = 0
    sepcount = 0
    xcount = 0
    for i, image in enumerate(images):
        func = functions[i]
        if func in SEPERATIONINDICATORS:
            if functions[i] != '0':
                y2 += stepsize
            x2 = 0
            renderedString = calibri30.render(seperationMessages[sepcount], 1, TEXTCOLOR)
            screen.blit(renderedString, [x2, y2])
            y2 += 50
            sepcount += 1
            xcount = 0
        button = btn(x2, y2, image, 64, 64)
        button.draw()
        if button.check_click() and not new_press and p_new_press and btnpress:
            makestring(functions[i])
        if xcount == 7:
            y2 += stepsize
            x2 = 0
            xcount = 0
        else:
            x2 += stepsize
            xcount += 1


def checkpress(keys):
    keys = dict(keys)
    addedKey = False
    if enterKeys:
        for i, item in enumerate(KEYCOMBS):
            if keys[item[0]] and keys[item[1]] and enterKeys:
                makestring(KEYCOMBSVALS[i])
                addedKey = True
        if not addedKey:
            for i, comb in enumerate(SINGLEKEYS):
                if keys[comb]:
                    makestring(SINGLEKEYREPLACELIST[i])
                    addedKey = True
    return addedKey


def replaceSubStrings(list2, substringlist, replacelist):
    finalstring = list2
    while True:
        for i, substring in enumerate(substringlist):
            index = ''.join(finalstring).find(substring)
            if index != -1:
                finalstring[index] = replacelist[i]
                del finalstring[index + 1:index + len(replacelist[i])]
                break
        break
    return finalstring


def getOutput(input):
    input = ''.join(replaceSubStrings(input, SUBSTRINGLIST, REPLACELIST))
    start = t()
    output = str(evaluate(input))
    end = t()
    print(f'time: {(end - start) / 1000000000} seconds')
    return output


# main loop:
while run:
    addBasicKey = True
    screen.fill(BGCOLOR)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            if event.key in KEYNUMS:
                keypressed[event.key] = True
                if checkpress(keypressed):
                    addBasicKey = False
            if addBasicKey:
                try:
                    if pygame.key.name(event.key) in CHARS and enterKeys:
                        makestring(pygame.key.name(event.key))
                    elif event.key == 32:
                        makestring(' ')
                except:
                    pass
        elif event.type == pygame.KEYUP:
            if event.key in KEYNUMS:
                keypressed[event.key] = False

    # main loop code
    # vars
    isLPressed = pygame.mouse.get_pressed()[0]
    cursorpos = pygame.mouse.get_pos()

    # check if left mouse button has been released
    if isLPressed:
        new_press = True
        if start == -1:
            start = cursorpos[1]
            pdiff = 0
        end = cursorpos[1]
        # scroll
        difference = end - start
        pos0 += difference
        ddiff = difference - pdiff
        pos += ddiff
        pdiff = difference
        # make sure you can type in the little box
        # if you are on the box
        if (p_new_press != new_press) and entryRect.collidepoint(cursorpos):
            if enterKeys == False:
                enterKeys = True
            else:
                enterKeys = False
    else:
        start = -1
        new_press = False
    if pos < SCROLLLIMITN:
        pos = SCROLLLIMITN
    elif pos > SCROLLLIMITP:
        pos = SCROLLLIMITP
    scroll(pos, new_press, p_new_press, mmt5)
    entry()
    # previous variables
    p_new_press = new_press
    mmt5 = start in range(end - 5, end + 5)
    # essential updating
    pygame.display.update()
    clock.tick(FPS)
# exit
pygame.display.quit()
pygame.quit()
