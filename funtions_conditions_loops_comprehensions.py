####################################################
# Fonksiyonlar
####################################################

# Fonksiyon okuryazarlığı
print("a")
print("a", "b")
print("a", "b", sep="__")


# Fonksiyon tanımlama

def calculate(x):
    """

    :param x:
    :return:
    """
    print(x * 2)


calculate(2)


# DOCSTRING
##################################################
# iki argümanlı/parametreli bir fonksiyon tanımlayalım
def summer(arg1, arg2):
    """
    Sum of two numbers
    Args:
        arg1: int, float
        arg2: int, float

    Returns:
        int, float

    Examples:

    Notes:

    """
    print(arg1 + arg2)


summer(12, 23)

summer(arg2=2, arg1=3)


######################################
# FONKSİYONLARIN STATEMENT/ BODY
#######################################

# def function_name(parameters/arguments):
#   statements(function body)

def say_hi(string):
    print(string)
    print("Hello")
    print("Hi")


say_hi("asd")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)

# Girilen değerleri liste içersinde saklamak
list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(10, 12)


#################################
# Ön tanımlı argümanlar/ Parametreler / default arguments/parameters
#################################

def divide(a, b):
    print(a / b)


divide(1, 2)


def divide2(a, b=2):
    print(a / b)


divide2(4)


def say_hi2(string="Helloooo"):
    print(string)
    print("Hello")
    print("Hi")


say_hi2()


## fonksiyonlar tekrarlanan işler varsa yazılır

def calculate(warm, moistre, charge):
    print((warm + moistre) / charge)


calculate(98, 12, 22)


#####################################################
# RETURN: FONKSİYON ÇIKTILARINI GİRDİ OLARAK KULLANMAK İÇİN KULLANILIR
#####################################################
def calculate2(warm, moistre, charge):
    return (warm + moistre) / charge


calculate2(98, 12, 22) * 2


def calculate3(warm, moistre, charge):
    warm = warm * 2
    moistre = moistre * 2
    charge = charge * 2
    output = (warm + moistre) / charge

    return warm, moistre, charge, output


w, m, c, o = calculate3(98, 12, 22)


# fonksiyon çerisinden fonksiyon çağırmak
#####################################################


def calculate4(warm, moistre, charge):
    return int((warm + moistre) / charge)


calculate4(98, 12, 22)


def stardardization(a, p):
    return a * 10 / 100 * p * p


stardardization(45, 1)


def all_function(varm, moisture, charge, p):
    a = calculate4(varm, moisture, charge)
    b = stardardization(a, p)
    return (b * 10)


all_function(12, 23, 2, 3)

# local ve global değişkenler
#####################################################

list_store = [1, 2] ## global etki alanı
type(list_store)

def add_element(a,b):
    c = a*b
    list_store.append(c) # local etki alanı global etki alanını değiştirir
    print(list_store)

add_element(12,2)