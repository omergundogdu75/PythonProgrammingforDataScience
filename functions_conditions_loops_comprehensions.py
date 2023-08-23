############################################################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
############################################################

# - FONKSİYONLAR - FUNCTIONS
# - KOŞULLAR - CONDITIONS
# - DÖNGÜLER - LOOPS
# - COMPREHENSIONS

############################################################
# FONKSIYON OKUR YAZARLIĞI
############################################################

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

list_store = [1, 2]  ## global etki alanı
type(list_store)


def add_element(a, b):
    c = a * b
    list_store.append(c)  # local etki alanı global etki alanını değiştirir
    print(list_store)


add_element(12, 2)

############################################################
# Koşullar (Conditions)
############################################################
# TRUE FALSE

1 == 2
1 == 1

# if

if 1 == 1:
    print("ÖMER GÜNDOĞDU")

if 1 == 2:
    print("GÜNDOĞDU")

number = 11

if number == 10:
    print("number is 10")

number = 10

number = 20


def number_check(number):
    if number == 10:
        print("number is 10")


number_check(10)


# if - else

def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")


number_check(12)


# elif
def number_check(number):
    if number > 10:
        print("number is greater than 10")
    elif number < 10:
        print("number is less than 10")
    else:
        print("equal to 10")


number_check(2)

#  DÖNGÜLER - LOOPS
###################
# FOR LOOP

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [100, 200, 300, 400]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary * 50 / 100 + salary))


def new_salary(salary, rate):
    return int(salary * rate / 100 + salary);


for salary in salaries:
    if salary >= 300:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))

##Example: girilen string ifadedeki çift indexteki karekterleri  büyük yapmılması

old_string = "miuul"
range(len(old_string))
range(0, 5)

for i in range(len(old_string)):
    print(i)

4 % 2 == 0


def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    return new_string


print(alternating("selam"))

######################################################
# Break & While & Continue
######################################################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

number = 1
while number < 5:
    print(number)
    number += 1

######################################################
# Enumarate: Otomatik index/counter ile for loop
######################################################

students = ["ÖMER", "GÜNDOĞDU", "FATİH", "MEHMET", "SULTAN"]
for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

for index, student in enumerate(students, 1):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

A
B

# Example: Bir fonksiyon yazarak, çift indexe sahip oolan öğrencileri bir listeye,
# Tek indexte yer alan öğrencileri bir başka listeye alınız.
# Fakat bu iki liste tek nir liste olarak return olsuyn

students = ["ÖMER", "GÜNDOĞDU", "FATİH", "MEHMET", "SULTAN"]


def divide_students(students):
    group = [[], []]

    for index, student in enumerate(students):
        if index % 2 == 0:
            group[0].append(student)
        else:
            group[1].append(student)

    print(group)
    return group


st = divide_students(students)
st[0]
st[1]


#####################################################
# Alternatin fonksiyonunun enumarate ile yazılması
###################################################

def alternating_with_enumarete(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating_with_enumarete("ömer gündoğdu merhaba")

#####################################################
# Zip : birbirinden farklı listeleri birlikte değerlendirmeyi sağlar
###################################################
students = ["ÖMER", "GÜNDOĞDU", "FATİH", "MEHMET"]
departments = ["IT", "ENGINEERNING", "ACCOUNT", "CEO"]
ages = [12, 32, 24, 21]

list(zip(students, departments, ages))


#####################################################
# lambda, map , filter, reduce
###################################################

def summer(a, b):
    return a + b


summer(1, 3) * 9

num_sum = lambda a, b: a + b

num_sum(2, 4)

# lambda, da bir fonsiyon oluşturur, kullan at fonksiyon şeklindedir

# map

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))
list(map(lambda x: x * 20 / 100 + x, salaries))
list(map(lambda x: x ** 2 / 100 + x, salaries))

# FILTER
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

#reduce: indirgemek demek
from functools import reduce
list_store = [1, 2, 3, 4]
reduce(lambda a,b: a+b, list_store)
