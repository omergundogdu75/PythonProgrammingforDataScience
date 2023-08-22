########################################
# Veri yapıları (Data structures)
#######################################

# Veri yapılarına giriş ve özet
#######################################
# Sayılar (Numbers): int, float, complex
x = 46
type(x)

y = 10.2
type(y)

z = 2 + 2j
type(z)

# Karekter dizileri(Strings): str
j = "Message"
type(j)

# Boolean (TRUE-FALSE): bool
True
False
type(True)
type(5 == 2)
3 == 2
5 == 5

# Liste (List)
x = ["btc", "eth", "avax"]
type(x)

# Sözlük (Dictionary)
x = {"name": "Peter", "Age": 35}
type(x)

# Demet (Tupple)
x = ("python", "java", "c++")
type(x)
# list, tuple ,set ve dictionary veri yapıları aynı zamanda Python Collections (arrays) olarakda geçmektedir.

# Set
x = {"name", "Age"}
type(x)

####################################################################
# Sayılar (Numbers): int, float, complex
####################################################################

a = 5
b = 10.5
a * 3
a / 7
a * b / 10
a ** 2
# pep8 py geliştirici kod yazma biçimidir.

####################
# Tipleri değiştirmek
####################
int(b)
float(a)

int(a * b / 10)

c = a * b / 10
c
int(c)

####################################################################
# Karekter dizileri(Strings): str
####################################################################

print("John")
print('John')

"John"

name = "ÖMER"
name = 'ÖMER'

## çok satırlı karekter dizileri
long_str = """Bu satır çok satırlı
String değişkenidir.
ÖMER GÜNDOĞDU"""

# karekter dizilerinde eleman seçme işlemi
long_str
long_str[0]
long_str[3]
long_str[2]
long_str[1]

# karekter dizilerinde slice işlemi
long_str[0:2]
long_str[0:10]

# String içerisinde karekter sorgulamak
"bu" in long_str
"Bu" in long_str

####################################################################
# String methodları(Karekter dizisi)
####################################################################
dir(str)  # dir burada veri yapısı için kullanabiliecek methodları göstermektedir.

# len
len(long_str)  # Stringin uzunluğunu belirler
type(len)
len("ÖMER GÜNDOĞDU")
len("miuul")
## bir function class yapısı içersinde ise method, değilse fonksiyondur

# upper() lower()  büyük küçük deönüşümleri
"ömer".upper()
"GÜNDOĞDU".lower()

# type(upper)

# Replace: karekter değiştirmek
greetings = " Hello ÖMER"
greetings.replace("Hello", "Hi")

# Split: böler / ön tanımlı değeri boşluktur
"Hello AI era!".split()

# strip: kırpar
" ogogoog".strip()
"ogogoogo".strip("o")

# capitalize: ilk harfi büyük yapar.
"ömer".capitalize()

dir("foo")

# .... ile  başlar mı
"foo".startswith("f")

####################################################################
# Boolean (TRUE-FALSE): bool
####################################################################


####################################################################
# Liste (List)
####################################################################
# değiştirilebilir, sıralıdır, index işlemleri yapılabilir, kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "c", "d"]

nt_nam = [1, 2, 3, "a", "b", True,[4, 5, 6]]

nt_nam[0]
nt_nam[6][2]
type(nt_nam[6][2])
nt_nam[0:2]

#Liste methodları
dir(nt_nam)

###############################################
#len: builtin python fonksiyonu, boyut bilgisi.
###############################################
len(nt_nam)
len(notes)

# append: eleman ekler,
notes
notes.append(2)

#pop: indexe göre eleman siler.
notes.pop(0)
notes

# insert: indexe ekler
notes.insert(1,10)
notes

####################################################################
# Sözlük (Dictionary)
####################################################################
#değiştirilebilir, sırasız(3.7den sonra), kapsayıcı
#key-value
dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

dictionary["REG"]

dictionary = {"REG": ["RMS",10],
              "LOG": ["MSE",20],
              "CART": ["SSE",30]}

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30}

dictionary["CART"][1]

#KEY DEĞERİNDE SORGULAR
"REG" in dictionary
"RMS" in dictionary

#key değerine göre sorgular
dictionary.get("REG")

#VALUE DEĞİŞTİRMEK
dictionary["REG"] = ["YSA",10]


#Tüm key ve valuelere erişmek liste
dictionary.keys()
dictionary.values()

#tüm key valueler tupple birlikte alır
dictionary.items()

#key value değerini güncellemek
dictionary.update({"REG": 11})
dictionary

#yeni key value eklemek
dictionary.update({"RF":20})
dictionary

####################################################################
# Demet (Tupple)
####################################################################
#değiştirilemez, sıralıdır, kapsayıcıdır

t = ("john", "mark", 1,2)
type(t)
t[0:3]
t[0] = 123 ##değiştirilemez - değiştirmek için ilk listeye çevirmen gerek

####################################################################
# Set
####################################################################
#değiştirilebilir, sırasız, eşsiz, kapsayıcı
#İKİ KÜME FARKI: difference()

set1= set([1,2,3,4])
set2= set([1,4,5,6])

#sett1 de olup set2de olmayanlar
set1.difference(set2)
#set2 de olup set1 de olmayanlar
set2.difference(set1)

#symmetric_difference(): iki kümedede birbirinde olmayanlar
set1.symmetric_difference(set2)

# intersection() iki küme keşisimi:
set1.intersection(set2)
set1 & set2
set1 - set2
set2 - set1

#union(): iki küme birleşimi
set1.union(set2)
set2.union(set1)

# iki küme kesişimi boş mu değilmi? isdisjoint()
set1.isdisjoint(set2)
set2.isdisjoint(set1)

# issubject(): Birküme diğerinin alt kümesimi
set1.issubset(set2)
set2.issubset(set1)

# issubsuperset() bir küme diğerini kapsiyor mu

set1.issuperset(set2)
set2.issuperset(set1)
