# Görev 1: Verilen değerlerin veri yapılarını inceleyiniz.
x = 8
y = 3.2
z = 8j + 18
a = "Hello world"
b = True
c = 22 < 21
l = [1, 2, 3, 4]
d = {
    "Name": "Jake",
    "Age": 27,
    "Adress": "Downrown"
}
t = ("Machine Learning", "Data Science")
s = {"py", "ml", "ds", "java"}
type(x)
type(y)
type(z)
type(a)
type(b)
type(c)
type(l)
type(d)
type(t)
type(s)

# Görev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
text = "The goal is to turn data into information, and into insight."
text = text.upper()
text = text.replace(".", "").replace(",", "")
text.split(" ")

# Görev3: Verilenlisteyeaşağıdakiadımlarıuygulayınız.
lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
# Adım 1: Verilen listenin eleman sayısına bakınız.
print(len(lst))
# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
print(lst[0], lst[10])
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
lst[0:4]
# Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)
# Adım 5: Yeni bir eleman ekleyiniz.
lst.append("O")
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")

# Görev4: Verilensözlükyapısınaaşağıdakiadımlarıuygulayınız.
dict = {'Christian': ["America", 18],
        'Daisy': ["UK", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]
        }
# Adım 1: Key değerlerine erişiniz.
dict.keys()
# Adım 2: Value'lara erişiniz.
dict.values()
# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict['Daisy'][1] = 13
dict
# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict['Ahmet'] = ['Turkey', 24]
dict
# Adım 5: Antonio'yu dictionary'den siliniz.
dict.pop('Antonio')
dict

# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki
# tek ve çift sayıları ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.

l = [2, 13, 18, 93, 22]
group = [[], []]


def sp_list(lst):
    for i in lst:
        if i % 2 == 0:
            group[0].append(i)
        else:
            group[1].append(i)
    return group[0], group[1]


even_lst, odd_lst = sp_list(l)
print(even_lst)
print(odd_lst)

# Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye
# giren öğrencilerin isimleri bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik
# fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]


def func(lst):
    for index, ogrenci in enumerate(ogrenciler, 1):
        if index > 3:
            print(f"Tıp fakültesi {index - 3}. öğrenci: {ogrenci}")
        else:
            print(f"Mühendislik fakültesi {index}. öğrenci: {ogrenci}")


func(ogrenciler)

# Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir
# dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır.
# Zip kullanarak ders bilgilerini bastırınız.

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

infos = list(zip(ders_kodu, kredi, kontenjan))

for info in infos:
    print(f"Kredisi {info[1]} olan {info[0]} kodlu dersin kontenjanı {info[2]} kişidir.")

# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak
# elemanlarını eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])
sonuc = ""
if kume1.issuperset(kume2):
    sonuc = kume1.intersection(kume2)
else:
    sonuc = kume2.difference(kume1)
print(sonuc)
