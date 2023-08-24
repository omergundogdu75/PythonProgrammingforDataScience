#####################################################################
# Python ile veri analizi
#####################################################################
# numpy
# pandas
# veri görselleştirme: matplotlib % seaborn
# gelişmiş fonksiyonel keşifci  veri analizi(advanced func exploratory data analysis)


#####################################################################
# numpy: bilimsel hesaplamalar için kullanılır, array, çok boyutlu arrayler, matrixler için performaslı yapılardır.
# Verimli veri saklama, vektorel operasyonlardır. fix type değişkeni ile listelere göre daha hızlı işlem yapar.
# listelerde her bir değişkenin bilgilerini ayrı ayrı tutarken, numpy sabir bir veri tutar. her biri için tip , boyut ve meta bilgileri tutmaz.
# döngü yazmaya gerek kalmadan bazı işlemleri yaptırır.
#####################################################################
# neden numpy
# numpy arrayi oluşturmak
# numpy array özellikleri
# yeniden şekillendirme
# index seçimi
# slicing
# fancy index
# numpy da koşullu işlemler
# matematiksel işlemler

#####################################################################
# neden numpy: hız, vektorsel seviyeden işlemler yaoar
#####################################################################
import numpy as np  # np burada kısa koddur

# klasik yöntem
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]
ab = []
for i in range(len(a)):
    ab.append(a[i] * b[i])

# numpy yöntemi
a = np.array(a)
b = np.array(b)
a * b
#####################################################################
# Numpy arrayı oluşturmak
#####################################################################

import numpy as np

d = np.array([1, 2, 3, 4, 5])
type(d)
np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))

#####################################################################
# numpy array özellikleri
#####################################################################

# ndim : boyut sayısı
# nshape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype
#####################################################################
# yeniden şekillendirme
#####################################################################
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 9, size=9)
ar.reshape(3, 3)

#####################################################################
# index seçimi
#####################################################################
a = np.random.randint(10, size=10)
a = np.random.randint(10, size=2)

a[0]
a[0:4]
a[0] = 999

a = np.random.randint(10, size=(4, 4))
a[1][1] = 12.2  # sabit veri olacağından 12 değerini alır

a[:, 0]  # tüm satırların, 0.sutununu seçer
a[1, :]  # 1.satırın tüm sutunlarını seçer
a[0:2, 1:5]  # 0.satırdan 2.satıra kadar 2. dahildeğil. 1.sutundan 5. sutuna kadar 5. dahil değil

#####################################################################
# fancy index seçimi
#####################################################################
v = np.arange(0, 30, 3)
print(v ** 2)
v[1]
v[4]
catch = [1, 2, 3]
v[catch]  # fancy index birden fazla index deki bilgileri array içerisinde getirir

#####################################################################
# numpy da koşullu işlemler
#####################################################################
v = np.array([1, 2, 3, 4, 5])
# klasik döngü
ab = []
for i in v:
    if i < 3:
        ab.append(i)

# numpy ile koşullu işlemler: true false işlemleride gönderebiliriz
v < 3
v[v < 3]  # true olanları seçer
v[v > 3]  # true olanları seçer
v[v >= 3]  # true olanları seçer
v[v <= 3]  # true olanları seçer
v[v == 3]  # true olanları seçer
v[v != 3]  # true olanları seçer
#####################################################################
# numpy da matematiksel işlemler
#####################################################################
v = np.array([1, 2, 3, 4, 5])
v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.ma()
np.var(v)

## iki bilinmeyenli denklem çözümü
a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])
np.linalg.solve(a, b)

#####################################################################
# pandas: veri manipülasyonu ve yada veri analizi dendiğinde akla gelen kütüphanedir. Ml, veri analizi, derin öğrenme için ödenmlidir
#####################################################################


# PANDAS SERİLERİ: En yaygın kullanılan, tek boyutlu ve index bilgisi barındıran veri tipidir.
###############################################
import pandas as pd

s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # index bilgisiyle liste verilerini pandas serisi oluştururur
type(s)
pd.Series({"Paris":[10], "Berlin":[20]})
s.index  # RangeIndex(start=0, stop=11, step=1)
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head()  # ilk 5 değeri getirir
s.head(3)  # ilk 3 değeri getirir
s.tail(3)  # son 3 değeri getirir

# VERİ OKUMA
###############################################
df = pd.read_csv("datasets/advertising.csv")
df.head()
# pandas cheatsheet -> pandasta kullanılan fonksiyonları toplu olarak bulabilirsin

# VERİYE HIZLI BAKIŞ
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape  # dataset ile ilgili satır sutun bilgisi verdi
df.info()  # data set için Gözlem miktarları boşluk varmı yokmu bakılır,
# category ile object ler kategorik değişkenlerdir
# hangi  data türünden kaçar tane olduğu gösterilir
df.columns
df.index
df.describe().T  # data set için özet bilgi verildi. max,min count std bilgisi
df.isnull().values.any()  # dataset içerisinde en az birer tana null var mı
df.isnull().sum()  # herbir değişkendeki eksiklik bilgilerini veriri
df["sex"].value_counts()  # bir dataset içerisindeki bir kategorik değişkene ait bilgilerden kaçar tane olduğu bilgisi

# PANDASTA SEÇİM İŞLEMLERİ
########################################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

df.index
df[0:23]
deleted_index = [1, 2, 3, 4]
df.drop(deleted_index, axis=0).head()  # 1,2,3,4 index in satırını sil, kalıcı değildir.

df.drop(0, axis=0).head()  # 0 index in satırını sil, kalıcı değildir.
# kalıcı olması için
# df = df.drop(deleted_index,axis=0,inplace=True) inplace ile değişiklik kalıcı yapılır.

## değişkeni indexe çevirmek
df["age"].head()
df.age.head()
df.index = df["age"]
# sutun bilgisini silmek
df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True).head()  # kalıcı silme işlemi
df.head()

## indexi değişkene çevirmek
#####################
# 1.yol
df.index
df["age"] = df.index  # seçilen data dataset içerisinde varsa bu değişken seçilirken, yoksa yeni değişken eklenir
df.head()

# 2.yol
df.drop("age", axis=1, inplace=True)
df.reset_index().head()  # indexi siler, sutun olarak yeni değişken ekler
df = df.reset_index().head()
df.head()

# Değişkenler üzerinde işlemler
##########################################
pd.set_option("display.max_columns", None)  # çıkıtıdaki ... dan kurtulmak için yapılır.
df = sns.load_dataset("titanic")
df.head()

"age" in df  # değişkenin varlığı dataframede sorgulanır

df["age"].head()
type(df["age"].head())  # pandas serisi
Ş
df[["age"]].head()
type(df[["age"]].head())  # dataframe olarak tipidir çift köşeli parantez için

df[["age", "alive"]].head()  # df içerisinden birden fazla değişken şeçmek için

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"] ** 2  # elimizde olmayan bir değişken yazarsan onuda ekler
df.head()

df["age3"] = df["age"] / df["age2"]  # elimizde olmayan bir değişken yazarsan onuda ekler
df.head()

# değişken silmek
df.drop("age3", axis=1).head()  # gözlem için bu şekildei kalıcı olması için inplace=True eklenmelidir.
df.drop(col_names, axis=1).head()

# belirli bir string ifade barındıran değişkenleri silmek istersek
df.loc[:, df.columns.str.contains("age")].head()  # label base seçim yapmak için kullanılır,
# tüm satırları seçip, age ifadesi yer alıyor mu bakılır
# silmek için
df.loc[:, ~df.columns.str.contains("age")].head()

## iloc: index bilgisi vererek seçim yapma işlemi yapar, integer baser selection
## loc: dataframelerde string vererek seçim için kullanılan özel yapılardır
df.iloc[0:3]  # satır seçme işlemi
df.iloc[0, 0]  # satır, sutun seçme işlemi

# loc: label based selection, mutlak olarak isimlendirmeyi seçer
df.loc[0:3]

# satırlardan 0dan 3 e kadar ama sutundanda label seçmek istersek
df.iloc[0:3, 0:3]  # hata veriri
df.loc[0:3, "age"]  # istediğimizi yapar.

# iloc: e kadar işlemi yapar, loc ise e kadar değil dahil işlemi yapar
# loc da fancy işlemi geçerlidir
col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

# KOŞULLU SEÇİM İŞLEMİ
######################################################################

# VERİ SETİNDE YASI 50DEN BÜYÜK OLANLARI ALMAK İÇİN
df[df["age"] > 50].head()
# VERİ SETİNDE YASI 50DEN BÜYÜK OLAN kaçkişi var
df[df["age"] > 50]["age"].count()
# YOLCULUUK SINIFLARINI MERAK EDERSEK
df.loc[df["age"] > 50, ["class", "age"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["class",
                                                  "age"]].head()  # Satır için birden fazla koşul giriliyorsa parantesze alınmalı

# Satır için birden fazla koşul giriliyorsa parantesze alınmalı
def_new = df.loc[
    (df["age"] > 50)
    & (df["sex"] == "male")
    & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
    ["class", "age", "embark_town"]
]

df["embark_town"].value_counts()
def_new["embark_town"].value_counts()

# TOPLULAŞTIRMA VE GRUPLAMA(Aggregation & Grouping)
# Bir veri yapısının içerisindeki verileri toplulaştırmaktır. Tek başına kullanıldığında özet istatistiği verir.
# Group by ile kullanabiliriz
##################################################################
# count(), first(), last(), mean(), median(), min(),max(), std(), sum(), var(), pivot table
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "embark_town": "count",
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"
})

# Pivot Table : Group by a benzer, kırılımlar açısından değerlendirmek ve ilgilendiğimiz özet istatistiği görüntüler
########################################################
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")  # kesişimde ne görülecek, satırda ne gürülecek, sutunda ne görülecek.
# keşisimlerdeki değerler ortlamadır. pivot table ön tanımlı değerleri mean dir bunu aşağıdaki gibi biçimlendiririz.

df.pivot_table("survived", "sex", ["embarked", "class"], aggfunc="std")

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40,
                                   90])  # cur ve qcut: elimizdeki sayısal değişkeni kategorik değişkene cevirir.
# elimizdeki sayısal değişkenleri hangi değişkenlere bölmek istiyorsak cut,
# bilmiyorsak çeyreklik değerlere bölmek istiyorsak, qcut kullanılır
df.head()

df.pivot_table("survived", "sex", ["new_age", "class"])
pd.set_option("display.width", 500)

# Apply & Lambda
##########################################################################
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

# Apply: satır yada sutunlarda otomatik olarak fonksiyon çalıştırmaya yarar
# Lambda: bir fonksiyon tanımlama şeklidir. Kullan at fonksiyonudur.
df['age2'] = df["age"] * 2
df['age3'] = df["age"] * 5
df.head()

(df["age"] / 10).head()
(df["age2"] / 10).head()
(df["age3"] / 10).head()

# değişkenler fonksiyon uygulamak için
# klasik yöntem
for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10
df.head()

# apply ve lambda uygulanarak
df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x / 10).head()

# standartlaştırma için
df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()


# standartlaştırma için
# df.loc[:,["age","age2","age3"]] = df.loc[:,df.columns.str.contains("age")].apply(standart_scaler)
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
df.head()

# BİRLEŞTİRME İŞLEMLERİ (JOIN), concat ve merge yöntemlerini kullanacağız

m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])  # iki dataframe yi birleştiriri,
pd.concat([df1, df2], ignore_index=True)  # index bilgisini sıfırlar
# sutun bazındada birleştirme yapabiliriz, ön tanımı satır şeklindedir.


# merge ile birleştirme işlemi
df1 = pd.DataFrame({
    'employees': ["john", "dennis", "mark", "maria"],
    'group': ["accounting", "engineering", "engineering", "hr"]
})

df2 = pd.DataFrame({
    'employees': ["john", "dennis", "mark", "maria"],
    'start_date': [2010,2011,2012,2013]
})

df3 = pd.merge(df1,df2) # neye göre birleştirme işlemini vermedirk eğer verirsek on kullanıırz
pd.merge(df1,df2,on="employees")
df4 = pd.DataFrame({ 'group': ["accounting", "engineering", "hr"],
                     'manager':["Caner","Mustafa","Berkcan"]})
df3.columns
# Her çalışanınn  müdür bilgisini istiyoruz.
pd.merge(df3,df4) # groupa göre getiriri
# genellikle bu işlemler veri tabanında yapılır. tekilleştirilmiş işlemler getirilir.


#####################################################################
# veri görselleştirme: matplotlib & seaborn
#####################################################################


#####################################################################
# gelişmiş fonksiyonel keşifci  veri analizi(advanced func exploratory data analysis)
#####################################################################
