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
pd.Series({"Paris": [10], "Berlin": [20]})
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
    'start_date': [2010, 2011, 2012, 2013]
})

df3 = pd.merge(df1, df2)  # neye göre birleştirme işlemini vermedirk eğer verirsek on kullanıırz
pd.merge(df1, df2, on="employees")
df4 = pd.DataFrame({'group': ["accounting", "engineering", "hr"],
                    'manager': ["Caner", "Mustafa", "Berkcan"]})
df3.columns
# Her çalışanınn  müdür bilgisini istiyoruz.
pd.merge(df3, df4)  # groupa göre getiriri
# genellikle bu işlemler veri tabanında yapılır. tekilleştirilmiş işlemler getirilir.


#####################################################################
# veri görselleştirme: matplotlib & seaborn
#####################################################################


#####################################################################
# matplotlib: veri görselleştirmenin atasıdır. low level bir kütüphanedir., seaborn ise high leveldir.
#####################################################################

# kategorik değişken: sutun grafiği ile değiştiriyoruz. countplot bar
# sayısal değşiken: hist, boxplot, -> veri görselleştirme için python uygun değildir. BI araçları daha iyidir.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

# sayısal değer görselleştirme
plt.hist(df['age'])
plt.show()

plt.boxplot(df["fare"])
plt.show()
#####################################################################
# Matplotlib özellikleri
#####################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
# katmanlı şekilde veri görselleştirme yapar
# plot özelliği: veriyi görselletirmek için kullanılır.
x = np.array([1, 8]);
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

x = np.array([2, 4, 6, 8, 10]);
y = np.array([1, 3, 5, 7, 8])
plt.plot(x, y)
plt.plot(x, y, "o")

plt.show()

#####################################################################
# marker
#####################################################################
y = np.array([13, 3, 35, 7, 58])

plt.plot(y, marker="*")
plt.show()

markers = ['o', '*', '.', ',', 'x', "X", "+", "P", "s", "D", "d", "H", "h"]

#####################################################################
# line: çizgi özelleği oluşturmak için kullanılır,
#####################################################################
y = np.array([14, 12, 44, 3])
plt.plot(y, linestyle="dashed")
plt.show()

y = np.array([14, 12, 44, 3])
plt.plot(y, linestyle="dotted")
plt.show()

y = np.array([14, 12, 44, 3])
plt.plot(y, linestyle="dashdot", color="r")
plt.show()

#####################################################################
# multiplelines:
#####################################################################
x = np.array([2, 10, 1, 17, 260])
y = np.array([13, 3, 35, 7, 58])
plt.plot(x, linestyle="dashdot", color="g")
plt.plot(y, linestyle="dashdot", color="r")
plt.title("Title")
plt.xlabel("X label")
plt.ylabel("Y label")
plt.grid()
plt.show()

#####################################################################
# SUBPLOT: birden fazla göreselin gösterilmesidir.
#####################################################################
x = np.array([2, 10, 1, 17, 260])
y = np.array([13, 3, 35, 7, 58])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

x = np.array([2, 10, 1, 17, 260])
y = np.array([13, 3, 35, 7, 58])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

x = np.array([2, 10, 1, 17, 260])
y = np.array([13, 3, 35, 7, 58])
plt.subplot(1, 3, 3)
plt.title("2")
plt.plot(x, y)
plt.show()

#####################################################################
# Seaborn ile Veri Görselleştirme. yüksek seviyeli bir veri görselleştirme kütüphanesidir.
#####################################################################
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df["sex"].value_counts().plot(kind="bar")
plt.show()

#####################################################################
# Seaborn ile sayısal Veri Görselleştirme.
#####################################################################

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

#####################################################################
#  Python ile Veri Analizi: Gelişmiş Fonksiyonel Keşifçi Veri Analizi
#####################################################################
# Genel Resim
# Kategorik değişken analizi (Analysis of categorical variables)
# Sayısal değişken analizi (Analysis of numerical variables)
# Hedef değişken analizi(Analysis of target variables)
# Korelasyon analizi (Analysis of oorrelation)
#####################################################################
#  Genel Resim: Elimize gelen veriyi tanıma, kaç gözlem,kaç değişken, ilk gözlemlere bakalım, değişkenlere bakalım,
# veri içerisinde eksiklik var mı,varsa hangi değişkende kaçar tane bakalım.
#####################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
## bir veriseti elimize geldiğinde yapılacaklar
df.head()
df.tail()
df.shape  # satır ve sutun bilgisi
df.info()  # tablonun hakkında özet bilgi
df.columns  # değişkenn isimlerini alırız
df.index  # index bilgisini almak istersek
df.describe().T  # sayısal bilgilerin betimsel istatistikleri
df.isnull().values.any()  # tüm veride eksik değer var mı yokmu
df.isnull().sum()  # eksik olan veri hangi sutunda kaçar tane


def check_df(dataframe, head=5):
    print("########################### Shape ###########################")
    print(dataframe.shape)
    print("########################### dtypes ###########################")
    print(dataframe.dtypes)
    print("########################### head ###########################")
    print(dataframe.head(head))
    print("########################### tail ###########################")
    print(dataframe.tail(head))
    print("########################### NA ###########################")
    print(dataframe.isnull().sum())
    print("########################### quantiles ###########################")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)


check_df(df)

df = sns.load_dataset("tips")
check_df(df)

df = sns.load_dataset("glue")
check_df(df)

#####################################################################
#  Kategorik değişken analizi (Analysis of categorical variables)
#
# Programatik şekilde kategorik şekilde analiz yapacağız. Göz yordamıyla çok veri için mümkün değildir.
# Tek değişken işin: value_counts fonksiyonunu kullanırsın ancak çok veri için biz fonkisyon yazalım.
#####################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df["embarked"].value_counts()
df["sex"].unique()  # benzersiz değerleri listeler
df["sex"].nunique()  # benzersiz değişken sayısı
df["class"].nunique()  # benzersiz değişken sayısı

df.info()  # veriseti içerisinde bütün olası kategoriler seçilecek
# öncelikle tip bilgisine göre seçeceğiz.(bool, category ve object)
# diğer sinsileler: survived, pclass,... gibi değişkenlerde kategoriktir
# biz bool, category, object dışında olası kategorik bilgiler içeren verileri yakalayacağız.


cat_cols = [col for col in df.columns if str(df[col].dtype) in ["category", "object", "bool"]]

# df["sex"].dtype
# str(df["sex"].dtype)
# str(df["sex"].dtype) in ["object"]

# numeriklerden kategorik değişken algılamak

# tipi int veya float olup eşsiz sınıf sayısı belirli bir değerden küçük olanları yakalatmaya calışalım
# belli eşik değeri yüksek olan örnek isim değişkenini varsayalım. 800 küsür adettir. Kardinatilesi yüksek değişken değerdir
# ölçülemeyecek kadar açıklanamayacak kadar yüksek değerler bilgi taşımadığı anlamına varılabilir
cat_cols = [col for col in df.columns if str(df[col].dtype) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtype) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                        }))
    print("--------------------------------------------------------------------------------")


# otomatik olarak dataframe içerisindeki kategorik verileri yüzdesel olarak dataframe içerisinde göstermek
for col in cat_cols:
    cat_summary(df, col)


# Ölçeklenebilirlik konusu olduğu için en basit olarak düşünülmelidir.
# Grafik özelliği ekleyelim
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                        }))
    print("--------------------------------------------------------------------------------")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(blok=True)


cat_summary(df, "sex", True)

for col in cat_cols:
    if df[col].dtype == "bool":  # ekranda görünmediği için bu şekilde önlem aldık.Ve bool tipini inte çevirdik.
        df[col] = df[col].astype(int)
        cat_summary(df, col, True)
    else:
        cat_summary(df, col, True)


####### aşağıdaki karmaşık oldu mümkün oldukça yuakarıdaki gibi ufak fonksiyonlara ayırıp yapmalıyız


def cat_summary(dataframe, col_name, plot=False):
    if df[col_name].dtype == "bool":
        df[col_name] = df[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                            }))
        print("--------------------------------------------------------------------------------")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(blok=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                            }))
        print("--------------------------------------------------------------------------------")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(blok=True)


for col in cat_cols:
    if df[col].dtype == "bool":  # ekranda görünmediği için bu şekilde önlem aldık.Ve bool tipini inte çevirdik.
        df[col] = df[col].astype(int)
        cat_summary(df, col, True)
    else:
        cat_summary(df, col, True)

#####################################################################
#  Sayısal değişken analizi (Analysis of numerical variables)
#####################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtype) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if
            df[col].dtypes in ["int", "float"]]  # sayısal değişkenleri seçtik, buradakilerden bazıları
# sayısal olmayabilir ve karegorik olabilir.
# num cols içerisinde olup cat cols içerisinde olmayanları seç
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_col):
    # describeyi detaylandırmak için çeyreklik değerleri almak isteyebiiliriz
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    # describeyi detaylandırmak için çeyreklik değerleri almak isteyebiiliriz
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.ylabel(numerical_col)
        plt.show(block=True)


num_summary(df, "age", True)

for col in num_cols:
    num_summary(df, col, True)

#####################################################################
# Değişkenlerin yakalanması ve işlemlerin genelleştirilmesi
#####################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()


# kategorik ve numerik değişkenleri ayrı ayrı getirilmesi istiyoruz.

# doc string: döküman yazma işlemi, bizden başka kişiler bunları kullanması için bir kılavuzdur.
# cat_th: belirtilen değerden küçükse kategorik değişken muamelesi yapacağız
# car_th: belirtilen değerden büyükse numerik değişken muamelesi yapacağız
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve  kategorik fakat kardinal değişkenlerinin isimlerini verir

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat numerik olan değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi,
    num_cols: list
        Numerik değişken listesi,
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes:
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ["int", "float"]]

    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtype) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

print("*************************************************************************************")


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                        }))
    print("--------------------------------------------------------------------------------")


for col in cat_cols:
    cat_summary(df, col)
print("*************************************************************************************")


def num_summary(dataframe, numerical_col, plot=False):
    # describeyi detaylandırmak için çeyreklik değerleri almak isteyebiiliriz
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.ylabel(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, True)

print("*************************************************************************************")
# help(grab_col_names)

# bonus - Bütün senaryoŞ
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                        }))
    print("--------------------------------------------------------------------------------")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(blok=True)


for col in cat_cols:
    cat_summary(df, col, True)


def num_summary(dataframe, numerical_col, plot=False):
    # describeyi detaylandırmak için çeyreklik değerleri almak isteyebiiliriz
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.ylabel(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, True)

#####################################################################
# Hedef değişken analizi(Analysis of target variables)
#####################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve  kategorik fakat kardinal değişkenlerinin isimlerini verir

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat numerik olan değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi,
    num_cols: list
        Numerik değişken listesi,
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes:
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ["int", "float"]]

    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtype) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["survived"].value_counts()


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                        }))
    print("--------------------------------------------------------------------------------")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(blok=True)


cat_summary(df, "survived", True)
##---##

df.groupby("sex")["survived"].mean()


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_NAME": dataframe.groupby(categorical_col)[target].mean()}))


target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)
    print("-------------------------------------------------------")

df.head()

##-------------
df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age": "mean"})


def target_summary_with_num(dataframe, target, categorical_col):
    print(dataframe.groupby(target).agg({categorical_col: "mean"}))


target_summary_with_num(df, "survived", "age")

#####################################################################
# Korelasyon analizi (Analysis of oorrelation)
##
#####################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]  # problemli değişkeni dışarıda bırakmak için 1:-1 yapıldı.
df.head()
# ısı haritası aracığıyla korelasyona bakacağız, yüksek korelasyonlu bir değişkenlerden bazılarını dışarıda bırakalım.

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

# ISI HARİTASI OLUŞTURACAĞIZ VE yüksek KORELASYONLARI ELEMEYE ÇALIŞACAĞIZ.
# korelasyon hesaplama: değişkenkerin birbirlerini arasındaki ilişkidir. -1,1 arasında değer alır.
# -1 veya 1 e yaklaştıkça ilişkinin kuvveti artar,
# iki değişken arası pozitifse pozitif korelasyon, iki değişken değeride artar
# iki değişken arası negatifse, bir değişken artarsa diğeri azalır.
# 0 ise korelasyon yok demektir. birbiryle yüksek korelasyon olan değişkenler olmamasını isteriz.
corr = df[num_cols].corr()
# Isı haritası
sns.set(rc={'figure.figsize': (12, 12)})  #
sns.heatmap(corr, cmap="RdBu")
plt.show()

## Yüksek korelasyonlu değişkenlerin silinmesi
# negatif veya pozitif olmasıyla ilgilenmiyoruz. aynı olacağını düşünerek mutlağını alıyoruz.
corr_matrix = df.corr().abs()

# birlerden oluşan, matrisin boyutunda bir numpay arrayı oluştururuz ve true false ceviriyoruz.
# daha sonra ücgen şekilde yani tekrarlanmayan yapıda korelasyon yapısı oluşturuyoryz
upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]

corr_matrix[drop_list]

df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})  #
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df, True)
drop_list = high_correlated_cols(df, True)

high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

###############

df = pd.read_csv("datasets/train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df,plot=True)

len(df.drop(drop_list,axis=1).columns)