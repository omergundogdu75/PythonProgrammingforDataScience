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
print(v**2)
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
b = np.array([12,10])
np.linalg.solve(a,b)

#####################################################################
# pandas: veri manipülasyonu ve yada veri analizi dendiğinde akla gelen kütüphanedir. Ml, veri analizi, derin öğrenme için ödenmlidir
#####################################################################


# PANDAS SERİLERİ: En yaygın kullanılan, tek boyutlu ve index bilgisi barındıran veri tipidir.
###############################################
import pandas as pd

s   = pd.Series([1,2,3,4,5,6,7,8,9,10,11]) # index bilgisiyle liste verilerini pandas serisi oluştururur
type(s)
s.index # RangeIndex(start=0, stop=11, step=1)
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head() # ilk 5 değeri getirir
s.head(3) # ilk 3 değeri getirir
s.tail(3)# son 3 değeri getirir


# VERİ OKUMA
###############################################


# VERİYE HIZLI BAKIŞ
# PANDASTA SEÇİM İŞLEMLERİ
# TOPLULAŞTIRMA VE GRUPLAMA
# APPLY LAMBDA
# BİRLEŞTİRME İŞLEMLERİ JOIN



#####################################################################
# veri görselleştirme: matplotlib & seaborn
#####################################################################


#####################################################################
# gelişmiş fonksiyonel keşifci  veri analizi(advanced func exploratory data analysis)
#####################################################################
