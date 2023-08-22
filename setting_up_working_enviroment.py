########################################
# Sayılar (Numbers) ve Karakter dizileri (String)
#######################################

print("Hello world")
print("Hello AI Era")

print(9)
9.2

type(9)
type(9.2)

type("Mrb")

########################################
# Atamalar ve Değişkenler (Assignments & Variables)
#######################################

a = 9
b = "hello ai era"
b
a

c = 10
a * c

d = a - c
d

########################################
# Virtual Enviroment (Sanal ortam) ve (Package Management) Paket yönetimi
#######################################

#Sanal ortam listelenmesi:  conda env list
#Sanal ortam oluşturma: conda create -n myenv
#Oluşturulan sanal ortamı aktif etme: conda activate myenv
#Ortam içerisindeki paketleri listelemek: conda list
#Sanal ortama paket yüklemek: conda install packagename
#Aynı anda birden fazla paket yüklemek: conda install packagename packagename1
# Bellirli bir özel paket versiyonunu kullanmak istersek: conda intall numpy=1.20.1
# Paket silme: conda remove packagename
# paket yükseltme için: conda upgrade numpy -> engüncel versiyona yükselir
# tüm paketleri güncellemek istersek: conda upgrade -all

#pip: pypi (python package index) paket yönetim aracı
#paket yükleme: pip install paketadı


# çalıştığımız yani kullandığımız paketleri başka yere aktarmak için: conda env export > environment.yaml (conda için)
# çalıştığımız yani kullandığımız paketleri başka yere aktarmak için: conda env export > requirement.txt (pip için)

# başka bir bilgisayarda çalışma ortamına geçerek yada arkadaşıma aktarmak istersek:
# bulunan sanal ortamı deaktif işlemi için: conda deactivate
# sanal ortamı silmek için: conda env remove -n myenv

# bir paket ve sanal ortam bilgisi içeren environment.yaml içerisinden sanal ortamı kurup paketleri yüklemek
# conda env create -f environment.yaml
