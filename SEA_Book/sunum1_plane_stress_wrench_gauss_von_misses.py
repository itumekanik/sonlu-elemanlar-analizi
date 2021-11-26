import numpy as np   # sayısal işlemler için python kütüphanesi
INV = np.linalg.inv  # matris ters alma fonksiyonu
DET = np.linalg.det  # matris determinant alma fonksiyonu
ABS = np.abs         # Mutlak değer alma fonksiyonu
np.set_printoptions(suppress=True)  # sayıların yazım (print) ayarları için

# 2D bölgede n=2 Gauss nokta-ağırlık integrasyon şeması
def IntegrateOn2DDomainWithGaussN2(g):
    total = 0
    p = [-1 / 3**0.5, 1 / 3**0.5]
    w = [1, 1]
    for i in range(2):
        for j in range(2):
            total += w[i] * w[j] * g(p[i], p[j])
    return total

# 2D sınırlarda n=2 Gauss nokta-ağırlık integrasyon şeması
def IntegrateOn2DBoundariesWithGaussN2(g):
    total = 0
    p = [-1 / 3**0.5, 1 / 3**0.5]
    w = [1, 1]
    for i in range(2):
        rb = [p[i], p[i], -1, 1]
        sb = [-1, 1, p[i], p[i]]
        for k in range(4): # k: sınır numarası
            total += w[i] * g(rb[k], sb[k], k)
    return total

# Şekil fonksiyonları vektörü [Fi] (4x1)
def SF(r, s):
    return 0.25 * np.asarray([(1 - r) * (1 - s),
                              (1 + r) * (1 - s),
                              (1 - r) * (1 + s),
                              (1 + r) * (1 + s)])
# Şekil fonksiyonlarının türev matrisi [Fi,r] (4x2)
def dSF_dr(r, s):
    return 0.25 * np.asarray([[-1 + s, -1 + r],
                              [ 1 - s, -1 - r],
                              [-1 - s,  1 - r],
                              [ 1 + s,  1 + r]])

nodes = dict()     # Nodları hafızada tutacak konteyner
elements = dict()  # Elemanları hafızada tutacak konteyner

class Node:
    def __init__(node, id, X, Y):
        node.id = id
        node.X, node.Y = X, Y   # Koordinatlar
        node.rest = [0, 0]      # Mesnet Koşulu [0: serbest, 1:tutulu]
        node.force = [0, 0]     # Tekil-Yük [Px, Py]
        node.disp = [0, 0]      # Mesnet Çökmesi [delta_x, delta_y]
        node.code = [-1, -1]    # Serbestlikler (Kod) [dx, dy]
        nodes[id] = node  # Nod nesnesi nodes içerisinde saklanıyor
        node.values = []  # Etraf elemanlardan gelecek çizim değerleri dizisi

    # grafik programlarının çizim için çağıracağı fonksiyon (Çözüm sonrası)
    def get_draw_value(node):
        return sum(node.values) / len(node.values)

class Element:
    def __init__(elm, id, conn, E, p, h):
        elm.id = id
        elm.conn = [nodes[id] for id in conn]  # Bağlantı Haritası [DN1, DN2, DN3, DN4]
        elm.E, elm.p, elm.h = E, p, h  # Malzeme ve kesit
        elm.boundaryForceX = [0] * 4   # Sınır-Yüzey X [q1x, q2x, q3x, q4x]
        elm.boundaryForceY = [0] * 4   # Sınır-Yüzey Y [q1y, q2y, q3y, q4y]
        elm.volumeForce = [0, 0]       # Hacim Kuvvetleri [bx, by]
        elements[id] = elm  # Eleman nesnesi elements içerisinde saklanıyor

    # Kod-Vektörü [u1, u2, u3, u4, v1, v2, v3, v4]
    def code(elm):
        return [n.code[0] for n in elm.conn] + \
               [n.code[1] for n in elm.conn]

    # Nodal koordinat matrisi [X] (2x4)
    def XM(elm):
        n1, n2, n3, n4 = elm.conn
        return np.asarray([[n1.X, n2.X, n3.X, n4.X], [n1.Y, n2.Y, n3.Y, n4.Y]])

    # Jacobian Matrisi [J]
    def JM(elm, r, s):
        return elm.XM() @ dSF_dr(r, s)

    # ABS(Det(JM))
    def detJM(elm, r, s):
        return ABS(DET(elm.JM(r, s)))

    # Şekil fonksiyonlarının gerçek koordinatlara göre türev matrisi [Fi,x] (4x2)
    def dSF_dx(elm, r, s):
        return dSF_dr(r, s) @ INV(elm.JM(r, s))  # [Fi,r].INV([J])

    # Genleme-yer değiştirme matrisi [B] (3x8)
    def BM(elm, r, s):
        empty = np.zeros((3, 8))
        mat = elm.dSF_dx(r, s)
        empty[0, 0:4] = mat[:, 0]
        empty[1, 4:8] = mat[:, 1]
        empty[2, 0:4] = mat[:, 1]
        empty[2, 4:8] = mat[:, 0]
        return empty

    # Bünye (malzeme) matrisi [C] (3x3)
    def C(elm):
        E, p = elm.E, elm.p
        return E / (1 - p**2) * np.asarray([[1, p, 0],
                                            [p, 1, 0],
                                            [0, 0, 0.5*(1-p)]])

    # Rijitlik Matrisi [Ke] (8x8)
    def K(elm):
        def dK(r, s):  # Matrisin integradı
            h = elm.h
            C = elm.C()
            BM = elm.BM(r, s)
            J = elm.detJM(r, s)
            return h * BM.T @ C @ BM * J
        return IntegrateOn2DDomainWithGaussN2(dK)

    # Eleman hacim-kuvvetleri vektörü (Be) (8x1)
    def B(elm):
        def dB(r, s):  # Vetörün İntegrandı
            bx, by = elm.volumeForce
            if (bx == 0 and by == 0): return np.zeros(8)
            h = elm.h
            J = elm.detJM(r, s)
            SFV = SF(r, s)
            SF8 = np.concatenate((SFV, SFV))
            return h * J * SF8 * [bx, bx, bx, bx, by, by, by, by]
        return IntegrateOn2DDomainWithGaussN2(dB)

    # Eleman sınır-yüzey dış yükleri vektörü (Se) (8x1)
    def S(elm):
        def dS(r, s, k):  # Vetörün İntegrandı
            qx = elm.boundaryForceX[k]
            qy = elm.boundaryForceY[k]
            if (qx == 0 and qy == 0): return np.zeros(8)
            SFV = SF(r, s)
            JM = elm.JM(r, s)
            JTJ = JM.T @ JM
            J = JTJ[0, 0]**0.5 if k in [0, 1] else JTJ[1, 1]**0.5
            return J * np.concatenate((SFV * qx, SFV * qy))
        return IntegrateOn2DBoundariesWithGaussN2(dS)

    # US ile verilen sistem deplasman vektöründen
    # ilgili elemanın deplasmanları ayıklanıyor (Çözüm sonrası)
    def setDisp(elm, US):
        code = elm.code()
        elm.U = US[code]

    # Eleman gerilme alanı hesaplanıyor  Sigma = [C].Be.Ue (Çözüm sonrası)
    def SigmaVec(elm, r, s):
        C = elm.C()
        BM = elm.BM(r, s)
        U = elm.U
        return C @ BM @ U

    # Elemana dışarıdan verilecek 4 adet değer. Bu değerler elemanın
    # nodlarındaki "values" dizisine eklemleniyor (Çözüm sonrası)
    def appendNodeValues(elm, v1, v2, v3, v4):
        n1, n2, n3, n4 = elm.conn
        n1.values.append(v1)
        n2.values.append(v2)
        n3.values.append(v3)
        n4.values.append(v4)

    # Eleman etraf nodlardan kontur çizmine esas değerleri elde eder.
    # Bu fonksiyon "Drawer" objesi tarafından kendi metodu gibi çağırılmaktadır
    def SigmaXAverage(elm): return [node.get_draw_value()
                                    for node in elm.conn]


# ----------------------------------------------------------------
# Sistemin oluşturulması, çözümü ve diğer çözüm sonrası işlemler
# ----------------------------------------------------------------
E = 70e6   # Elastisite modülü
p = 0.33   # Poisson's ratio
h = 0.012  # Kalınlık

# Text dosyasından nod bilgileri okunuyor ve nod nesneleri başlatılıyor
with open("./meshes/nodes.txt", "r") as f:
    for line in f.readlines():
        ID, X, Y, Z = line.split(",")
        Node(int(ID), float(X), float(Y))

# Text dosyasından eleman bilgileri okunuyor ve eleman nesneleri başlatılıyor
with open("./meshes/elements.txt", "r") as f:
    for line in f.readlines():
        ID, n1, n2, n4, n3 = line.split(",")
        Element(int(ID), [int(n1), int(n2), int(n3), int(n4)], E, p, h)

print("Node Count:", len(nodes))
print("Element Count:", len(elements))

# Mesnet noktaları tanımlanıyor
for id in [515, 102, 514, 513,
           467, 468, 469, 87]:
    nodes[id].rest = [1, 1]

# Nodal yükler tanımlanıyor
for id in [*range(165, 185)]:
    nodes[id].force = [0, -1]

#Nodlarda tanımlı serbestlik numaralarının (code) belirlenmesi
M = 0  # M:Toplam serbestlik sayısı
N = 0  # N:Bilinmeyen serbestliklerin sayısı

# Tutulu olmayanlar (rest==0) numaralanıyor
for id, node in nodes.items():
    for index, rest in enumerate(node.rest):
        if rest == 0:
            node.code[index] = M
            M += 1

N = M  # Toplam bilinmeyen serbestlik sayısı N de saklanıyor

# Tutulu olanlar (rest==1) numaralanıyor
for id, node in nodes.items():
    for index, rest in enumerate(node.rest):
        if rest == 1:
            node.code[index] = M
            M += 1

print("Toplam Serbestlik Sayısı (M)    :", M)
print("Bilinmeyen Serbestlik Sayısı (N):", N)

# Sistem denklem takımının oluşturulması (Birleştirme)
US = np.zeros(M)  # Sistem yer değiştirme vektörü
PS = np.zeros(M)  # Sistem tekil kuvvet vektörü
SS = np.zeros(M)  # Sistem dış sınır yükleri vektörü
BS = np.zeros(M)  # Sistem hacim kuvvetleri vektörü

rows = []  # Sistem rijitlik matrisi terimlerinin satır numaraları
cols = []  # Sistem rijitlik matrisi terimlerinin sütun numaraları
data = []  # Sistem rijitlik matrisi terimleri vektörü

# Rijitlik matrisinin düz-vektör formda oluşturulması
for id, elm in elements.items():
    code = elm.code()
    Ke = elm.K()
    rows += np.repeat(code, len(code)).tolist()
    cols += code * len(code)
    data += Ke.flatten().tolist()

import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

KS = sp.sparse.coo_matrix((data, (rows, cols)), shape=(M, M), dtype=float).tocsc()

for id, elm in elements.items():
    code = elm.code()
    BS[code] += elm.B()  # Hacim kuvvetleri
    SS[code] += elm.S()  # Dış-sınır çizgisel yükleri

for id, node in nodes.items():
    code = node.code
    US[code] = node.disp   # Nodal mesnet-çökmeleri
    PS[code] = node.force  # Nodal tekil-kuvvetler

# Sistem denklem takımının çözümü
K11 = KS[0:N, 0:N]
K12 = KS[0:N, N:M]
K21 = KS[N:M, 0:N]
K22 = KS[N:M, N:M]

U2 = US[N:M]

P1 = PS[0:N]
B1 = BS[0:N]
B2 = BS[N:M]
S1 = SS[0:N]
S2 = SS[N:M]

# Herhangi bir sebepten dolayı rijitlik matrisnin diyagonalinde
# çok küçük bir terim olması durumunda, matrise ufak bir değer
# eklenip uyarı veriliyor.
for i in range(N):
  if abs(K11[i, i])<1e-10:
      K11[i, i]=1e-5
      print("Sistemde rijit mod olabilir!!!")

U1 = sp.sparse.linalg.bicg(K11, P1 + S1 + B1 - K12 @ U2)[0]
P2 = K21 @ U1 + K22 @ U2 - S2 - B2

US = np.concatenate((U1, U2))
PS = np.concatenate((P1, P2))

# ----------------------------------------------------
# Çözüm çıktılarının yazdırılması ve grafik işlemleri
# ----------------------------------------------------
for id, node in nodes.items():
    ux, uy = US[node.code]
    px, py = PS[node.code]
    print(f"Node {id}: [ux:{ux:.4f}, uy:{uy:.4f}] [px:{px:.2f}, py:{py:.2f}]")

# Von Mises gerilmesi hesabı (kitapta bahsedilmedi)
def vonMises(Gx, Gy, Txy):
    return (Gx**2 - Gx*Gy + Gy**2 + 3 * Txy**2)**0.5

# Eleman deplasmanları ayıklanıyor ve kontur çizimine esas SigmaX değerleri
# nodlara aktarılıyor
for key, elm in elements.items():
    elm.setDisp(US)
    Gx1, Gy1, Txy1 = elm.SigmaVec(-1, -1)
    Gx2, Gy2, Txy2 = elm.SigmaVec(1, -1)
    Gx3, Gy3, Txy3 = elm.SigmaVec(-1, 1)
    Gx4, Gy4, Txy4 = elm.SigmaVec(1, 1)
    elm.appendNodeValues(vonMises(Gx1, Gy1, Txy1),
                         vonMises(Gx2, Gy2, Txy2),
                         vonMises(Gx3, Gy3, Txy3),
                         vonMises(Gx4, Gy4, Txy4))
    # elm.appendNodeValues(Gx1, Gx2, Gx3, Gx4)

# Grafik büyütme faktörünün hesaplanması:
# Sistemin Lx boyu hesaplanıyor ve Y doğrultusundaki maksimum
# yer değiştirme büyütmesinin Lx in %30’u kadar olması sağlanıyor
minX= min(node.X for id, node in nodes.items())
maxX= max(node.X for id, node in nodes.items())
Lx = maxX - minX
abs_uy_max = max(abs(US[node.code][1]) for id, node in nodes.items())
factor = 1
if abs_uy_max > 0: factor = 0.3 * Lx / abs_uy_max

# Yer değiştirmiş nod koordinatları hesaplanıyor (x=X+ux, y=Y+uy, z=Z+uz)
for id, node in nodes.items():
    node.Z, node.z = 0, 0
    ux, uy = US[node.code]
    node.x = node.X + factor * ux
    node.y = node.Y + factor * uy

from drawing import Drawer, LineMaps, TriangleMaps

class Draw(Drawer):
    # Bir elemanı çizmek için gerekli üçgenlerin rölatif nod haritası
    trigs = TriangleMaps([0, 1, 2],
                         [1, 2, 3])

    # Bir elemanı çizmek için gerekli çizgilerin rölatif nod haritası
    lines =  LineMaps([0, 1], [1, 3], [3, 2], [2, 0])

draw = Draw(elements=elements,
            nodes=nodes,
            on1=['X', 'Y', 'Z'],
            on2=['x', 'y', 'z'],
            connectivity_name="conn",
            mesh=False,
            name="SigmaX",
            lookat=[0, 0, 1])

draw.SigmaXAverage()
