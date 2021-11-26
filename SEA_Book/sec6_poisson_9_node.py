import numpy as np    # sayısal işlemler için Python kütüphanesi
INV = np.linalg.inv   # matris ters alma fonksiyonu
DET = np.linalg.det   # matris determinant alma fonksiyonu
LEN = np.linalg.norm  # vektör boy hesabı için kullanılacak
ABS = np.abs          # Mutlak değer alma fonksiyonu
np.set_printoptions(suppress=True)  # sayıların yazım (print) ayarları için

# Master eleman nod koordinatları [p1...p9]
master_points = [[-1, -1],
                 [ 1, -1],
                 [ 1,  1],
                 [-1,  1],
                 [ 0, -1],
                 [ 1,  0],
                 [ 0,  1],
                 [-1,  0],
                 [ 0,  0]]

# 2D bölgede n=3 Gauss integrasyon şeması
def IntegrateOn2DDomainWithGaussN3(h):
    p = [-(3 / 5)**0.5, 0, (3 / 5)**0.5]
    w = [5/9, 8/9, 5/9]
    return sum((w[i] * w[j]) * h(p[i], p[j])
               for i in range(3)
               for j in range(3))

# Şekil fonksiyonları vektörü (9x1)
def SF(r, s):
    return np.asarray([0.25 * r * s * (r - 1) * (s - 1),
                       0.25 * r * s * (r + 1) * (s - 1),
                       0.25 * r * s * (r + 1) * (s + 1),
                       0.25 * r * s * (r - 1) * (s + 1),
                       -0.5 * s * (r ** 2 - 1) * (s - 1),
                       -0.5 * r * (r + 1) * (s ** 2 - 1),
                       -0.5 * s * (r ** 2 - 1) * (s + 1),
                       -0.5 * r * (r - 1) * (s ** 2 - 1),
                       (r ** 2 - 1) * (s ** 2 - 1)])

# Şekil fonksiyonlarının master koordinatlara göre türev matrisi (9x2)
def dSF_dr(r, s):
    return np.asarray([
        [0.25 * (-1 + 2 * r) * (-1 + s) * s,   0.25 * (-1 + r) * r * (-1 + 2 * s)],
        [0.25 * (1 + 2 * r) * (-1 + s) * s,    0.25 * (1 + r) * r * (-1 + 2 * s) ],
        [0.25 * (1 + 2 * r) * (1 + s) * s,     0.25 * (1 + r) * r * (1 + 2 * s)  ],
        [0.25 * (-1 + 2 * r) * (1 + s) * s,    0.25 * (-1 + r) * r * (1 + 2 * s) ],
        [-r * (-1 + s) * s,                   -0.5 * (-1 + r ** 2) * (-1 + 2 * s)],
        [-0.5 * (1 + 2 * r) * (-1 + s ** 2),  -r * (1 + r) * s                   ],
        [-r * s * (1 + s),                    -0.5 * (-1 + r ** 2) * (1 + 2 * s) ],
        [-0.5 * (-1 + 2 * r) * (-1 + s ** 2), -(-1 + r) * r * s                  ],
        [2 * r * (-1 + s ** 2),                2 * (-1 + r ** 2) * s             ]
    ])

nodes = dict()     # Nodları hafızada tutacak konteyner
elements = dict()  # Elemanları hafızada tutacak konteyner


class Node:
    def __init__(node, id, Y, Z):
        node.id = id
        node.X = 0             # Çizim için gerekli koordinat
        node.Y, node.Z = Y, Z  # Problemin esas Koordinatları
        node.rest = [0]        # Sınır Koşulu [0: serbest, 1:tutulu]
        node.code = [-1]       # Serbestlik Kodu [U]
        nodes[id] = node       # Nod nesnesi nodes içerisinde saklanıyor
        node.values = []

    def mean_value(node):
        return sum(node.values) / len(node.values)


class Element:
    def __init__(elm, id, conn, G=1): # Kayma modülü verilmediği durumda G=1 kabul edilmiştir
        elm.id = id
        elm.G, elm.w = G, 1  # w: kesit birim dönmesi değeri w=1 kabul edilmiştir
        elm.conn = [nodes[id] for id in conn]  # Bağlantı haritası [DN1 ... DN9]
        elements[id] = elm  # Eleman nesnesi elements içerisinde saklanıyor

    # Kod-Vektörü [C1 ... C9]
    def code(elm):
        return [n.code[0] for n in elm.conn]

    # Nodal koordinat matrisi [X] (2x9)
    def XM(elm):
        n1, n2, n3, n4, n5, n6, n7, n8, n9 = elm.conn
        return np.asarray([[n1.Y, n2.Y, n3.Y, n4.Y, n5.Y, n6.Y, n7.Y, n8.Y, n9.Y],
                           [n1.Z, n2.Z, n3.Z, n4.Z, n5.Z, n6.Z, n7.Z, n8.Z, n9.Z]])

    # Jacobian Matrisi [J] (2x2)
    def JM(elm, r, s):
        return elm.XM() @ dSF_dr(r, s)

    # Jacobian Matrisi Determinantı ABS(Det(JM))
    def detJM(elm, r, s):
        return ABS(DET(elm.JM(r, s)))

    # Şekil fonksiyonlarının gerçek koordinatlara göre türev matrisi
    def dSF_dx(elm, r, s):
        return dSF_dr(r, s) @ INV(elm.JM(r, s))

    # Rijitlik Matrisi [K] (9x9)
    def K(elm):
        def dK(r, s):  # Matrisin integrandı
            dF_dX = elm.dSF_dx(r, s)
            detJ = elm.detJM(r, s)
            return dF_dX @ dF_dX.T * detJ
        return IntegrateOn2DDomainWithGaussN3(dK)

    # Eleman sağ-taraf vektörü (9x1)
    def B(elm):
        def dB(r, s):  # Vetörün integrandı
            detJ = elm.detJM(r, s)
            SFV = SF(r, s)
            return 2 * SFV * detJ
        return IntegrateOn2DDomainWithGaussN3(dB)

    # solver a gönderilecek olan rijitlik matrisi
    def StiffnessMatrixToContribute(elm):
        return elm.K()

    # solver a gönderilecek olan RHS (sağ-taraf)
    def RHSVectorsToContribute(elm):
        return elm.B()

    # Hesaplanmış olan sistem serbestliklerini alıp
    # içinden ilgili elemana ait olanları ayırır ve elm.U da saklar
    def setSolution(elm, US):
        code = elm.code()
        elm.U = US[code]

    # Elemanın kesit burulma rijitliğine olan katkısını hesaplar
    def getIb(elm):
        def dIb(r, s):  # Vetörün integrandı
            detJ = elm.detJM(r, s)
            SFV = SF(r, s)
            XM = elm.XM()
            dF_dx = elm.dSF_dx(r, s)
            return -(SFV.T @ XM.T @ dF_dx.T) @ elm.U * detJ
        return IntegrateOn2DDomainWithGaussN3(dIb)

    # Kayma gerilmesi vektör alanı [TauXY(r,s), TauXZ(r,s)]
    def SigmaVec(elm, r, s):
        Gw = elm.G * elm.w
        dF_dx = elm.dSF_dx(r, s)
        return [Gw, -Gw] * (dF_dx.T @ elm.U)

    # Kayma gerilmesi vektörü nodal değerleri
    def SigmaAtNodes(elm):
        return [LEN(elm.SigmaVec(ri, si)) for ri, si in master_points]

    # verilen nodal değerleri nodlara aktarır
    def appendNodeValues(elm, nodal_values):
        for node, value in zip(elm.conn, nodal_values):
            node.values.append(value)


# Kesit boyutları ve eleman ağı tanımlamaları
L = 2  # Kare kesitin bir kenar uzunluğu
Ny, Nz = 32, 32    # X ve Y doğrultularında ağ bölüm sayıları

Ly, Lz = L, L      # X ve Y doğrultularının boyutları
dy, dz = Ly / (2 * Ny), Lz / (2 * Nz)

# Nodlar oluşturuluyor
id = 1
for zi in range(2 * Nz + 1):
    for yi in range(2 * Ny + 1):
        Node(id=id, Y=yi * dy, Z=zi * dz)
        id += 1

# Elemanlar oluşturuluyor
id = 1
for zi in range(Nz):
    for yi in range(Ny):
        n1 = 1 + 2 * yi + 2 * zi * (2 * Ny + 1)
        n2 = n1 + 2
        n3 = n2 + 2 * (2 * Ny + 1)
        n4 = n3 - 2
        n5 = n1 + 1
        n6 = n2 + (2 * Ny + 1)
        n7 = n3 - 1
        n8 = n6 - 2
        n9 = n8 + 1
        Element(id=id, conn=[n1, n2, n3, n4, n5, n6, n7, n8, n9])
        id += 1

# İlk ve son nodlar işaretleniyor
firstNode = nodes[1]
lastNode = nodes[len(nodes)]

# Dış sınırlarda nodlar tutuluyor
for id, node in nodes.items():
    if abs(node.Y - firstNode.Y) < 0.0001 or abs(node.Y - lastNode.Y) < 0.0001:
        node.rest = [1]
    if abs(node.Z - firstNode.Z) < 0.0001 or abs(node.Z - lastNode.Z) < 0.0001:
        node.rest = [1]

# Sistem denklemi oluşturulup çözüm yapılıyor
from solver import solve
US, PS = solve(nodes, elements)

# Yapılan çözüm elemanlara aktarılıyor
for key, elm in elements.items():
    elm.setSolution(US)

# Burulma rijitliği hesaplanıyor
Ib = sum(elm.getIb() for key, elm in elements.items())

# Elemanların nodal kayma gerilme değerleri nodlara aktarılıyor
for key, elm in elements.items():
    elm.appendNodeValues(elm.SigmaAtNodes())

#Kare kesit için analitik ve Sonlu Elemanlar çözümleri karşılaştırılıyor
print(f"\nKare kesit kenar uzunluğu (L) : {L}")
print(f"Ağ yapısı                       : {Ny}x{Nz}")

# Burulma rijitliği karşılaştırması
Ib_analytical = 2.25 * (L / 2) ** 4
print(f"\nIb (Analitik): {Ib_analytical}")
print(f"Ib (SEY)       : {Ib:.6f}")
print(f"Fark           : %{abs((Ib_analytical-Ib)/Ib_analytical)*100:.2f}")

# Maksimum kayma gerilmesi karşılaştırması
Tau_max_analytical = 4.808 * Ib_analytical / L ** 3  # Mb = G w Ib değeri için
Tau_max = max(node.mean_value() for id, node in nodes.items())
print(f"\nMaksimum Kayma Gerilmesi (Analitik): {Tau_max_analytical:.05f}")
print(f"Maksimum Kayma Gerilmesi (SEY)       : {Tau_max:.05f}")
print(f"Fark           : %{abs((Tau_max_analytical-Tau_max)/Tau_max_analytical)*100:.2f}")

# Bilinmeyen serbestlik sayısı (N): 3969
# Toplam serbestlik sayısı (M)    : 4225
#
# Kare kesit kenar uzunluğu (L) : 2
# Ağ yapısı                       : 32x32
#
# Ib (Analitik): 2.25
# Ib (SEY)       : 2.249231
# Fark           : %0.03
#
# Maksimum Kayma Gerilmesi (Analitik): 1.35225
# Maksimum Kayma Gerilmesi (SEY)     : 1.35009
# Fark           : %0.16


from drawing import Drawer, LineMaps, TriangleMaps


class Draw(Drawer):
    # Bir elemanı çizmek için gerekli üçgenlerin rölatif nod haritası
    trigs = TriangleMaps([0, 4, 8],
                         [0, 7, 8],
                         [1, 4, 8],
                         [1, 5, 8],
                         [2, 5, 8],
                         [2, 6, 8],
                         [3, 6, 8],
                         [3, 7, 8])

    # Bir elemanı çizmek için gerekli çizgilerin rölatif nod haritası
    lines =  LineMaps([0, 4], [4, 1],
                      [1, 5], [5, 2],
                      [2, 6], [6, 3],
                      [3, 7], [7, 0])


draw = Draw(elements=elements,
            nodes=nodes,
            on1=['X', 'Y', 'Z'],
            on2=['X', 'Y', 'Z'],
            connectivity_name="conn",
            mesh=False,
            name="|| Tau ||",
            lookat=[1, 0, 0])

draw.SigmaAtNodes()