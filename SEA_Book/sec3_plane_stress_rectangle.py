import numpy as np  # sayısal işlemler için python kütüphanesi

INV = np.linalg.inv  # matris ters alma fonksiyonu
np.set_printoptions(suppress=True)  # sayıların yazım (print) ayarları için

nodes = dict()  # Nodları hafızada tutacak koleksiyon
elements = dict()  # Elemanları hafızada tutacak koleksiyon


class Node:
    def __init__(node, id, X, Y):
        node.X, node.Y = X, Y  # Koordinatlar
        node.rest = [0, 0]  # Mesnet Koşulu [0: serbest, 1:tutulu]
        node.force = [0, 0]  # Tekil-Yük [Px, Py]
        node.disp = [0, 0]  # Mesnet Çökmesi [delta_x, delta_y]
        node.code = [-1, -1]  # Serbestlikler (Kod) [dx, dy]
        nodes[id] = node  # Nod nesnesi nodes içerisinde saklanıyor


class Element:
    def __init__(elm, id, conn, E, p, h):
        elm.conn = [nodes[id] for id in conn]  # Bağlantı [DN1, DN2, DN3, DN4]
        elm.E, elm.p, elm.h = E, p, h  # Malzeme ve kesit
        elm.boundaryForceX = [0] * 4  # Sınır-Yüzey X [q1x, q2x, q3x, q4x]
        elm.boundaryForceY = [0] * 4  # Sınır-Yüzey Y [q1y, q2y, q3y, q4y]
        elm.volumeForce = [0, 0]  # Hacim Kuvvetleri [bx, by]
        elements[id] = elm  # Eleman nesnesi elements içerisinde saklanıyor

    def ab(elm):
        n1, n2, n3, n4 = elm.conn  # Eleman düğüm noktaları conn içinden alınıyor
        return n2.X - n1.X, n3.Y - n1.Y  # a, b (eleman yatay ve düşey boyutları) hesaplanıyor

    def code(elm):  # Kod-Vektörü [u1, u2, u3, u4, v1, v2, v3, v4]
        return [n.code[0] for n in elm.conn] + \
               [n.code[1] for n in elm.conn]

    def B(elm):  # Eleman hacim-kuvvetleri vektörü (Be)
        h = elm.h
        a, b = elm.ab()
        bx, by = elm.volumeForce
        V = a * b / 4
        return h * V * np.asarray([bx, bx, bx, bx, by, by, by, by])

    def S(elm):  # Eleman sınır-yüzey dış yükleri vektörü (Se)
        a, b = elm.ab()
        q1x, q2x, q3x, q4x = elm.boundaryForceX
        q1y, q2y, q3y, q4y = elm.boundaryForceY
        return 0.5 * np.asarray([
            a * q1x + b * q3x, a * q1x + b * q4x, a * q2x + b * q3x, a * q2x + b * q4x,
            a * q1y + b * q3y, a * q1y + b * q4y, a * q2y + b * q3y, a * q2y + b * q4y,
        ])

    def K(elm):  # Eleman rijitlik matrisi (Ke)
        a, b = elm.ab()
        a2, b2 = a ** 2, b ** 2
        E, p, h = elm.E, elm.p, elm.h
        c1 = E / (1 - p ** 2)
        c2 = E / (2 + 2 * p)
        k0 = h / (12 * a * b)
        k1 = 2 * (a2 * c2 + b2 * c1)
        k2 = 2 * (a2 * c2 - 2 * b2 * c1)
        k3 = 2 * (2 * a2 * c2 - b2 * c1)
        k4 = 2 * (a2 * c1 + b2 * c2)
        k5 = 2 * (a2 * c1 - 2 * b2 * c2)
        k6 = 2 * (2 * a2 * c1 - b2 * c2)
        k7 = 3 * a * b * (c2 + p * c1)
        k8 = 3 * a * b * (c2 - p * c1)
        return k0 * np.asarray([
            [2 * k1, k2, -k3, -k1, k7, -k8, k8, -k7],
            [k2, 2 * k1, -k1, -k3, k8, -k7, k7, -k8],
            [-k3, -k1, 2 * k1, k2, -k8, k7, -k7, k8],
            [-k1, -k3, k2, 2 * k1, -k7, k8, -k8, k7],
            [k7, k8, -k8, -k7, 2 * k4, k5, -k6, -k4],
            [-k8, -k7, k7, k8, k5, 2 * k4, -k4, -k6],
            [k8, k7, -k7, -k8, -k6, -k4, 2 * k4, k5],
            [-k7, -k8, k8, k7, -k4, -k6, k5, 2 * k4]
        ])

    def deformed_shape(elm): return [0]*4


# Eleman Ağının oluşturulması
Lx, Ly = 0.80, 0.24  # Yatay ve düşey boyutlar [metre]
Nx, Ny = 8, 4  # Yatay ve düşey ağ bölme sayıları
dLx, dLy = Lx / Nx, Ly / Ny  # Yatay ve düşey boyut artımları
E = 70e6  # Elastisite modülü [kN/m^2]
p = 0.33  # Poisson oranı
h = 0.012  # Levha kalınlığı [m]
ro = 2.7  # Birim hacim kütlesi [gram/cm^3]=[ton/m^3]
g = 9.81  # m/sn^2
Px_tekil = 200  # Tekil-yük [kN]

for j in range(Ny + 1):  # Düğüm Noktaları oluşturuluyor
    for i in range(Nx + 1):
        Node(id=j * (Nx + 1) + i + 1, X=i * dLx, Y=j * dLy)

for j in range(Ny):  # Elemanlar oluşturuluyor
    for i in range(Nx):
        n1 = j * (Nx + 1) + i + 1
        n2 = n1 + 1
        n3 = n1 + Nx + 1
        n4 = n3 + 1
        Element(id=j * Nx + i + 1, conn=[n1, n2, n3, n4], E=E, p=p, h=h)

# Sınır koşullarının tanımlanması
nodes[2 * (Nx + 1)].force = [Px_tekil, 0]  # İlgili noda tekil-kuvvet tanımlanıyor

for id, node in nodes.items():
    if (node.X == 0): node.rest = [1, 1]  # X=0 nodlarının bilinen değerler olduğu belirleniyor
    if (node.X == 0): node.disp = [0.005 * node.Y, 0]  # X=0 sınırına mesnet çökmesi veriliyor

for id, elm in elements.items():
    elm.volumeForce = [0, -ro * g]  # Birim hacim kuvveti (Yerçekimi kaynaklı)
    if id in [29, 30, 31]:  # 29, 30 ve 31 nolu elemanın q2y leri (dış-sınır-yük)
        elm.boundaryForceY = [0, -100, 0, 0]

# Nodlarda tanımlı serbestlik numaralarının (code) belirlenmesi
M = 0  # M:Toplam serbestlik sayısı
N = 0  # N:Bilinmeyen serbestliklerin sayısı
for id, node in nodes.items():  # Tutulu olmayanlar (rest==0) numaralanıyor
    for index, rest in enumerate(node.rest):
        if rest == 0: node.code[index] = M; M += 1

N = M  # Bilinmeyen sayısı N de saklanıyor

for id, node in nodes.items():  # Tutulu olanlar (rest==1) numaralanıyor
    for index, rest in enumerate(node.rest):
        if rest == 1: node.code[index] = M; M += 1

# Sistem denklem takımının oluşturulması (Birleştirme)
KS = np.zeros((M, M))
US = np.zeros(M)
PS = np.zeros(M)
SS = np.zeros(M)
BS = np.zeros(M)

for id, elm in elements.items():  # Rijitlik matrisi
    code = elm.code()
    Ke = elm.K()
    dim = len(code)
    for i in range(dim):
        for j in range(dim):
            KS[code[i], code[j]] += Ke[i, j]

for id, elm in elements.items():
    code = elm.code()
    BS[code] += elm.B()  # Hacim kuvvetleri
    SS[code] += elm.S()  # Dış-sınır çizgisel yükleri

for id, node in nodes.items():
    code = node.code
    US[code] = node.disp  # Nodal mesnet-çökmeleri
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

U1 = INV(K11) @ (P1 + S1 + B1 - K12 @ U2)
P2 = K21 @ U1 + K22 @ U2 - S2 - B2

US = np.concatenate((U1, U2))
PS = np.concatenate((P1, P2))

for id, node in nodes.items():
    ux, uy = US[node.code]
    px, py = PS[node.code]
    print(f"Node {id}: [ux:{ux:.4f}, uy:{uy:.4f}] [px:{px:.2f}, py:{py:.2f}]")


print(elements[29].K())

#------------ Buradan sonraki işlemler kitapta yer almamıştır --------------#

# Yer değiştirmiş nod koordinatları hesaplanıyor (x=X+ux, y=Y+uy, z=Z+uz)
abs_ux_max = max(abs(US[node.code][0]) for id, node in nodes.items())
abs_uy_max = max(abs(US[node.code][1]) for id, node in nodes.items())
factor = 1 # Grafik büyütme faktörü
if abs_uy_max > 0: factor = 0.1 * Lx / abs_uy_max
for id, node in nodes.items():
    node.Z, node.z = 0, 0
    ux, uy = US[node.code]
    node.x = node.X + factor * ux
    node.y = node.Y + factor * uy


from drawing import Drawer, LineMaps, TriangleMaps

class Draw(Drawer):
    trigs = TriangleMaps([0, 1, 2],
                         [1, 2, 3])

    lines =  LineMaps([0, 1], [1, 3], [3, 2], [2, 0])


draw = Draw(elements=elements,
            nodes=nodes,
            on1=['x', 'y', 'z'],
            on2=['x', 'y', 'z'],
            connectivity_name="conn",
            mesh=True,
            name="Deformed Shape",
            lookat=[0, 0, 1])

draw.deformed_shape()
