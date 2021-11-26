import numpy as np  # sayısal işlemler için python kütüphanesi

INV = np.linalg.inv  # matris ters alma fonksiyonu
DET = np.linalg.det  # matris determinant alma fonksiyonu
np.set_printoptions(suppress=True)  # sayıların yazım (print) ayarları için


# 2D bölgede n=2 Gauss integrasyon şeması
def IntegrateOn2DDomainWithGaussN2(h):
    total = 0
    p = [-1 / 3 ** 0.5, 1 / 3 ** 0.5]
    w = [1, 1]
    for i in range(2):
        for j in range(2):
            total += w[i] * w[j] * h(p[i], p[j])
    return total


# 2D sınırlarda n=2 Gauss integrasyon şeması
def IntegrateOn2DBoundariesWithGaussN2(h):
    total = 0
    p = [-1 / 3 ** 0.5, 1 / 3 ** 0.5]
    w = [1, 1]
    for i in range(2):
        rb = [p[i], p[i], -1, 1]
        sb = [-1, 1, p[i], p[i]]
        for k in range(4):
            total += w[i] * h(rb[k], sb[k], k)
    return total


def SF(r, s):  # Şekil fonksiyonları vektörü
    return 0.25 * np.asarray([(1 - r) * (1 - s),
                              (1 + r) * (1 - s),
                              (1 - r) * (1 + s),
                              (1 + r) * (1 + s)])


def dSF_dr(r, s):  # Şekil fonksiyonlarının türev matrisi (4x2)
    return 0.25 * np.asarray([[-1 + s, -1 + r],
                              [1 - s, -1 - r],
                              [-1 - s, 1 - r],
                              [1 + s, 1 + r]])


nodes = dict()  # Nodları hafızada tutacak konteyner
elements = dict()  # Elemanları hafızada tutacak konteyner


class Node:
    def __init__(node, id, X, Y):
        node.X, node.Y = X, Y  # Koordinatlar
        node.rest = [0, 0]  # Mesnet Koşulu [0: serbest, 1:tutulu]
        node.force = [0, 0]  # Tekil-Yük [Px, Py]
        node.disp = [0, 0]  # Mesnet Çökmesi [delta_x, delta_y]
        node.code = [-1, -1]  # Serbestlikler (Kod) [dx, dy]
        nodes[id] = node  # Nod nesnesi nodes içerisinde saklanıyor
        node.values = []

    def get_draw_value(node):
        return sum(node.values) / len(node.values)

class Element:
    def __init__(elm, id, conn, E, p, h):
        elm.conn = [nodes[id] for id in conn]  # Bağlantı [DN1, DN2, DN3, DN4]
        elm.E, elm.p, elm.h = E, p, h  # Malzeme ve kesit
        elm.boundaryForceX = [0] * 4  # Sınır-Yüzey X [q1x, q2x, q3x, q4x]
        elm.boundaryForceY = [0] * 4  # Sınır-Yüzey Y [q1y, q2y, q3y, q4y]
        elm.volumeForce = [0, 0]  # Hacim Kuvvetleri [bx, by]
        elements[id] = elm  # Eleman nesnesi elements içerisinde saklanıyor

    def code(elm):  # Kod-Vektörü [u1, u2, u3, u4, v1, v2, v3, v4]
        return [n.code[0] for n in elm.conn] + \
               [n.code[1] for n in elm.conn]

    def XM(elm):  # Nodal koordinat matrisi (2x4)
        n1, n2, n3, n4 = elm.conn
        return np.asarray([[n1.X, n2.X, n3.X, n4.X],
                           [n1.Y, n2.Y, n3.Y, n4.Y]])

    def JM(elm, r, s):  # Jacobian Matrisi
        return elm.XM() @ dSF_dr(r, s)

    def detJM(elm, r, s):  # Jacobian (Det(JM))
        return DET(elm.JM(r, s))

    # Şekil fonksiyonlarının gerçek koordinatlara göre türev matrisi
    def dSF_Dx_T(elm, r, s):
        return INV(elm.JM(r, s)).T @ dSF_dr(r, s).T

    def BM(elm, r, s):  # Genleme-yer değiştirme matrisi (3x8)
        empty = np.zeros((3, 8))
        mat = elm.dSF_Dx_T(r, s)
        empty[0, 0:4] = mat[0]
        empty[1, 4:8] = mat[1]
        empty[2, 0:4] = mat[1]
        empty[2, 4:8] = mat[0]
        return empty

    def C(elm):  # Bünye (malzeme) matrisi (3x3)
        E, p = elm.E, elm.p
        return E / (1 - p ** 2) * np.asarray([[1, p, 0],
                                              [p, 1, 0],
                                              [0, 0, 0.5 * (1 - p)]])

    def K(elm):  # Rijitlik Matrisi
        def dK(r, s):  # Rijitlik Matrisi integradı
            h = elm.h
            C = elm.C()
            BM = elm.BM(r, s)
            J = elm.detJM(r, s)
            return h * BM.T @ C @ BM * J

        return IntegrateOn2DDomainWithGaussN2(dK)

    def B(elm):  # Eleman hacim-kuvvetleri vektörü (Be)
        def dB(r, s):  # Vetörün İntegrandı
            bx, by = elm.volumeForce
            if (bx == 0 and by == 0): return np.zeros(8)
            h = elm.h
            J = elm.detJM(r, s)
            SFV = SF(r, s)
            SF8 = np.concatenate((SFV, SFV))
            return h * J * SF8 * [bx, bx, bx, bx, by, by, by, by]

        return IntegrateOn2DDomainWithGaussN2(dB)

    def S(elm):  # Eleman sınır-yüzey dış yükleri vektörü (Se)
        def dS(r, s, k):
            qx = elm.boundaryForceX[k]
            qy = elm.boundaryForceY[k]
            if (qx == 0 and qy == 0): return np.zeros(8)
            SFV = SF(r, s)
            JM = elm.JM(r, s)
            JTJ = JM.T @ JM
            J = JTJ[0, 0] ** 0.5 if k in [0, 1] else JTJ[1, 1] ** 0.5
            return J * np.concatenate((SFV * qx, SFV * qy))

        return IntegrateOn2DBoundariesWithGaussN2(dS)

    def setDisp(elm, US):
        code = elm.code()
        elm.disp = US[code]

    def SigmaVec(elm, r, s):
        C = elm.C()
        BM = elm.BM(r, s)
        U = elm.disp
        return C @ BM @ U

    def appendNodeValues(elm, v1, v2, v3, v4):
        n1, n2, n3, n4 = elm.conn
        n1.values.append(v1)
        n2.values.append(v2)
        n3.values.append(v3)
        n4.values.append(v4)

    def SigmaXAverage(elm): return [node.get_draw_value()
                                    for node in elm.conn]


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

for key, elm in elements.items():
    elm.setDisp(US)
    Gx1, Gy1, Txy1 = elm.SigmaVec(-1, -1)
    Gx2, Gy2, Txy2 = elm.SigmaVec(1, -1)
    Gx3, Gy3, Txy3 = elm.SigmaVec(-1, 1)
    Gx4, Gy4, Txy4 = elm.SigmaVec(1, 1)
    elm.appendNodeValues(Gx1, Gx2, Gx3, Gx4)

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
            name="SigmaX",
            lookat=[0, 0, 1])

draw.SigmaXAverage()
