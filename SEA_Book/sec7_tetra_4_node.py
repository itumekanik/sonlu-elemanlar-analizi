import numpy as np  # sayısal işlemler için Python kütüphanesi

INV = np.linalg.inv  # matris ters alma fonksiyonu
DET = np.linalg.det  # matris determinant alma fonksiyonu
LEN = np.linalg.norm  # vektör boy hesabı için kullanılacak
np.set_printoptions(suppress=True)  # sayıların yazım (print) ayarları için

# Master eleman nod koordinatları [p1...p8]
master_points = [[ 0, 0, 0],
                 [ 1, 0, 0],
                 [ 0, 1 ,0],
                 [ 0, 0, 1]]

# 3D tetrahedral bölgede tek noktalı integrasyon kuralı
def IntegrateOn3DDomainWithQuadN1(h):
    return h(0.25, 0.25, 0.25) / 6

# Tetrahedralin 4 sınır yüzeyinde tek noktalı integral alınıyor (w=1/2)
def IntegrateOn3DTriangeAreas(h):
    total = 0
    for k in range(4):
        if k == 0: r, s, t = 0, 1/3, 1/3
        if k == 1: r, s, t = 1/3, 0, 1/3
        if k == 2: r, s, t = 1/3, 1/3, 0
        if k == 3: r, s, t = 1/3, 1/3, 1/3
        total += 0.5 * h(r, s, t, k)
    return total

# Şekil fonksiyonları vektörü (4x1)
def SF(r, s, t):
    return np.asarray([1-r-s-t, r, s, t])

# Şekil fonksiyonlarının türev matrisi (4x3)
def dF_dr(r, s, t):
    return np.asarray([[-1, -1, -1],
                       [ 1,  0,  0],
                       [ 0,  1,  0],
                       [ 0,  0,  1]])

nodes = dict()     # Nodları hafızada tutacak konteyner
elements = dict()  # Elemanları hafızada tutacak konteyner

class Node:
    def __init__(node, id, X, Y, Z):
        node.id = id
        node.X, node.Y, node.Z = X, Y, Z  # Koordinatlar
        node.rest = [0, 0, 0]     # Mesnet Koşulu [0: serbest, 1:tutulu]
        node.force = [0, 0, 0]    # Tekil-Yük [Px, Py, Pz]
        node.disp = [0, 0, 0]     # Mesnet Çökmesi [delta_x, delta_y, delta_z]
        node.code = [-1, -1, -1]  # Serbestlikler (Kod) [dx, dy, dz]
        nodes[id] = node  # Nod nesnesi nodes içerisinde saklanıyor
        node.values = []

    def mean_value(node):
        return sum(node.values) / len(node.values)

    def ExternalForceVectorToContribute(node):
        return node.force

class Element:
    def __init__(elm, id, conn, E, p, alpha=0.0):
        elm.id = id
        elm.conn = [nodes[id] for id in conn]  # Bağlantı haritası [DN1 ... DN4]
        elm.E, elm.p, elm.alpha = E, p, alpha  # Malzeme sabitleri
        elm.boundaryForceX = [0] * 4  # Sınır-Yüzey X [q1x, q2x, q3x, q4x]
        elm.boundaryForceY = [0] * 4  # Sınır-Yüzey Y [q1y, q2y, q3y, q4y]
        elm.boundaryForceZ = [0] * 4  # Sınır-Yüzey Z [q1z, q2z, q3z, q4z]
        elm.volumeForce = [0] * 3     # Hacim Kuvvetleri [bx, by, bz]
        elm.temperatureChange = 0     # Uniform sıcaklık değişimi (delta_T)
        elements[id] = elm  # Eleman nesnesi elements içerisinde saklanıyor

    # Kod-Vektörü [u1 ... u4, v1 ... v4, w1 ... w4]
    def code(elm):
        return [n.code[0] for n in elm.conn] + \
               [n.code[1] for n in elm.conn] + \
               [n.code[2] for n in elm.conn]

    # Nodal koordinat matrisi (3x4)
    def XM(elm):
        n1, n2, n3, n4 = elm.conn
        return np.asarray([[n1.X, n2.X, n3.X, n4.X],
                           [n1.Y, n2.Y, n3.Y, n4.Y],
                           [n1.Z, n2.Z, n3.Z, n4.Z]])

    # Jacobian Matrisi
    def JM(elm, r, s, t):
        return elm.XM() @ dF_dr(r, s, t)

    # Jacobian (Det(JM))
    def detJM(elm, r, s, t):
        return abs(DET(elm.JM(r, s, t)))

    # Şekil fonksiyonlarının gerçek koordinatlara göre türev matrisi
    def dF_DX(elm, r, s, t):
        return dF_dr(r, s, t) @ INV(elm.JM(r, s, t))

    # Genleme-yer değiştirme matrisi (6x12)
    def BM(elm, r, s, t):
        dF_dX = elm.dF_DX(r, s, t)
        B = np.zeros((6, 12))
        B[0, 0:4] = dF_dX[:, 0]
        B[1, 4:8] = dF_dX[:, 1]
        B[2, 8:12] = dF_dX[:, 2]
        B[3, 4:8] = dF_dX[:, 2]
        B[3, 8:12] = dF_dX[:, 1]
        B[4, 0:4] = dF_dX[:, 2]
        B[4, 8:12] = dF_dX[:, 0]
        B[5, 0:4] = dF_dX[:, 1]
        B[5, 4:8] = dF_dX[:, 0]
        return B

    # Bünye (malzeme) matrisi [C] (6x6)
    def C(elm):
        E, p = elm.E, elm.p
        return E / ((1 + p) * (1 - 2 * p)) * \
               np.asarray([[1 - p, p, p, 0, 0, 0],
                           [p, 1 - p, p, 0, 0, 0],
                           [p, p, 1 - p, 0, 0, 0],
                           [0, 0, 0, 0.5 - p, 0, 0],
                           [0, 0, 0, 0, 0.5 - p, 0],
                           [0, 0, 0, 0, 0, 0.5 - p]])

    # Rijitlik Matrisi [K] (12x12)
    def K(elm):
        def dK(r, s, t):  # Rijitlik Matrisi integrandı
            C = elm.C()
            B = elm.BM(r, s, t)
            J = elm.detJM(r, s, t)
            return B.T @ C @ B * J
        return IntegrateOn3DDomainWithQuadN1(dK)

    # Eleman hacim-kuvvetleri vektörü (12x1)
    def B(elm):
        def dB(r, s, t):  # Vetörün integrandı
            bx, by, bz = elm.volumeForce
            if bx == 0 and by == 0 and bz == 0: return np.zeros(12)
            J = elm.detJM(r, s, t)
            SFV = SF(r, s, t)
            SF12 = np.concatenate((SFV, SFV, SFV))
            return J * SF12 * ([bx] * 4 + [by] * 4 + [bz] * 4)
        return IntegrateOn3DDomainWithQuadN1(dB)

    # Eleman sınır-yüzey dış yükleri vektörü (12x1)
    def S(elm):
        def dS(r, s, t, k):  # Vetörün integrandı
            qx = elm.boundaryForceX[k]
            qy = elm.boundaryForceY[k]
            qz = elm.boundaryForceZ[k]
            if qx == 0 and qy == 0 and qz == 0: return np.zeros(12)
            SFV = SF(r, s, t)
            surfaceMap = [[1, 3, 4], [1, 4, 2], [1, 2, 3], [2, 3, 4]]
            XM = elm.XM()
            Xa = XM[:, surfaceMap[k][0] - 1]
            Xb = XM[:, surfaceMap[k][1] - 1]
            Xc = XM[:, surfaceMap[k][2] - 1]
            J = LEN(np.cross(Xb - Xa, Xc - Xa))
            return J * np.concatenate((SFV * qx, SFV * qy, SFV * qz))
        return IntegrateOn3DTriangeAreas(dS)

    # Eleman sıcaklık değişimi vektörü (12x1)
    def T(elm):
        def dT(r, s, t):  # Vetörün integrandı
            alpha = elm.alpha
            deltaT = elm.temperatureChange
            if alpha == 0 or deltaT == 0: return np.zeros(12)
            ALPHA = np.asarray([alpha, alpha, alpha, 0, 0, 0])
            C = elm.C()
            B = elm.BM(r, s, t)
            J = elm.detJM(r, s, t)
            return B.T @ C @ ALPHA * deltaT * J
        return IntegrateOn3DDomainWithQuadN1(dT)

    # solver a gönderilecek olan rijitlik matrisi
    def StiffnessMatrixToContribute(elm):
        return elm.K()

    # solver a gönderilecek olan sağ taraf (RHS)
    def RHSVectorsToContribute(elm):
        return elm.B() + elm.S() + elm.T()

    # Hesaplanmış olan sistem serbestliklerini (yer-değiştirmeleri) alıp
    # içinden ilgili elemana ait olanları ayırır ve elm.U da saklar
    def setSolution(elm, US):
        code = elm.code()
        elm.U = US[code]

    # Gerilme vektörü
    def SigmaVec(elm, r, s, t):
        U = elm.U
        C = elm.C()
        BM = elm.BM(r, s, t)
        alpha = elm.alpha
        deltaT = elm.temperatureChange
        ALPHA = np.asarray([alpha, alpha, alpha, 0, 0, 0])
        return C @ (BM @ U - ALPHA * deltaT)

    # verilen nodal değerleri nodlara paslar
    def appendNodeValues(elm, nodal_values):
        for i, node in enumerate(elm.conn):
            node.values.append(nodal_values[i])

    def SigmaX(elm): return [elm.SigmaVec(ri, si, ti)[0]
                             for ri, si, ti in master_points]

    def SigmaXAverage(elm): return [node.mean_value()
                                    for node in elm.conn]


# ----------------------------
# Eleman ağının oluşturulması
# ----------------------------
#
E = 28.0e6  # [kN/m^2]
p = 0.22    # Poisson oranı
unit_weight = 24.0  # [kN/m^3]
thermal_expansion_coefficient = 1.0e-5  # [1/Celsius]

weight_factor = 1
temperature_change = 20.0    # [Celsius]
top_surface_load_z = -150.0  # [kN/m^2]

Lx, Ly, Lz = 6, 0.2, 0.4
Nx, Ny, Nz = 30, 2, 4
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

id = 1
for zi in range(Nz + 1):
    for yi in range(Ny + 1):
        for xi in range(Nx + 1):
            Node(id=id, X=xi * dx, Y=yi * dy, Z=zi * dz)
            id += 1

# Tuğla Elemanı içinden 6 adet Tetrahedral Eleman çıkar
# Aşağıdaki değişken bu tetrahedralleri oluştural tuğla
# eleman nod numaraları listesidir
brickToTetraMap = [[0, 5, 1, 6],
                   [0, 5, 4, 6],
                   [0, 6, 4, 7],
                   [0, 6, 2, 1],
                   [0, 6, 2, 3],
                   [0, 6, 3, 7]]

id = 1
for zi in range(Nz):
    for yi in range(Ny):
        for xi in range(Nx):
            n1 = (xi + 1) + yi * (Nx + 1) + zi * (Nx + 1) * (Ny + 1)
            n2 = n1 + 1
            n3 = n2 + (Nx + 1)
            n4 = n3 - 1
            n5 = n1 + (Nx + 1) * (Ny + 1)
            n6 = n5 + 1
            n7 = n6 + (Nx + 1)
            n8 = n7 - 1
            conn = np.asarray([n1, n2, n3, n4, n5, n6, n7, n8], dtype=int)
            for map in brickToTetraMap:
                Element(id=id, conn=conn[map],
                        E=E, p=p, alpha=thermal_expansion_coefficient)
                id += 1

# Hacim, sıcaklık ve dış sınır-yüzey yüklemeleri yapılıyor
for id, elm in elements.items():
    elm.volumeForce = [0, 0, -unit_weight * weight_factor]
    elm.temperatureChange = temperature_change
    n1, n2, n3, n4 = elm.conn
    if abs(n2.Z - Lz) < 0.000001 and \
       abs(n3.Z - Lz) < 0.000001 and \
       abs(n4.Z - Lz) < 0.000001:
        elm.boundaryForceZ = [0, 0, 0, top_surface_load_z]


# Sağ ve sol nodlar mesnetleniyor
for id, node in nodes.items():
    if abs(node.X - 0) < 0.0001 or abs(node.X - Lx) < 0.0001:
        node.rest = [1, 1, 1]

# ----------------------------------------------------------------
# Sistem denklem takımının oluşturulması, çözümü ve diğer işlemler
# ----------------------------------------------------------------
#
# Sistem denklem takımının birleştirilmesi ve çözümü
from solver import solve
US, PS = solve(nodes, elements)

# Çözümün elemanlar üzerine geri yansıtılması ve Gx değerlerinin
# nodlarda ortalanması
for key, elm in elements.items():
    elm.setSolution(US)
    sigmaX_At_Nodes = [elm.SigmaVec(ri, si, ti)[0]
                       for ri, si, ti in master_points]
    elm.appendNodeValues(sigmaX_At_Nodes)

# -------------------------------
# Çözüm çıktılarının yazdırılması
# -------------------------------
#
# Boyut ve Mesh bilgileri
print(f"\nÇubuk boyutları: Lx: {Lx}, Ly: {Ly}, Lz: {Lz}")
print(f"Ağ yapısı: Nx: {Nx}, Ny: {Ny}, Nz: {Nz}")

# Hesaplanan mesnet tekpkilerinin toplamlarının
# Dış Yük toplamları ile karşılaştırılması
RestraintSum = np.sum([PS[node.code] for id, node in nodes.items()], axis=0)
print("\n*** Mesnet Tepkileri Toplamı ***")
print(f"Rx: {RestraintSum[0]:0.4f}")
print(f"Ry: {RestraintSum[1]:0.4f}")
print(f"Rz: {RestraintSum[2]:0.4f}\n")

surfaceLoadSum = Lx * Ly * top_surface_load_z
volumeLoadSum = -weight_factor * unit_weight * Lx * Ly * Lz
total = surfaceLoadSum + volumeLoadSum
print("*** Z doğrultusundaki Dış Yük Toplamları ***")
print(f"Dış-sınır yüzey yüklemesi toplamı: {surfaceLoadSum:0.4f}")
print(f"Hacim yüklemesi toplamı          : {volumeLoadSum:0.4f}")
print(f"Toplam Dış yükler                : {total:0.4f}\n")

# Sistemde Z yönünde oluşan maksimum yer-değiştirme bulunuyor
abs_uz_max = max(abs(US[node.code][2]) for id, node in nodes.items())
print(f"*** z doğrultusundaki en büyük yer-değiştirme ***")
print(f"uz_max: {abs_uz_max:.9f}\n")

# Gx gerilme alanı ekstremum değerleri hesaplanıyor
# Eleman uç noktalarında minimum ve maksimum Gx değerleri hesaplanıyor
print(f"*** SigmaX maksimum ve minimum gerilme hesabı ***")
min_Gx_elm = min(min(elm.SigmaVec(ri, si, ti)[0]
                     for ri, si, ti in master_points)
                     for id, elm in elements.items())

max_Gx_elm = max(max(elm.SigmaVec(ri, si, ti)[0]
                      for ri, si, ti in master_points)
                      for id, elm in elements.items())

print("Gx eleman (min):", min_Gx_elm)
print("Gx eleman (max):", max_Gx_elm)

# Nodlarda ortalaması alınmış Gx değerlerinin minimum ve maksimumu hesaplanıyor
min_Gx_node_average = min(node.mean_value() for id, node in nodes.items())
max_Gx_node_average = max(node.mean_value() for id, node in nodes.items())
print("Gx nod ortalama (min):", min_Gx_node_average)
print("Gx nod ortalama (max):", max_Gx_node_average)

# Yer değiştirmiş nod koordinatları hesaplanıyor (x=X+ux, y=Y+uy, z=Z+uz)
factor = 1 # Grafik büyütme faktörü
if abs_uz_max > 0: factor = 0.1 * Lx / abs_uz_max
for id, node in nodes.items():
    ux, uy, uz = US[node.code]
    node.x = node.X + factor * ux
    node.y = node.Y + factor * uy
    node.z = node.Z + factor * uz


from drawing import Drawer, LineMaps, TriangleMaps


class Draw(Drawer):
    trigs = TriangleMaps([0, 2, 3],
                         [0, 3, 1],
                         [0, 1, 2],
                         [1, 2, 3])

    lines =  LineMaps([0, 1], [0, 2], [0, 3],
                      [1, 2], [2, 3], [3, 1])


draw = Draw(elements=elements,
            nodes=nodes,
            on1=['x', 'y', 'z'],
            on2=['x', 'y', 'z'],
            connectivity_name="conn",
            mesh=True,
            name="SigmaX",
            lookat=[0, 1, 0])

draw.SigmaXAverage()

# Bilinmeyen serbestlik sayısı (N): 1305
# Toplam serbestlik sayısı (M)    : 1395
#
# Çubuk boyutları: Lx: 6, Ly: 0.2, Lz: 0.4
# Ağ yapısı: Nx: 30, Ny: 2, Nz: 4
#
# *** Mesnet Tepkileri Toplamı ***
# Rx: 0.0001
# Ry: -0.0000
# Rz: 191.5200
#
# *** Z doğrultusundaki Dış Yük Toplamları ***
# Dış-sınır yüzey yüklemesi toplamı: -180.0000
# Hacim yüklemesi toplamı          : -11.5200
# Toplam Dış yükler                : -191.5200
#
# *** z doğrultusundaki en büyük yer-değiştirme ***
# uz_max: 0.002664659
#
# *** SigmaX maksimum ve minimum gerilme hesabı ***
# Gx eleman (min): -19748.506603249218
# Gx eleman (max): 6607.998225907715
# Gx nod ortalama (min): -19641.957740150257
# Gx nod ortalama (max): 5541.988532093969

# draw[lambda elm:abs(np.mean(elm.SigmaXAverage())) <= 5000].SigmaXAverage()
# draw.SigmaX()