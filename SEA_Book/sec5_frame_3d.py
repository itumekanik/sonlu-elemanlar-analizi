import numpy as np   # sayısal işlemler için python kütüphanesi
INV = np.linalg.inv  # matris ters alma fonksiyonu
DET = np.linalg.det  # matris determinant alma fonksiyonu
np.set_printoptions(suppress=True)  # sayıların yazım (print) ayarları için

materials = dict()  # Malzemeleri hafızada tutacak konteyner
sections = dict()   # Kesitleri hafızada tutacak konteyner
nodes = dict()      # Nodları hafızada tutacak konteyner
elements = dict()   # Elemanları hafızada tutacak konteyner


class Material:
    def __init__(mat, id, E, p):
        mat.id = id
        mat.E = E  # Elastisite modülü
        mat.p = p  # Poisson oranı
        mat.G = 0.5 * E / (1 + p)  # Kayma modülü
        materials[id] = mat

    def props(mat):
        return [mat.E, mat.p, mat.G]


class Section:
    def __init__(sec, id, A, Ix, Iy, Iz, Iyz):
        sec.id = id
        sec.A = A  # Alan
        sec.Ix = Ix  # Burulma atalet momenti
        sec.Iy = Iy  # y ekseni etrafında eğilme atalet momenti
        sec.Iz = Iz  # z ekseni etrafında eğilme atalet momenti
        sec.Iyz = Iyz  # asimetrik kesit eğilme atalet momenti
        sections[id] = sec

    def props(sec):
        return [sec.A, sec.Ix, sec.Iy, sec.Iz, sec.Iyz]


class NodeFrame3D:
    def __init__(node, id, X, Y, Z):
        node.id = id
        node.X, node.Y, node.Z = X, Y, Z  # Koordinatlar
        node.rest = [0] * 6  # Mesnet Koşulu [0: serbest, 1:tutulu]
        node.disp = [0] * 6  # Mesnet Çökmesi [cx, cy, cx, ctx, cty, ctz]
        node.code = [-1] * 6  # Serbestlikler (Kod) [dx, dy, dz, tx, ty, tz]
        node.EulerZYX = [0] * 3  # Lokal eksen Euler açıları [betaZ, beyaY, betaX]
        node.GlobalForces = [0] * 6  # Global Tekil-Yükler [Px, Py, Pz, Mx, My, Mz]
        nodes[id] = node  # Nod nesnesi nodes içerisinde saklanıyor

    def getNodeLokalAxesTransformation(node):  # ZYX Euler dönüşüm matrisi
        ezd, eyd, exd = node.EulerZYX
        rx, ry, rz = np.pi / 180 * np.asarray([exd, eyd, ezd])
        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])
        return np.asarray([[cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
                           [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
                           [-sy, cy * sx, cx * cy]])

    @property
    def force(node):  # Sistem denklemine katkı yapacak .force vektörü hesaplanıyor.
        Px, Py, Pz, Mx, My, Mz = node.GlobalForces
        ET_INV = INV(node.getNodeLokalAxesTransformation())
        p = ET_INV @ [Px, Py, Pz]
        m = ET_INV @ [Mx, My, Mz]
        return [*p, *m]


class ElementFrame3D:  # omega açısı derece olarak verilmeli
    def __init__(elm, id, conn, mat, sec, omega=0, q=[0] * 6):
        elm.id = id
        elm.conn = [nodes[id] for id in conn]  # Bağlantı haritası [n1, n2]
        elm.mat = materials[mat]  # Malzeme
        elm.sec = sections[sec]  # Kesit
        elm.omega = np.pi / 180 * omega  # Kesit duruş açısı (n1 etrafında dönüş)
        elm.q = q  # Yayılı yükler [qx, qy, qz, mx, my, mz]
        elements[id] = elm  # Eleman nesnesi elements içerisinde saklanıyor

    def code(elm):  # Kod-Vektörü [u1,v1,w1,t1x,t1y,t1z,u2,v2,w2,t2x,t2y,t2z]
        n1, n2 = elm.conn
        return n1.code + n2.code

    def nx_ny_nz_L(elm):  # Doğrultman kosinüsleri ve eleman boyu
        n1, n2 = elm.conn
        Lx, Ly, Lz = n2.X - n1.X, n2.Y - n1.Y, n2.Z - n1.Z
        L = (Lx ** 2 + Ly ** 2 + Lz ** 2) ** 0.5
        nx, ny, nz = Lx / L, Ly / L, Lz / L
        return nx, ny, nz, L

    def TLG(elm):
        omg = elm.omega
        TOMG = np.asarray([[1, 0, 0],
                           [0, np.cos(omg), np.sin(omg)],
                           [0, -np.sin(omg), np.cos(omg)]])

        nx, ny, nz, L = elm.nx_ny_nz_L()
        TALFA = np.identity(3)
        if np.abs(1 - nz ** 2) < 0.001:
            TALFA = np.asarray([[nx, ny, nz],
                                [1, 0, 0],
                                [0, nz, -ny]])
        else:
            a = 1 / (1 - nz ** 2)
            TALFA = np.asarray([[nx, ny, nz],
                                [-a * nx * nz, -a * ny * nz, 1],
                                [a * ny, -a * nx, 0]])

        n1, n2 = elm.conn
        TBETA1 = n1.getNodeLokalAxesTransformation()
        TBETA2 = n2.getNodeLokalAxesTransformation()

        _TLG = np.identity(12)
        _TLG[0:3, 0:3] = TOMG @ TALFA @ TBETA1
        _TLG[3:6, 3:6] = TOMG @ TALFA @ TBETA1
        _TLG[6:9, 6:9] = TOMG @ TALFA @ TBETA2
        _TLG[9:12, 9:12] = TOMG @ TALFA @ TBETA2

        return _TLG

    def K_Local(elm):  # Lokal rijitlik matrisi
        E, p, G = elm.mat.props()
        A, Ix, Iy, Iz, Iyz = elm.sec.props()
        EA, GIx, EIy, EIz, EIyz = E * A, G * Ix, E * Iy, E * Iz, E * Iyz
        nx, ny, nz, L = elm.nx_ny_nz_L()
        L2, L3 = L ** 2, L ** 3
        ku = EA / L
        ktx = GIx / L
        k1z, k1y, k1yz = 2 * EIz / L, 2 * EIy / L, 2 * EIyz / L
        k2z, k2y, k2yz = 6 * EIz / L2, 6 * EIy / L2, 6 * EIyz / L2
        k3z, k3y, k3yz = 12 * EIz / L3, 12 * EIy / L3, 12 * EIyz / L3
        return np.asarray([
            [ku, 0, 0, 0, 0, 0, -ku, 0, 0, 0, 0, 0],
            [0, k3z, k3yz, 0, -k2yz, k2z, 0, -k3z, -k3yz, 0, -k2yz, k2z],
            [0, k3yz, k3y, 0, -k2y, k2yz, 0, -k3yz, -k3y, 0, -k2y, k2yz],
            [0, 0, 0, ktx, 0, 0, 0, 0, 0, -ktx, 0, 0],
            [0, -k2yz, -k2y, 0, 2 * k1y, -2 * k1yz, 0, k2yz, k2y, 0, k1y, -k1yz],
            [0, k2z, k2yz, 0, -2 * k1yz, 2 * k1z, 0, -k2z, -k2yz, 0, -k1yz, k1z],
            [-ku, 0, 0, 0, 0, 0, ku, 0, 0, 0, 0, 0],
            [0, -k3z, -k3yz, 0, k2yz, -k2z, 0, k3z, k3yz, 0, k2yz, -k2z],
            [0, -k3yz, -k3y, 0, k2y, -k2yz, 0, k3yz, k3y, 0, k2y, -k2yz],
            [0, 0, 0, -ktx, 0, 0, 0, 0, 0, ktx, 0, 0],
            [0, -k2yz, -k2y, 0, k1y, -k1yz, 0, k2yz, k2y, 0, 2 * k1y, -2 * k1yz],
            [0, k2z, k2yz, 0, -k1yz, k1z, 0, -k2z, -k2yz, 0, -2 * k1yz, 2 * k1z]])

    def K(elm):  # Global rijitlik matrisi
        T = elm.TLG()
        return INV(T) @ elm.K_Local() @ T

    def q_Local(elm):  # Lokal yayılı yük vektörü
        qx, qy, qz, mx, my, mz = elm.q
        nx, ny, nz, L = elm.nx_ny_nz_L()
        return [0.5 * qx * L, 0.5 * qy * L - mz, 0.5 * qz * L + my, 0.5 * mx * L, -qz * L ** 2 / 12, qy * L ** 2 / 12,
                0.5 * qx * L, 0.5 * qy * L + mz, 0.5 * qz * L - my, 0.5 * mx * L, qz * L ** 2 / 12, -qy * L ** 2 / 12]

    def B(elm):  # Birleştirilecek yayılı yük vektörü INV(T) @ q_Lokal
        T = elm.TLG()
        return INV(T) @ elm.q_Local()


Material(id="steel", E=210e6, p=0.3)
Section(id="L", A=0.002364, Ix=0.00000002805048,
        Iy=0.00000458124835, Iz=0.00001597484835, Iyz=0.00000501624365)

NodeFrame3D(id="A", X=-3, Y=-3, Z=0)
NodeFrame3D(id="B", X=3, Y=3, Z=0)
NodeFrame3D(id="C", X=3, Y=-3, Z=3)

ElementFrame3D(id=1, conn=["A", "C"], mat="steel", sec="L", omega=20)
ElementFrame3D(id=2, conn=["B", "C"], mat="steel", sec="L")

nodes["C"].EulerZYX = [30, 20, 10]

nodes["A"].rest = [1, 1, 1, 1, 1, 1]
nodes["B"].rest = [1, 1, 1, 1, 1, 1]

nodes["C"].GlobalForces = [0, 0, -10, 0, 0, 0]

#------------ Buradan sonraki işlemler kitapta yer almamıştır --------------#

# Sistem denklemi oluşturulup çözüm yapılıyor
from solver import solve
US, PS = solve(nodes, elements)

# Sonuçlar ekrana basılıyor
for id, node in nodes.items():
    ux, uy, uz, tx, ty, tz = US[node.code]
    print(f"Node {id}: [ux:{ux:.8f}, uy:{uy:.8f}, uz:{uz:.8f}]")
    print(f"Node {id}: [tx:{tx:.8f}, ty:{ty:.8f}, tz:{tz:.8f}]")

for id, node in nodes.items():
    px, py, pz, mx, my, mz = PS[node.code]
    print(f"Node {id}: [px:{px:.3f}, py:{py:.3f}, pz:{pz:.3f}]")
    print(f"Node {id}: [mx:{mx:.3f}, my:{my:.3f}, mz:{mz:.3f}]")

# Sistemin 2D çizimi matplotlib kütüphanesi ile yapılıyor
import matplotlib as mpl
def draw(nodes, elements, filename, US=None, factor=1):
  X_max = max(node.X for id, node in nodes.items())
  X_min = min(node.X for id, node in nodes.items())
  Y_max = max(node.Y for id, node in nodes.items())
  Y_min = min(node.Y for id, node in nodes.items())
  x_max = max(node.X + US[node.code][0] for id, node in nodes.items())
  x_min = min(node.X + US[node.code][0] for id, node in nodes.items())
  y_max = max(node.Y + US[node.code][1] for id, node in nodes.items())
  y_min = min(node.Y + US[node.code][1] for id, node in nodes.items())
  Lx = max(X_max, x_max) - min(X_min, x_min)
  Ly = max(Y_max, y_max) - min(Y_min, y_min)
  lines_mesh = []
  lines_deformed = []
  for id, elm in elements.items():
    n1, n2 = elm.conn
    if US is not None: u1, u2 = [US[node.code] for node in elm.conn]
    x1, y1 = n1.X, n1.Y
    x2, y2 = n2.X, n2.Y
    lines_mesh.append([(x1,y1), (x2,y2)])
    if US is not None:
      x1, y1 = n1.X + u1[0] * factor, n1.Y + u1[1] * factor
      x2, y2 = n2.X + u2[0] * factor, n2.Y + u2[1] * factor
      lines_deformed.append([(x1,y1), (x2,y2)])
  mpl.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib import collections  as mc
  lc0 = mc.LineCollection(lines_mesh, linewidths=0.5)
  if US is not None: lc = mc.LineCollection(lines_deformed, linewidths=2)
  fig = plt.figure(figsize=(Lx*1.5, Ly*1.5))
  ax = fig.add_subplot(111)
  ax.add_collection(lc0)
  if US is not None: ax.add_collection(lc)
  ax.autoscale()
  ax.margins(0.1)
  fig.savefig(filename)


# Çizim sonuçlarını görmek için "results" klasörüne bakınız
# Eğer "results" klasörü mevcut değil ise kodu çalıştırmadan önce bu klasörü oluşturunuz
filename = r'.\results\sec5_frame3d_deformed_shape.png'
draw(nodes, elements, filename=filename, US=US, factor=20)

# İşletim sistemi izin verdiği takdirde oluşturulan çizim dosyası otomatik olarak açılacaktır
import os
os.startfile(filename)