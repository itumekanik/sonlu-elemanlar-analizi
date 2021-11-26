import numpy as np   # sayısal işlemler için python kütüphanesi
INV = np.linalg.inv  # matris ters alma fonksiyonu
DET = np.linalg.det  # matris determinant alma fonksiyonu
np.set_printoptions(suppress=True)  # sayıların yazım (print) ayarları için

nodes = dict()     # Nodları hafızada tutacak konteyner
elements = dict()  # Elemanları hafızada tutacak konteyner

class NodeFrame2D:
    def __init__(node, id, X, Y):
        node.id = id
        node.X, node.Y = X, Y      # Koordinatlar
        node.rest = [0, 0, 0]      # Mesnet Koşulu [0: serbest, 1:tutulu]
        node.force = [0, 0, 0]     # Tekil-Yük [Px, Py, Mz]
        node.disp = [0, 0, 0]      # Mesnet Çökmesi [delta_x, delta_y, delta_teta]
        node.code = [-1, -1, -1]   # Serbestlikler (Kod) [dx, dy, tz]
        nodes[id] = node           # Nod nesnesi nodes içerisinde saklanıyor

class ElementFrame2D:
    def __init__(elm, id, conn, EA, EI, qx=0, qy=0, mz=0, hinges=None):
        elm.id = id
        elm.conn = [nodes[id] for id in conn]  # Bağlantı haritası [n1, n2]
        elm.EA = EA           # Eksenel kuvvet rijitliği
        elm.EI = EI           # Eğilme rijitliği
        elm.q = [qx, qy, mz]  # Yayılı yükler [qx, qy, mz]
        elm.hinges = hinges   # {"ku1": redör1, "kv1": redör2, "kt1": redör3, ...}
        elements[id] = elm    # Eleman nesnesi elements içerisinde saklanıyor

    def code(elm):  # Kod-Vektörü [u1, u2, u3, u4, u5, u6]
        n1, n2 = elm.conn
        return n1.code + n2.code

    def nx_ny_L(elm):  # Doğrultman kosinüsleri ve eleman boyu
        n1, n2 = elm.conn
        Lx = n2.X - n1.X
        Ly = n2.Y - n1.Y
        L = (Lx**2 + Ly**2)**0.5
        nx, ny = Lx/L, Ly/L
        return nx, ny, L

    def T_Alfa(elm):
        nx, ny, L = elm.nx_ny_L()
        return np.asarray([
            [ nx,  ny,   0,   0,    0,   0],
            [-ny,  nx,   0,   0,    0,   0],
            [ 0,   0,    1,   0,    0,   0],
            [ 0,   0,    0,   nx,   ny,  0],
            [ 0,   0,    0,  -ny,   nx,  0],
            [ 0,   0,    0,   0,    0,   1]])

    def K_Local_Rigid_Connection(elm):  # Lokal rijitlik matrisi (Rijit Bağlantılı)
        EA = elm.EA
        EI = elm.EI
        nx, ny, L = elm.nx_ny_L()
        L2, L3 = L**2, L**3
        return np.asarray([
            [ EA/L,   0,          0,        -EA/L,    0,          0      ],
            [ 0,      12*EI/L3,   6*EI/L2,   0,      -12*EI/L3,   6*EI/L2],
            [ 0,      6*EI/L2,    4*EI/L,    0,      -6*EI/L2,    2*EI/L ],
            [-EA/L,   0,          0,         EA/L,    0,          0      ],
            [ 0,     -12*EI/L3,  -6*EI/L2,   0,       12*EI/L3,  -6*EI/L2],
            [ 0,      6*EI/L2,    2*EI/L,    0,      -6*EI/L2,    4*EI/L ]])

    def get_k(elm):
        EA = elm.EA
        EI = elm.EI
        nx, ny, L = elm.nx_ny_L()
        L2, L3 = L ** 2, L ** 3
        h = elm.hinges
        kv = [h.get("ku1", 1e4 * EA / L), h.get("kv1", 1e6 * EI / L3),
              h.get("kt1", 1e4 * EI / L), h.get("ku2", 1e4 * EA / L),
              h.get("kv2", 1e6 * EI / L3), h.get("kt2", 1e4 * EI / L)]
        k = np.identity(6) * kv
        return k

    def K_Local(elm):  # Lokal rijitlik matrisi
        K = elm.K_Local_Rigid_Connection()
        if elm.hinges is None: return K
        k = elm.get_k()
        return K @ INV(K + k) @ k

    def K(elm):   # Global rijitlik matrisi
        T = elm.T_Alfa()
        return INV(T) @ elm.K_Local() @ T

    def q_Local(elm):  # Lokal yayılı yük vektörü
        qx, qy, mz = elm.q
        nx, ny, L = elm.nx_ny_L()
        q = [0.5*qx*L, 0.5*qy*L-mz, qy*L**2/12, 0.5*qx*L, 0.5*qy*L+mz, -qy*L**2/12]
        if elm.hinges is None: return q
        k = elm.get_k()
        K = elm.K_Local_Rigid_Connection()
        return (np.identity(6) - K @ INV(K + k)) @ q

    def B(elm):  # Birleştirilecek yayılı yük vektörü INV(T) @ q_Lokal
        T = elm.T_Alfa()
        return INV(T) @ elm.q_Local()


NodeFrame2D(id=1, X=0.0, Y=0.0)
NodeFrame2D(id=2, X=6.0, Y=0.0)
NodeFrame2D(id=3, X=2.0, Y=3.0)
NodeFrame2D(id=4, X=6.0, Y=3.0)
NodeFrame2D(id=5, X=10.0, Y=3.0)

EA, EI = 4200000, 87500
ElementFrame2D(id=1, conn=[1, 3], EA=EA, EI=EI, hinges={"kt1": 0.5*EI})
ElementFrame2D(id=2, conn=[2, 4], EA=EA, EI=EI, qy=20, hinges={"kv2": 0})
ElementFrame2D(id=3, conn=[3, 4], EA=EA, EI=EI, mz=10, hinges={"kt2": 0})
ElementFrame2D(id=4, conn=[4, 5], EA=EA, EI=EI, qy=-30, hinges={"ku1": 0.3*EA})

nodes[1].rest = [1, 1, 1]
nodes[2].rest = [1, 1, 1]
nodes[5].rest = [1, 1, 0]
nodes[3].force = [100, 0, 0]
nodes[5].force = [0, 0, 50]

#------------ Buradan sonraki işlemler kitapta yer almamıştır --------------#

# Sistem denklemi oluşturulup çözüm yapılıyor
from solver import solve
US, PS = solve(nodes, elements)

# Sonuçlar ekrana basılıyor
for id, node in nodes.items():
    ux, uy, tz = US[node.code]
    px, py, mz = PS[node.code]
    print(f"Node {id}: [ux:{ux:.8f}, uy:{uy:.8f}, tz:{tz:.8f}] [px:{px:.4f}, py:{py:.4f}, mz:{mz:.4f}]")

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
  from matplotlib import collections as mc
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
filename = r'.\results\sec5_frame2d_deformed_shape.png'
draw(nodes, elements, filename=filename, US=US, factor=2000)

# İşletim sistemi izin verdiği takdirde oluşturulan çizim dosyası otomatik olarak açılacaktır
import os
os.startfile(filename)