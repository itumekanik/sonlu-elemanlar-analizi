import numpy as np  # sayısal işlemler için python kütüphanesi

nodes = dict()     # Nodları hafızada tutacak konteyner
elements = dict()  # Elemanları hafızada tutacak konteyner


class NodeTruss2D:
    def __init__(node, id, X, Y):
        node.Z = 0
        node.id = id
        node.X, node.Y = X, Y  # Koordinatlar
        node.rest = [0, 0]     # Mesnet Koşulu [0: serbest, 1:tutulu]
        node.force = [0, 0]    # Tekil-Yük [Px, Py]
        node.disp = [0, 0]     # Mesnet Çökmesi [delta_x, delta_y]
        node.code = [-1, -1]   # Serbestlikler (Kod) [dx, dy]
        nodes[id] = node  # Nod nesnesi nodes içerisinde saklanıyor


class ElementTruss2D:
    def __init__(elm, id, conn, EA):
        elm.id = id
        elm.conn = [nodes[id] for id in conn]  # Bağlantı haritası [n1, n2]
        elm.EA = EA         # Malzeme ve kesit
        elements[id] = elm  # Eleman nesnesi elements içerisinde saklanıyor

    # Kod-Vektörü [u1, u2, u3, u4]
    def code(elm):
        n1, n2 = elm.conn
        return n1.code + n2.code

    # Doğrultman kosinüsleri ve eleman boyunu hesaplar
    def nx_ny_L(elm):
        n1, n2 = elm.conn
        Lx = n2.X - n1.X
        Ly = n2.Y - n1.Y
        L = (Lx ** 2 + Ly ** 2) ** 0.5
        nx, ny = Lx / L, Ly / L
        return nx, ny, L

    # Global-Rijitlik Matrisi
    def K(elm):
        EA = elm.EA
        nx, ny, L = elm.nx_ny_L()
        return EA / L * np.asarray([
            [nx * nx, nx * ny, -nx * nx, -nx * ny],
            [ny * nx, ny * ny, -ny * nx, -ny * ny],
            [-nx * nx, -nx * ny, nx * nx, nx * ny],
            [-ny * nx, -ny * ny, ny * nx, ny * ny]])


NodeTruss2D(id=1, X=0.0, Y=0.0)
NodeTruss2D(id=2, X=2.4, Y=0.0)
NodeTruss2D(id=3, X=4.8, Y=0.0)
NodeTruss2D(id=4, X=1.2, Y=1.5)
NodeTruss2D(id=5, X=3.6, Y=1.5)

ElementTruss2D(id=1, conn=[1, 2], EA=5000)
ElementTruss2D(id=2, conn=[2, 3], EA=5000)
ElementTruss2D(id=3, conn=[1, 4], EA=5000)
ElementTruss2D(id=4, conn=[4, 2], EA=5000)
ElementTruss2D(id=5, conn=[2, 5], EA=5000)
ElementTruss2D(id=6, conn=[5, 3], EA=5000)
ElementTruss2D(id=7, conn=[4, 5], EA=5000)

nodes[1].rest = [1, 1]
nodes[3].rest = [0, 1]
nodes[2].force = [0, -30]
nodes[4].force = [20, 0]

#------------ Buradan sonraki işlemler kitapta yer almamıştır --------------#

# Sistem denklemi oluşturulup çözüm yapılıyor
from solver import solve
US, PS = solve(nodes, elements)

# Sonuçlar ekrana basılıyor
for id, node in nodes.items():
    ux, uy = US[node.code]
    px, py = PS[node.code]
    print(f"Node {id}: [ux:{ux:.4f}, uy:{uy:.4f}] [px:{px:.2f}, py:{py:.2f}]")

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
  lc0 = mc.LineCollection(lines_mesh, linewidths=1.0, color="grey")
  if US is not None: lc = mc.LineCollection(lines_deformed, linewidths=2.5, color="k")
  fig = plt.figure(figsize=(Lx*1.5, Ly*1.5))
  ax = fig.add_subplot(111)
  ax.add_collection(lc0)
  if US is not None: ax.add_collection(lc)
  ax.autoscale()
  ax.margins(0.1)
  fig.savefig(filename)


# Çizim sonuçlarını görmek için "results" klasörüne bakınız
# Eğer "results" klasörü mevcut değil ise kodu çalıştırmadan önce bu klasörü oluşturunuz
filename = r'.\results\sec5_truss2d_deformed_shape.png'
draw(nodes, elements, filename=filename, US=US, factor=5)

# İşletim sistemi izin verdiği takdirde oluşturulan çizim dosyası otomatik olarak açılacaktır
import os
os.startfile(filename)