'''
 f'' + 2 = 0        x=[0,1]        f(0)=0,   f(1)=0
 diferansiyel denkleminin SEY ile çözümü
'''
import numpy as np   # sayısal işlemler için python kütüphanesi
INV = np.linalg.inv  # matris ters alma fonksiyonu
np.set_printoptions(suppress=True)  # sayıların yazım (print) ayarları için

nodes = dict()     # Nodları hafızada tutacak koleksiyon
elements = dict()  # Elemanları hafızada tutacak koleksiyon


class Node:
    def __init__(node, id, X):
        node.X = X
        node.code = [-1]   # Serbestlik numarası (gerçek numaralar sonradan verilecek)
        node.rest = [0]    # (0: serbestlik bilinmiyor, 1: serbestlik biliniyor)
        node.disp = [0]    # Diriclet Koşulu (Serbestlik biliniyor ise dikkate alınır)
        node.force = [0]   # Neumann Koşulu
        nodes[id] = node   # Nod nesnesi nodes içerisinde saklanıyor


class Element:
    def __init__(elm, id, conn):
        elm.conn = [nodes[id] for id in conn]  # Bağlantı [DN1, DN2]
        elements[id] = elm  # Eleman nesnesi elements içerisinde saklanıyor

    def L(elm):
        n1, n2  = elm.conn  # Eleman düğüm noktaları conn içinden alınıyor
        return n2.X - n1.X  # Elemanın boyu hesaplanıyor

    def code(elm):  # Kod-Vektörü [f_left, f_rigth]
        n1, n2 = elm.conn
        return [n1.code[0], n2.code[0]]

    def B(elm):  # Eleman sağ taraf vektörü
        L = elm.L()
        return np.asarray([L, L])

    def K(elm):  # Eleman rijitlik matrisi (Ke)
        L = elm.L()
        return np.asarray([
            [ 1/L, -1/L],
            [-1/L,  1/L]
        ])

# ----------------------------------------------------
# Problem verilerinin oluşturulması
# ----------------------------------------------------
#
# Eleman Ağının oluşturulması
Lx = 1        # Problemin tanım Bölgesi [0,1]
Nx = 4        # Eleman adedi
dLx= Lx / Nx  # Yatay boyut artımı

# Düğüm Noktaları oluşturuluyor (i saymaya sıfırdan başlar)
for i in range(Nx + 1):
    Node(id=i+1, X=dLx*i)

# Elemanlar oluşturuluyor (i saymaya sıfırdan başlar)
for i in range(Nx):
    n1 = i + 1
    n2 = n1 + 1
    Element(id=i+1, conn=[n1, n2])

# Sınır koşulları belirleniyor
for id, node in nodes.items():
    if (node.X == 0 or node.X == Lx): node.rest =  [1]


# ----------------------------------------------------
# Nodlarda tanımlı serbestlik numaralarının (code) belirlenmesi
# ----------------------------------------------------
#
M = 0  # M:Toplam serbestlik sayısı
N = 0  # N:Bilinmeyen serbestliklerin sayısı

# Tutulu olmayanlar (rest==0) numaralanıyor
for id, node in nodes.items():
    for index, rest in enumerate(node.rest):
        if rest == 0: node.code[index] = M; M += 1

N = M  # Bilinmeyen sayısı N de saklanıyor

# Tutulu olanlar (rest==1) numaralanıyor
for id, node in nodes.items():
    for index, rest in enumerate(node.rest):
        if rest == 1: node.code[index] = M; M += 1

# ----------------------------------------------------
# Sistem denklem takımının oluşturulması (Birleştirme)
# ----------------------------------------------------
#
KS = np.zeros((M, M))   # Rijitlik matrisi
US = np.zeros(M)        # Serbestlik vektörü
PS = np.zeros(M)        # Tekillik vektörü (sağ taraf)
BS = np.zeros(M)        # Sağ taraf vektörü

# Rijitlik matrislerinin birleştirilmesi
for id, elm in elements.items():
    code = elm.code()
    Ke = elm.K()
    dim = len(code)
    for i in range(dim):
        for j in range(dim):
            KS[code[i], code[j]] += Ke[i, j]

# Denklemin sağ taraf vektörünün birleştirilmesi
for id, elm in elements.items():
    code = elm.code()
    BS[code] += elm.B()

# Sınır koşullarının birleştirilmesi
# Problemde 0 (sıfır) oldukları için bu adıma gerek yoktu
# Yine de, bütünlüğü bozmamak bakımından dahil edilmiştir
for id, node in nodes.items():
    code = node.code
    US[code] = node.disp  # Dirichlet Koşulları
    PS[code] = node.force # Neumann Koşulları

# ----------------------------------------------------
# Sistem denklem takımının çözümü
# ----------------------------------------------------
#
K11 = KS[0:N, 0:N]
K12 = KS[0:N, N:M]
K21 = KS[N:M, 0:N]
K22 = KS[N:M, N:M]

U2 = US[N:M]

P1 = PS[0:N]
B1 = BS[0:N]
B2 = BS[N:M]

U1 = INV(K11) @ (P1 + B1 - K12 @ U2)
P2 = K21 @ U1 + K22 @ U2 - B2

US = np.concatenate((U1, U2))
PS = np.concatenate((P1, P2))

# ----------------------------------------------------
# Çözüm çıktılarının yazdırılması
# ----------------------------------------------------
#
for id, node in nodes.items():
    ux, = US[node.code]
    px, = PS[node.code]
    print(f"Node {id}: [ux:{ux:.4f}] [px:{px:.2f}]")

# ----------------------------------------------------
# Çözüm çıktıları
# ----------------------------------------------------
#
# Node 1: [ux:0.0000] [px:-1.00]
# Node 2: [ux:0.1875] [px:0.00]
# Node 3: [ux:0.2500] [px:0.00]
# Node 4: [ux:0.1875] [px:0.00]
# Node 5: [ux:0.0000] [px:-1.00]