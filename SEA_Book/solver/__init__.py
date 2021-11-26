import numpy as np
"""
Bu fonksiyonun görevi:
[KS].US = RHS + PS
doğrusal denklem takımını oluştutup çözmektir.
[KS] : Elemanların "StiffnessMatrixToContribute" metodu ile gönderdiği 
      rijitlik matrislerinin birleştirilmesi ile oluşan sistem rijitlik matrisi
RHS  : Elemanların "RHSVectorsToContribute" metodu ile gönderdiği 
      vektörlerin birleştirilmesi ile oluşturulan sağ taraf vekötrü
PS   : Nodların "ExternalForceVectorToContribute" metodu ile gönderdiği 
      vektörlerin birleştirilmesi ile oluşturulan sağ taraf vektörü
US   : Öncelikle, nodların "DisplacementVectorToContribute" metodu ile gönderdiği 
       bilinen-serbestliklerin birleştirildiği vektör. Çözüm sonrasında, problemin
       çözümünü içeren sistem serbestlik vektörüne dönüşür.
"""
def solve(nodes, elements):
    # Nodlarda tanımlı serbestlik numaralarının (code) belirlenmesi
    M = 0  # M:Toplam serbestlik sayısı
    N = 0  # N:Bilinmeyen serbestliklerin sayısı
    for id, node in nodes.items():  # Tutulu olmayanlar (rest==0) numaralanıyor
        for index, rest in enumerate(node.rest):
            if rest == 0:
                node.code[index] = M
                M += 1

    N = M  # Bilinmeyen sayısı N de saklanıyor

    for id, node in nodes.items():  # Tutulu olanlar (rest==1) numaralanıyor
        for index, rest in enumerate(node.rest):
            if rest == 1:
                node.code[index] = M
                M += 1

    print("Bilinmeyen serbestlik sayısı (N):", N)
    print("Toplam serbestlik sayısı (M)    :", M)

    # Sistem denklem takımının oluşturulması (Birleştirme)
    US = np.zeros(M)
    PS = np.zeros(M)
    RHS = np.zeros(M)

    rows = []
    cols = []
    data = []

    for id, elm in elements.items():  # Rijitlik matrisi birleştiriliyor
        code = elm.code()
        try:
            Ke = elm.StiffnessMatrixToContribute()
        except:
            try:
                Ke = elm.K()
            except: pass
        rows += np.repeat(code, len(code)).tolist()
        cols += code * len(code)
        data += Ke.flatten().tolist()

    import scipy as sp
    import scipy.sparse
    import scipy.sparse.linalg

    KS = sp.sparse.coo_matrix((data, (rows, cols)), shape=(M, M), dtype=float).tocsc()

    # Eleman denklemi sağ tarafı katkısı birleştiriliyor
    for id, elm in elements.items():
        code = elm.code()
        # Eleman sağ taraf vektörleri birleştiriliyor
        try:
            RHS[code] += elm.RHSVectorsToContribute()
        except:
            if callable(getattr(elm, "B", None)):RHS[code] += elm.B()
            if callable(getattr(elm, "S", None)): RHS[code] += elm.S()

    for id, node in nodes.items():
        code = node.code
        # Nodal mesnet-çökmeleri birleştiriliyor
        try:
            US[code] = node.DisplacementVectorToContribute()
        except:
            try:
                US[code] = node.disp
            except: pass
        # Nodal tekil-kuvvetler birleştiriliyor
        try:
            PS[code] = node.ExternalForceVectorToContribute()
        except:
            try:
                PS[code] = node.force
            except: pass

    # Sistem denklem takımının çözümü
    K11 = KS[0:N, 0:N]
    K12 = KS[0:N, N:M]
    K21 = KS[N:M, 0:N]
    K22 = KS[N:M, N:M]

    U2 = US[N:M]

    P1 = PS[0:N]
    RHS1 = RHS[0:N]
    RHS2 = RHS[N:M]

    # Sistem matrisinin köşegeninde 0 (sıfır) a rastlandığında
    # ilgili terim küçük bir değer seçilerek uyarı basılıyor.
    for i in range(N):
        if abs(K11[i, i]) < 1e-10:
            K11[i, i] = 0.00001
            print("Uyarı: matris köşegeni sıfır olamaz!!!")

    # Burada çözümde tuple (bir çeşit python listesi) geliyor.
    # Bu yüzden sonuç değerin sıfırıncı indeksine konumlanıyor (bkz: scipy.org).
    U1 = sp.sparse.linalg.bicg(K11, P1 + RHS1 - K12 @ U2, tol=1e-06)[0]
    P2 = K21 @ U1 + K22 @ U2 - RHS2

    # Önceden bilinen ve sonradan hesaplan değerler tekrar birleştirilerek
    # sistem vektörlerinin tam formları yeniden elde ediliyor.
    US = np.concatenate((U1, U2))
    PS = np.concatenate((P1, P2))

    return US, PS