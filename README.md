## Sonlu Elemanlar Analizi

- [Kodları Görmek için Tıklayınız](https://github.com/itumekanik/sonlu-elemanlar-analizi/tree/master/SEA_Book)

- [Sunumları Görmek için Tıklayınız](https://github.com/itumekanik/sonlu-elemanlar-analizi/tree/master/SEA_Presentation)

**Not:** 2/Mart/2023 tarihi itibari ile kodların çalıştırılması ile ilgili güncellemeler yapılmıştır.
         Bu tarihten önce yapılan yüklemelerin tekrardan yapılması tavsiye edilmektedir.

**Not:** Tüm dosyaları indirmek için yukarıdaki "Code" butonundan "Download ZIP" seçeneğini kullanınız.

**Not:** Sunumlar ve kodlar güncellendikçe buradan duyurulacaktır.

**Not:** Sunumlar ve kodlarla ilgili öneri ve istekleriniz için "Issues" tabından bildirimde bulunmanız rica olunur.


# Kodların çalıştırılabilmesi için aşağıdaki adımları uygulayınız

- Aşağıdaki adresten "miniconda" programını indirerek bilgisayarınıza kurunuz.
- https://docs.conda.io/en/latest/miniconda.html

- Bu YouTube videosu kurulum konusunda yardımcı olacaktır
- https://youtu.be/XCvgyvBFjyM

- Bilgisayarınızdan "command prompt" u (Terminal i) açınız. 
  Bu işlemi windows arama çubuğunda "cmd" komutunu aratarak kolayca yapabilirsiniz.
  Alternatif olarak, herhangi bir windows explorer penceresindeki adress çubuğuna "cmd" komutu yazabilirisiniz.
  Yardımıcı olması icin aşağıdaki YouTube videosunu izleyiniz.
- https://youtu.be/uE9WgNr3OjM

- Command promt (siyah ekran) açıldığında aşağıdaki komutu giriniz (Not: "V" harfi büyük olmalı).
- conda -V
- Bu komut yüklemiş olduğunuz miniconda 'nın versiyonunu size geri bildirecektir. 
  Eğer bir hata alıyorsanız yükleme ile ilgili bir problem yaşanmış demekdir. miniconda yı tekrar yüklemeniz gerekecektir.


- Aşağıdaki komutu command prompt ekranında giriniz (bu aşamada hangi klasör içinde olduğunuzun bir önemi yoktur)
- conda create -n sea python=3.10 numpy scipy matplotlib
- Bu komut "sea" adında bir "python" çalışma ortamı oluşturacak ve ek olarak da "numpy", "scipy" ve "matplotlib" kütüphanelerini kullanıma hazır hale getirecektir.


- Kurulum tamamlandıktan sonra "SEA_BOOK" klasörüne konumlanınız.
- Bu işlemi herhangi bir windows explorer penceresinden yapabilirsiniz.
- explorer penceresindeki adres çubuğuna giderek mevcut klasör isminin yerine "cmd" komutunu yazınız.
- Bu işlem sonucunda otomatik olarak ilgili klasöre hedeflenmiş bir siyah ekran (command prompt) açılacaktır.


- Bu siyah ekranda aşağıdaki komutu giriniz.
- conda activate sea
- Bu komut önceki adımlarda oluşturmuş olduğumuz "sea" isimli python ortamını aktif hale getirecektir.
- Artık klasör içerisindeki tüm dosyalar çalıştırılabilir durumdadır.


- Kodları çalıştırmak için dosya isimlerini referans alan aşağıdaki tipte komutları kullanabilirsiniz.
- python sec1_simple_1d_equation.py
- ...
- ...
- python sunum1_plane_stress_wrench_gauss_von_misses.py

- Sorularınız için "yilmazmura@itu.edu.tr" ye mesaj atabilirsiniz.
