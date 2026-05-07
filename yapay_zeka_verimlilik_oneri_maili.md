# Konu: Lokal yapay zekâ ile AR-GE, PLM, analiz, test ve üretim verimliliği için öneri planı

Merhaba [İsim],

Üst yönetimin “yapay zekâyı nasıl daha etkin kullanabiliriz?” sorusu için kendi gözlem ve önerilerimi toparladım. Bence bizim durumda ana yaklaşım şu olmalı: **cloud tabanlı çözümler yerine tamamen lokal/on-prem çalışan, şirket verisini dışarı çıkarmayan, Teamcenter + ortak disk + analiz/test verileriyle entegre çalışan bir mühendislik yapay zekâ katmanı kurmak.**

Şu anda web chat üzerinden Qwen3 Coder Next 80B ve openoss/gpt-oss-120B gibi modelleri kullanabiliyoruz. Basit Python scriptleri, doküman özetleme, çeviri, rapor metni taslağı, mail taslağı gibi işler için zaten fayda görüyoruz. Ancak bence asıl verimlilik artışı, bu modelleri sadece “chat ekranı” olarak bırakmayıp **mevcut mühendislik iş akışlarına bağlamakla** gelecek.

Özet fikrim:

- Kısa vadede: Mevcut lokal modelleri daha erişilebilir, ölçülebilir ve güvenli hale getirip hızlı kazanımlar alalım.
- Orta vadede: Teamcenter, ortak disk, CAE/CFD raporları ve test verileri üzerinde kaynak gösteren mühendislik asistanları kuralım.
- Uzun vadede: PLM dijital ipliğini güçlendirip analiz/test verilerinden öğrenen tasarım optimizasyonu, test azaltma ve karar destek sistemleri geliştirelim.

Aşağıda daha somut planı yazdım.

---

## 1. Mevcut altyapı hakkında değerlendirme

Elimizde HPC ve GPU altyapısı olması büyük avantaj. Dış cloud kullanımı kabul edilebilir değilse bile, lokal modellerle oldukça fazla iş yapılabilir.

Şu an verdiğimiz bilgiye göre modeller 1000 kelimelik bir metni yaklaşık 10-12 saniyede üretebiliyor. Bu, kaba hesapla tek istek için yaklaşık **100-200 token/saniye** mertebesinde bir çıktı hızına işaret ediyor. Tam değer; Türkçe/İngilizce tokenizasyonuna, prompt uzunluğuna, kullanılan quantization seviyesine, batch ayarına, context uzunluğuna ve inference altyapısına göre değişir.

Bu performans bireysel kullanım için iyi görünüyor. Fakat yüzlerce kişinin gün içinde kullanacağı bir yapı için sadece “tek prompt kaç saniye sürdü?” ölçümü yeterli değil. Bilişim ekibinden şu benchmark’ı istemek faydalı olur:

- 25 / 50 / 100 eşzamanlı kullanıcı senaryosu
- Kısa cevap, uzun cevap, kod üretimi, doküman özeti senaryoları
- 2k, 8k, 32k context uzunluğu senaryoları
- p50 / p95 cevap süresi
- Toplam token/saniye
- GPU kullanım oranı
- Kuyrukta bekleme süresi
- Hangi model hangi iş için daha uygun?

Bence burada tek bir modelle her şeyi çözmeye çalışmamalıyız. Daha doğru mimari şu olur:

- Kod ve otomasyon işleri için Qwen3 Coder Next veya benzeri kod ağırlıklı model
- Genel mühendislik soru-cevap ve rapor işleri için gpt-oss/openoss-120B veya benzeri güçlü reasoning modeli
- Uzun doküman/çoklu doküman özetleme için ayrı optimize edilmiş model
- Görsel/doküman/çizim yorumlama için ileride multimodal model
- Hızlı sınıflandırma/etiketleme işleri için daha küçük 7B-30B modeller

Yani “herkes tek web chat’e girsin” yerine, arkada model havuzu olan bir **şirket içi AI gateway/API** kurmak daha sağlıklı olur.

---

## 2. En yüksek fayda alanı: AR-GE bilgi asistanı

Bence ilk büyük kazanım, AR-GE’de daha önce yapılmış mühendislik bilgisinin tekrar kullanılabilmesi olur.

Bugün birçok teknik doküman Teamcenter yerine ortak disk alanlarında duruyor. Parça-doküman ilişkileri zayıf, trace link’ler büyük ölçüde konfigürasyon mühendislerinin elle kurduğu yapılara dayanıyor. Bu yüzden geçmiş analiz, test ve tasarım kararlarına erişmek zaman alıyor.

Burada lokal bir yapay zekâ asistanı şu şekilde çalışabilir:

- Ortak disk, Teamcenter dokümanları, analiz raporları, test raporları ve tasarım karar notlarını indeksler.
- Kullanıcının yetkisine göre sadece görebileceği dokümanlardan cevap üretir.
- Cevap verirken mutlaka kaynak doküman, revizyon, tarih ve dosya yolunu gösterir.
- “Bu bilgi güncel mi, eski revizyona mı ait?” ayrımı yapar.
- Benzer geçmiş problemleri bulur.
- Parça numarası, revizyon, test numarası, analiz case ID gibi teknik kimliklerle arama yapabilir.

Örnek kullanım:

> “Şu silindir kapağı revizyonunda daha önce termal yorulma problemi yaşanmış mı? Hangi analizler ve testler yapılmış?”

> “Bu CFD çalışmasında kullanılan sınır şartlarına benzer geçmiş Fluent/CFX/STAR-CCM+ raporlarını bul.”

> “Bu parçada yapılan tasarım değişikliği hangi test planlarını ve hangi FEA case’lerini etkiler?”

Bu kullanım doğrudan mühendislik zamanı kazandırır. Ayrıca yeni başlayan mühendislerin kurumsal bilgiye erişimini ciddi hızlandırır.

---

## 3. Teamcenter tarafı için öneri

Teamcenter Copilot gibi çözümler kesinlikle değerlendirilmeli, fakat bizim için kritik şart şu olmalı: **tamamen on-prem veya bizim kontrolümüzde çalışan bir kurulum olmalı; şirket verisi dış cloud LLM servislerine gitmemeli.**

Teamcenter Copilot’un güçlü tarafı, PLM verisine doğal olarak yakın olması. BOM, doküman, gereksinim, parça revizyonu, kaynak gösterme ve yetki kontrolü gibi konularda hazır entegrasyon avantajı olabilir. Ancak bizim mevcut durumda dokümanların önemli kısmı Teamcenter dışında ortak disklerde olduğu için, sadece Copilot almak tüm problemi çözmez.

Benim önerim iki paralel hat:

### Hat 1 — Teamcenter olgunlaştırma

- Hangi dokümanların Teamcenter’a taşınması gerektiğini belirleyelim.
- Ortak disklerdeki dokümanları AI ile sınıflandıralım: analiz raporu, test raporu, çizim eki, prosedür, müşteri dokümanı, tedarikçi dokümanı vb.
- AI, olası parça numarası / revizyon / proje / test ilişkilerini önersin.
- Konfigürasyon mühendisleri bu önerileri onaylasın veya reddetsin.
- Bu şekilde trace link kurma işi tamamen elle değil, AI destekli yarı otomatik hale gelsin.

### Hat 2 — Teamcenter Copilot / Teamcenter AI pilotu

- Siemens/entegratör ile on-prem deployment, desteklenen LLM, lisans ve veri güvenliği netleştirilsin.
- Eğer kendi lokal LLM endpoint’lerimizle veya on-prem desteklenen açık modellerle çalışabiliyorsa pilot yapılmalı.
- Pilot kapsamı küçük tutulmalı: örneğin belirli bir ürün ailesi, belirli bir proje veya belirli doküman havuzu.
- Başarı kriteri “güzel cevap veriyor” değil, “doğru kaynağı buluyor, revizyonu doğru ayırıyor, kullanıcı yetkisine uyuyor, mühendislik kararı için izlenebilir çıktı üretiyor” olmalı.

Bence Teamcenter Copilot alınsa bile, onun yanında kendi lokal RAG/AI altyapımızın olması önemli. Çünkü ortak diskler, analiz dosyaları, test verileri, scriptler ve eski raporlar Teamcenter dışında yaşamaya devam ediyor.

---

## 4. CAE/CFD ekipleri için kısa vadeli verimlilik alanları

AR-GE ekibinin yaklaşık üçte biri Ansys Mechanical, CalculiX, Fluent, CFX, STAR-CCM+ gibi araçlarla çalışıyor. Bu ekipler için AI çok hızlı fayda sağlayabilir.

Kısa vadede yapılabilecekler:

- Solver log özetleme: non-convergence, contact problemi, mesh uyarısı, residual davranışı, boundary condition hatası vb.
- Analiz raporu taslağı: case tanımı, sınır şartları, malzeme, mesh, sonuç tablosu, riskler, açık noktalar.
- Post-processing script üretimi: Python, APDL, journal, bash, MATLAB vb.
- Parametrik analiz dosyalarının hazırlanması.
- Eski raporlardan benzer case bulma.
- Mesh kalite raporlarının otomatik yorumlanması.
- FEA/CFD sonuçlarının test verisiyle karşılaştırılması için script ve tablo üretimi.

Burada önemli sınır şu: AI mühendislik kararını tek başına vermemeli. Ama raporun ilk taslağını, log analizini, scriptleri, karşılaştırma tablolarını ve tekrar eden işleri çok hızlandırabilir.

Örneğin bir analiz mühendisi şu şekilde çalışabilmeli:

> “Bu Fluent case logunu ve sonuç dosyasını özetle. Baseline case ile basınç kaybı, debi dağılımı ve maksimum sıcaklık farklarını tablo yap. Rapor taslağı oluştur.”

veya:

> “Bu Mechanical sonuçlarından maksimum gerilme bölgelerini, safety factor limitlerini ve mesh bağımlılığı notlarını rapor formatına getir.”

Bunu web chat yerine analiz dosyalarına erişebilen, lokal çalışan, yetkili bir AI arayüzüyle yapmak gerekir.

---

## 5. Mekanik tasarım ekipleri için kullanım alanları

Mekanik tasarım tarafında Siemens NX/MX kullanan ekipler için de ciddi potansiyel var.

Öneriler:

- Tasarım kontrol listesi asistanı: malzeme, tolerans, üretilebilirlik, montaj, bakım, erişilebilirlik, standard parça kullanımı.
- Geçmiş benzer parça arama: “Bu geometriye/fonksiyona benzer geçmiş tasarımlar neler?”
- Revizyon değişiklik özeti: “Bu revizyonda teknik olarak ne değişti, hangi dokümanlar etkilenir?”
- Tasarım gözden geçirme toplantıları için otomatik aksiyon listesi.
- Çizim notları ve teknik şartname metinlerinin standardizasyonu.
- Tasarım değişikliğinin analiz/test/üretim dokümanlarına etkisini çıkarma.

Uzun vadede görsel/multimodal modellerle teknik resim ve CAD ekran görüntülerinden sınırlı yorumlama da yapılabilir. Ancak ilk aşamada daha güvenli ve değerli alan, metin/doküman/revizyon ilişkileri ve tasarım kontrol listeleri olur.

---

## 6. Test, üretim ve enstrümantasyon ekipleri

Kalan ekiplerde de AI kullanım alanı çok fazla.

### Test ve enstrümantasyon

- Test planı taslakları
- Kanal listesi kontrolü: birim, sensör aralığı, kalibrasyon tarihi, eksik kanal
- Günlük test özeti
- Test sapma raporu
- Test verisi ile analiz sonucunun karşılaştırılması
- Anomali tespiti için time-series algoritmaları + AI ile açıklama/raporlama
- Post-test rapor taslağı

Burada LLM tek başına anomali tespit motoru olmamalı. Kritik sinyal izleme için deterministik kurallar, istatistiksel yöntemler veya time-series ML kullanılmalı; LLM ise sonucu özetlemeli ve rapora çevirmeli.

### Üretim ve kalite

- İş talimatı taslağı
- Üretim sapması/NCR/8D rapor taslağı
- CMM raporu özetleme
- Hurda/rework nedenlerinin sınıflandırılması
- Tedarikçi dokümanı ve malzeme sertifikası kontrolü
- Revizyon değişikliğinin üretim dokümanlarına etkisi
- Operatör/teknisyen için onaylı dokümanlardan soru-cevap asistanı

Burada da AI doğrudan üretim parametresi değiştirmemeli. Öneri ve taslak üretmeli; onay yine ilgili mühendis/kalite/proses sahibi tarafından verilmeli.

---

## 7. Mail, ortak disk ve raporlama yükü

Yoğun Outlook trafiği, dağınık ortak alanlar, elle rapor yazma ve elle dosya sınıflandırma bence kısa vadede en hızlı verimlilik alanlarından biri.

Yapılabilecekler:

- Lokal mail özetleyici: uzun mail zincirinden kararlar, açık aksiyonlar, sorumlular, tarihler.
- Toplantı notundan aksiyon listesi.
- Ortak disk dosya sınıflandırıcı: proje, parça, revizyon, doküman tipi, tarih.
- Rapor şablonu doldurucu: analiz/test sonuçlarından ilk rapor taslağı.
- Teknik çeviri ve terminoloji standardizasyonu.
- Doküman karşılaştırma: iki revizyon arasında ne değişti?

Bu işler mühendislik kararından daha düşük riskli ama zaman kazancı yüksek işlerdir. İlk pilotlar için uygun olur.

---

## 8. Kısa, orta ve uzun vadeli plan önerisi

## Kısa vade: 0-3 ay

Amaç: Mevcut lokal modellerden kontrollü ve ölçülebilir hızlı fayda almak.

Önerilen işler:

1. **AI gateway/API kurulumu**  
   Web chat yanında API erişimi açılmalı. Böylece scriptler, rapor araçları, analiz klasörleri ve iç portallar modele bağlanabilir.

2. **Kullanım politikası**  
   Hangi veri girilebilir, log tutulacak mı, çıktı nasıl doğrulanacak, teknik karar için insan onayı nasıl olacak netleştirilmeli.

3. **Model benchmark çalışması**  
   Qwen3 Coder Next 80B, openoss/gpt-oss-120B ve gerekirse Gemma 4 31B/26B gibi modeller aynı test setiyle denenmeli. Test seti bizim işlerimizden oluşmalı: analiz logu, test raporu, teknik çeviri, Python script, PLM dokümanı, mail özeti.

4. **Prompt ve şablon kütüphanesi**  
   Analiz raporu, test raporu, mail özeti, kod üretimi, doküman karşılaştırma, teknik çeviri için standart promptlar hazırlanmalı.

5. **İlk 3 pilot**  
   - Ortak disk + raporlar üzerinde AR-GE bilgi asistanı
   - CAE/CFD log ve rapor taslağı asistanı
   - Test raporu/günlük test özeti asistanı

6. **Başarı metriği**  
   Her pilot için zaman kazanımı, hata azalması, tekrar kullanılan bilgi sayısı, kullanıcı memnuniyeti ve yanlış/kanıtsız cevap oranı ölçülmeli.

## Orta vade: 3-9 ay

Amaç: AI’ı gerçek mühendislik veri kaynaklarına bağlamak.

Önerilen işler:

1. **Teamcenter + ortak disk RAG sistemi**  
   Kaynak gösteren, revizyon farkını anlayan, yetki kontrollü bir mühendislik arama/asistan sistemi kurulmalı.

2. **Teamcenter Copilot pilotu**  
   Sadece on-prem şartı sağlanıyorsa ve verinin dışarı çıkmayacağı garanti ediliyorsa pilot yapılmalı. Kapsam sınırlı tutulmalı.

3. **Doküman sınıflandırma ve trace link önerme**  
   AI, ortak disklerdeki dokümanların Teamcenter’daki parça/revizyon/proje/test ile olası ilişkilerini önersin. Konfigürasyon mühendisleri onaylasın.

4. **CAE/CFD otomasyon paketi**  
   Analiz log okuma, post-processing, baseline karşılaştırma, rapor üretme ve script üretme işleri standartlaştırılsın.

5. **Test verisi asistanı**  
   Test planı, kanal listesi, günlük test özeti, sapma raporu ve analiz-test korelasyonu için araç geliştirilsin.

6. **Outlook/toplantı/aksiyon entegrasyonu**  
   Lokal çalışan mail ve toplantı özeti sistemi kurulabilir. Aksiyonlar Jira/PLM/Excel vb. takip sistemlerine öneri olarak düşebilir.

## Uzun vade: 9-24 ay

Amaç: AI destekli dijital mühendislik ve karar destek altyapısı kurmak.

Önerilen işler:

1. **Dijital iplik/digital thread**  
   Parça → revizyon → gereksinim → analiz → test → üretim → kalite → saha verisi ilişkileri güçlendirilmeli.

2. **Tasarım değişikliği etki analizi**  
   Bir parça veya revizyon değiştiğinde AI hangi analizleri, testleri, üretim dokümanlarını ve kalite kontrollerini etkilediğini önersin.

3. **Surrogate model ve optimizasyon**  
   FEA/CFD/test verilerinden öğrenen, tasarım uzayını daha hızlı tarayan, pahalı analiz ve test sayısını azaltan modeller kurulabilir.

4. **HPC job asistanı**  
   AI; analiz case hazırlama, job submission, log izleme, hata durumunda öneri verme ve sonuçları özetleme süreçlerinde yardımcı olabilir.

5. **Kurumsal mühendislik hafızası**  
   Eski projelerdeki dersler, failure investigation raporları, test sonuçları ve tasarım kararları erişilebilir hale gelmeli.

6. **Üretim ve kalite karar destek sistemi**  
   NCR, 8D, CMM, tedarikçi kalite, proses sapması ve revizyon etkisi gibi alanlarda AI destekli ama insan onaylı süreçler kurulabilir.

---

## 9. Dikkat edilmesi gereken riskler

Bence en kritik riskler şunlar:

- AI yanlış ama ikna edici cevap verebilir; bu yüzden kaynak gösterme zorunlu olmalı.
- Revizyon ayrımı doğru yapılmazsa eski bilgiyle karar verilebilir.
- Teamcenter dışındaki ortak disk verileri düzensiz olduğu için önce veri temizliği gerekir.
- Yetki kontrolü iyi kurulmazsa hassas bilgi yanlış kişiye açılabilir.
- AI çıktısı teknik onay yerine geçmemeli.
- “Chatbot var, AI kullandık” yaklaşımıyla kalırsak büyük verimlilik alınmaz.

Bu yüzden bence her AI çıktısı şu formatı sağlamalı:

- Cevap
- Kullanılan kaynaklar
- Revizyon/tarih bilgisi
- Emin olunmayan noktalar
- Eksik veri
- Önerilen sonraki kontrol/adım

---

## 10. Benim önerdiğim ilk aksiyon

Bence bunu şirket içinde küçük ama güçlü bir program olarak başlatabiliriz:

**“On-prem Mühendislik AI Verimlilik Programı”**

İlk 3 ay için önerilen ekip:

- 1 ürün sahibi / koordinatör
- Bilişimden 1-2 AI/HPC/platform sorumlusu
- Teamcenter/PLM tarafında 1-2 kişi
- CAE/CFD’den birkaç pilot kullanıcı
- Test/enstrümantasyondan birkaç pilot kullanıcı
- Konfigürasyon/kalite tarafında temsilci

İlk 3 pilot:

1. AR-GE bilgi asistanı: ortak disk + geçmiş analiz/test raporları  
2. CAE/CFD log ve rapor asistanı  
3. Test raporu ve kanal listesi kontrol asistanı  

3 ay sonunda şu sorulara cevap verebiliriz:

- Mühendis başına haftada kaç saat kazanç var?
- Hangi işlerde AI gerçekten iyi, hangilerinde değil?
- Mevcut GPU altyapısı kaç kullanıcıyı kaldırıyor?
- Hangi model hangi iş için daha iyi?
- Teamcenter Copilot veya Siemens AI tarafına yatırım mantıklı mı?
- Teamcenter’a veri taşıma ve trace link kurma işinde AI ne kadar yardımcı oluyor?

Kısaca benim görüşüm şu: Bizde AI’dan verimlilik almak için en doğru yol, cloud chatbot değil; **lokal modeller + yetki kontrollü RAG + Teamcenter/ortak disk/CAE/test entegrasyonu + insan onaylı mühendislik süreçleri** olmalı. İlk hedef tasarım yapay zekâsı değil, mühendislik bilgisini bulmayı, raporlamayı, analiz/test işlerini ve doküman düzenini hızlandırmak olmalı. Bu temel oturduktan sonra tasarım optimizasyonu ve daha ileri karar destek sistemlerine geçebiliriz.

Selamlar,  
[Adın]
