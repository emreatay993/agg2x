# Yapay Zeka Kullanımı — Kısa, Orta ve Uzun Vadeli Öneriler

**Konu:** Şirket içi YZ kullanımının yaygınlaştırılması — değerlendirme ve yol haritası önerisi

Merhaba [Lider Adı],

Geçen gün üst yönetimden gelen "yapay zekayı nasıl daha etkin kullanırız" sorusu kafamda dönüp duruyordu, biraz oturup düşündüm ve aklımdakileri toparlayıp yazmak istedim. Konu uzun olduğu için kısa-orta-uzun vade diye böldüm; istersen önce **hızlı özet**i okuyup, ilgini çekenlerin detayına inebilirsin.

---

## TL;DR

- **Veri güvenliği gereği** her şey on-premise kalmalı — bu konuda zaten hemfikiriz, cloud (ChatGPT, Claude.ai, Gemini vb.) masada değil.
- HPC üzerinde Qwen3 Coder 80B ve gpt-oss-120b zaten kurulu, web chat üzerinden günlük basit işler için kullanılıyor. **Asıl fırsat bunu kurumsal ölçeğe taşımakta.**
- **Kısa vade (0–3 ay):** mevcut modellere kapasite/erişim genişletme + ekibe kullanım eğitimi + birkaç hızlı kazanım (mail özetleme, doküman çeviri/özet, basit kod üretimi).
- **Orta vade (3–9 ay):** ortak alandaki teknik dokümanlar üzerinde **RAG** (Retrieval Augmented Generation) tabanlı soru-cevap sistemi, Outlook entegrasyonu, ANSYS/CFD log ve rapor analizi için domain-specific akışlar.
- **Uzun vade (9+ ay):** Teamcenter ile entegrasyon (Copilot dahil), parça-doküman trace link otomasyonu, fatigue/FEA için ajan tabanlı yardımcılar, üretim ve test ekiplerinde sensör/log verisi üzerinde anomali tespiti.

---

## 1. Mevcut Durum — Kısa Değerlendirme

### Altyapı tarafı
HPC + GPU altyapımız var, **Qwen3 Coder 80B** ve **gpt-oss-120b** kurulu, web chat üzerinden erişiyoruz. 1000 kelimelik makale ~10–12 saniyede çıkıyor. Kabaca hesap edersek bu **~110–130 token/saniye** civarı bir throughput demek (1000 kelime ≈ 1300–1500 token kabul ediyorum). Tek kullanıcı için gayet iyi, ama **arge + üretim ekibinde yüzlerce kişiye yayılınca** bu rakam concurrent kullanıcıyla bölündüğünde sıkıntı çıkar — eş zamanlı 20–30 kullanıcı bile bu hızı 5–10 saniye/cevap seviyesine düşürebilir, daha fazlası kuyruğa girer.

### Modeller hakkında
- **Qwen3 Coder 80B** ve **gpt-oss-120b** kod ve genel amaçlı işler için yeterli. 
- Eğer ek model kuracaksak öncelik bence şöyle:
  - **Gemma 3 27B** (sanırım "Gemma 4" derken bunu kastettin, henüz Gemma 4 yok bildiğim kadarıyla) — multimodal, hafif, çok kullanıcılı senaryolarda hızlı yanıt.
  - **Qwen2.5-VL** veya benzeri bir **vision-language modeli** — teknik çizim, ekran görüntüsü, FEA sonuç görselleri okutmak için.
  - **Embedding modeli** (örn. `bge-m3`, `nomic-embed-text`) — RAG için zorunlu, ayrı kurulmalı.

### Asıl darboğaz: kullanım kültürü
Şu an web chat'i "yan masadaki akıllı asistan" gibi kullanıyoruz — basit kod, çeviri, özet. Bu güzel ama **potansiyelin %10'u bile değil.** Asıl değer; bu modellerin **bizim kendi dokümanlarımıza, mail trafiğimize, parça veritabanımıza erişebildiği** noktada ortaya çıkıyor.

---

## 2. Kısa Vade Önerileri (0–3 ay) — "Düşük asılı meyveler"

### 2.1. BT ile konuşup kapasite/erişim planlaması
- Mevcut HPC kuyruğunda **YZ inference'a dedike GPU dilimi** ayrılmalı (analiz işleriyle çakışmasın).
- Web chat arayüzüne **SSO + kullanıcı bazlı kullanım istatistiği** eklenmeli — kim ne için kullanıyor görelim, nereye yatırım yapacağımızı bilelim.
- Concurrent kullanıcı testleri yapılmalı: 10, 50, 100 kullanıcıyla nasıl davranıyor?

### 2.2. Ekip içi farkındalık ve eğitim
- 1–2 saatlik kısa workshoplar: "şu işleri YZ ile şöyle hızlandırabilirsin" tarzı somut örneklerle. Soyut "YZ devrimi" sunumları işe yaramıyor, **kendi iş akışlarından örnek** lazım.
- Bir **iç wiki / Confluence sayfası** — "iyi prompt örnekleri", "şunu yapma", "şunda dikkat et" gibi.
- Birkaç **gönüllü champion** belirlemek (her ekipten 1 kişi) — onlar yaygınlaştırır.

### 2.3. Hızlı kazanım pilotları
Bunların hepsi mevcut modellerle, ek altyapı olmadan yapılabilir:
- **Toplantı notu özetleme** (Teams kayıtlarından transkript alıp özet + aksiyon maddesi).
- **Teknik doküman çeviri** (TR↔EN, özellikle tedarikçi yazışmaları ve standartlar için).
- **Mail taslağı yazma** — tedarikçiye, müşteriye, iç yazışmalar.
- **Kod review yardımcısı** — APDL, Python, MATLAB scriptleri için.

---

## 3. Orta Vade Önerileri (3–9 ay) — "Asıl iş burada başlıyor"

### 3.1. Ortak alan üzerinde RAG sistemi
Bu bence **en yüksek ROI'li** yatırım. Mantık şu:
- Ortak diskteki teknik dokümanları (PDF, Word, Excel, çizimler) **vektör veritabanına** indeksleyelim (Qdrant, Weaviate veya Milvus, hepsi self-hosted).
- Üzerine bir chat arayüzü — kullanıcı "X parçasının fatigue limit'i nedir, hangi dokümanda geçiyor" diye sorduğunda **kaynağıyla beraber** cevap dönsün.
- Bu, Teamcenter'ın yapamadığı (veya bizim yapamadığımız) **trace link sorununu da kısmen çözer** — çünkü model dokümanlar arası ilişkileri otomatik kuruyor.

İlk fazda 2–3 ekibin dokümanlarıyla pilot, sonra yaygınlaştırma. Tahminim **3–4 ay'da çalışır prototip**, 6 ayda olgun versiyon.

### 3.2. Outlook + mail asistanı
- Lokal kurulu bir LLM, Exchange üzerinden mail kuyruğunu okuyabilir (kullanıcı izniyle).
- "Bugünün önemli mailleri", "bu konuda ne konuşmuştuk geçen ay", "şu maile taslak cevap yaz" gibi işler.
- Microsoft Copilot'un cloud bağımlılığı sorun olduğu için **kendi versiyonumuzu** kurmamız gerekiyor — yapılabilir, ama biraz iş.

### 3.3. ANSYS/CFD için domain-specific akışlar
Burası senin de bildiğin gibi benim ilgi alanım. Somut fikirler:
- **Solver log parser** — 1000 satırlık ANSYS/Fluent çıktısından "convergence sorunu var mı, mesh kalitesi nasıl, hata var mı" özeti.
- **Rapor otomasyonu** — analiz sonuçlarından (görseller dahil, vision modeli ile) standart rapor şablonu doldurma.
- **APDL/Workbench script üretici** — "şu submodeli şu harmonik yükle koş" tarzı doğal dil → APDL.
- **Standart sorgulama** — "EN 13445'e göre şu şartta izin verilen gerilme nedir" — RAG ile standart dokümanlarına bağlı.

### 3.4. Doküman sınıflandırma ve dağınıklık
Ortak alandaki dağınıklığı **insan eliyle düzeltmek yıllar alır.** Ama bir LLM agent + embedding modeli ile:
- Otomatik kategorizasyon (proje, parça no, doküman tipi),
- Duplicate tespiti,
- Eksik metadata önerisi,
- Teamcenter'a göç için **otomatik sınıflandırma + öneri** üretilebilir.
Bu Teamcenter geçişini hızlandırır.

---

## 4. Uzun Vade Önerileri (9+ ay) — "Stratejik dönüşüm"

### 4.1. Teamcenter Copilot ve PLM entegrasyonu
- Siemens'in kendi Copilot çözümü var (Teamcenter AI / Industrial Copilot), **on-premise opsiyonu da mevcut** — bunu Siemens ile konuşmaya değer.
- Eğer Siemens tarafında uygun değilse, **kendi RAG sistemimizi Teamcenter API'leriyle** entegre edebiliriz.
- Asıl hedef: konfigürasyon mühendisinin elle kurduğu trace linklerin **önerilmesi** (tamamen otomatik değil, "bu parça muhtemelen şu dokümanla ilişkili, onaylar mısın" şeklinde human-in-the-loop).

### 4.2. Ajan tabanlı (agentic) yardımcılar
- Tek tek prompt yerine, **çok adımlı işleri otomatik yapan** ajanlar.
- Örnek: "Şu projenin geçen ayki tüm CFD koşularını topla, convergence sorunu olanları işaretle, rapor hazırla, ilgili tasarım mühendisine mail at."
- Bu seviye 1–2 yıl içinde olgunlaşır, şimdiden temellerini atmaya başlamak gerek.

### 4.3. Üretim, test ve enstrumantasyon tarafı
- **Sensör verisi üzerinde anomali tespiti** (klasik ML, ama LLM ile doğal dil arayüzü).
- Test raporlarının **otomatik yapılandırılması**.
- Kalibrasyon/sertifika dokümanlarının **otomatik takibi** (son kullanma tarihi, eksik kalibrasyonlar vb.).

### 4.4. Fine-tuning değerlendirmesi
- 1 yıl boyunca yeterli kullanım verisi biriktiğinde, açık ağırlıklı modellerden birini **kendi domain'imize fine-tune** edebiliriz (LoRA ile, full fine-tune değil — daha ucuz).
- Bu, özellikle FEA/CFD jargonu, iç parça kodları, kurum içi standartlar konusunda **belirgin doğruluk artışı** sağlar.

---

## 5. Risk ve Dikkat Edilecekler

- **Hallucination her zaman var** — kritik kararlarda mutlaka human-in-the-loop. Özellikle FEA, malzeme özellikleri, standart yorumlama gibi konularda.
- **Veri sızıntısı** — on-premise olsa bile, kullanıcıların hassas IP'yi promptlara yapıştırması konusunda iç politika lazım.
- **Aşırı bağımlılık** — yeni başlayan mühendislerin temel beceriyi öğrenmeden YZ'ye yaslanması uzun vadeli risk. Eğitim materyallerinde bunu vurgulamak lazım.
- **Beklenti yönetimi** — üst yönetime "YZ her şeyi çözer" diye sunmayalım, somut metrikler ve pilotlarla ilerleyelim.

---

## 6. Önerim — İlk Adım

Bence en mantıklı başlangıç şu üçü paralel:

1. **BT ile bir toplantı** ayarlayıp kapasite/erişim planlamasını netleştirmek.
2. **Bir RAG pilotu** başlatmak (küçük, 1 ekibin dokümanlarıyla, 4–6 hafta).
3. **2 saatlik bir kullanım workshop'u** organize edip ekibi mevcut araçlarla daha verimli kullanmaya yönlendirmek.

Bu üçü 1 ay içinde başlatılırsa, üst yönetime 2–3 ay sonra **somut sonuç** ve gerçekçi yol haritası sunabiliriz.

İstersen bunları detaylandırıp bir sunum/dokuman haline getirebilirim, ya da önce bunlardan birini seçip onun üzerinde derinleşelim. Müsait olduğunda 15–20 dakika oturup konuşalım istersen.

Selamlar,
Emre
