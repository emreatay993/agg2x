# RAG Pilotu — Teknik Plan ve Yol Haritası

**Hedef:** 4–6 hafta içinde, ortak diskteki teknik dokümanlar üzerinde çalışan, kaynak referanslı soru-cevap yapan bir prototip RAG sistemi kurmak.

**Kapsam (pilot):** Tek bir AR-GE alt ekibinin (örn. yapısal analiz veya konfigürasyon mühendisliği) ~500–2000 dokümanı.

**Out of scope (bu fazda):** Teamcenter entegrasyonu, fine-tuning, multi-tenancy, agent davranışı.

---

## 1. Mimari — Yüksek Seviye

```
[Ortak Disk / Network Share]
          │
          ▼
  ┌───────────────┐
  │  Ingestion    │  → dosya tarayıcı + format dönüştürücü
  │  Pipeline     │     (PDF, DOCX, XLSX, PPTX, MD, TXT)
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │  Chunking +   │  → semantic / structural chunking
  │  Embedding    │     (bge-m3 veya nomic-embed)
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │  Vector DB    │  → Qdrant (önerim)
  │  (on-prem)    │
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │  Retriever +  │  → hybrid search (dense + BM25)
  │  Reranker     │     + cross-encoder reranking
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │  LLM (mevcut) │  → Qwen3 Coder 80B veya gpt-oss-120b
  │  + prompt     │     üzerinden API çağrısı
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │  Web UI       │  → kaynaklı yanıt + chunk highlight
  │  (chat)       │
  └───────────────┘
```

---

## 2. Teknoloji Seçimleri ve Gerekçeleri

### 2.1. Embedding Model

| Aday | Avantaj | Dezavantaj | Karar |
|------|---------|------------|-------|
| **bge-m3** | Çok dilli (TR güçlü), hibrit (dense+sparse+colbert), açık ağırlık | Boyut biraz büyük (~2.3GB) | **Birinci tercih** |
| nomic-embed-text-v1.5 | Hafif (~550MB), hızlı, MIT lisans | TR performansı bge-m3 kadar iyi değil | İkincil/yedek |
| Qwen3-Embedding-0.6B/4B | Aynı aileyle uyum, son dönem güçlü | Daha az test edilmiş, ekosistem dar | İzlemeye al |

**Notum:** TR + EN karışık doküman varsa **bge-m3** rahat birinci, diğer iki seçeneği tartmaya bile gerek yok bence.

### 2.2. Vector Database

| Aday | Avantaj | Dezavantaj | Karar |
|------|---------|------------|-------|
| **Qdrant** | Rust tabanlı, hızlı, hibrit search built-in, on-prem self-hosted, REST + gRPC | Topluluk biraz daha küçük | **Birinci tercih** |
| Weaviate | Olgun, modüler, çok feature | Kaynak tüketimi yüksek, kurulum ağır | İkinci tercih |
| Milvus | Ölçeklenir, K8s native | Tek node için over-engineered, ops yükü | Pilot için aşırı |
| pgvector | PostgreSQL üstünde basit | Hibrit search ve filtreleme zayıf | Sadece DB zaten Postgres'se |

**Notum:** Pilot için **Qdrant** en sağlıklı seçim — kurulumu tek `docker compose up` kadar basit, performansı çok iyi, ileride scale gerekirse cluster moduna çıkar.

### 2.3. Orchestration Framework

| Aday | Notlar | Karar |
|------|--------|-------|
| **LlamaIndex** | RAG odaklı, Python, yoğun integration desteği, prod-ready | **Birinci tercih** |
| LangChain | Daha geniş ama dağınık, hızlı değişen API | Kullanılır ama pilotta gerek yok |
| Haystack | Olgun, Almanya menşeli, kurumsal odak | Geçerli alternatif |
| Hiçbiri (custom) | Tam kontrol ama yazılım yükü | Pilot için over-engineering |

**Notum:** LlamaIndex'in `RAG` ve `Workflows` API'leri tam bu pilot tipi için optimize. Custom yazmak 3 hafta yer.

### 2.4. LLM (Inference)

Mevcut HPC üstünde kurulu **Qwen3 Coder 80B** veya **gpt-oss-120b** kullanılacak. Pilot için ikisinden birini sabit seçip değerlendirelim — bence **gpt-oss-120b** genel sohbet ve özetleme için daha uygun (Qwen3 Coder kod-odaklı).

**API erişimi:** Web chat'in altındaki inference engine'in (vLLM / TGI / SGLang olduğunu varsayıyorum) **OpenAI-compatible endpoint**'i açılmalı. Bu BT ile konuşulacak ilk teknik şart.

### 2.5. Reranker (Opsiyonel ama Önerilen)

Top-50 retrieval sonrası top-5'e indirgemek için cross-encoder reranker:
- **bge-reranker-v2-m3** — bge-m3 ile uyumlu, açık ağırlık
- Faz 1'de atlanabilir, Faz 2'de eklenmeli (cevap kalitesini belirgin artırır)

### 2.6. UI / Frontend

| Aday | Notlar | Karar |
|------|--------|-------|
| **Open WebUI** | Mevcut webchat zaten benzer altyapı olabilir, hazır UI | **Birinci tercih (mümkünse)** |
| Streamlit | Python-only, hızlı prototip, prod için zayıf | Pilot için ok |
| Gradio | Streamlit benzeri, biraz daha esnek | Pilot için ok |
| Custom React | En esnek, en uzun | Faz 3'te düşünülür |

**Notum:** Mevcut webchat altyapısı eğer Open WebUI ise (büyük ihtimal), aynı altyapıya RAG pipeline'ı eklemek ideal — kullanıcı için yeni arayüz öğrenme yükü olmaz.

---

## 3. Doküman Hazırlama Stratejisi

### 3.1. Format dönüştürme

| Format | Tool | Notlar |
|--------|------|--------|
| PDF (text) | `pypdf` veya `pdfplumber` | Çoğu metin tabanlı PDF için yeterli |
| PDF (taranmış) | **`docling` + OCR (Tesseract)** | Eski tarama dokümanları için zorunlu |
| DOCX | `python-docx` | Tablo ve heading hiyerarşisi korunmalı |
| XLSX | `openpyxl` + custom serializer | Her sheet ayrı doküman, header'ları koru |
| PPTX | `python-pptx` | Slide bazında chunk |
| Çizim/CAD | **Faz 2** — vision model gerekir (Qwen2.5-VL) | Pilot fazda hariç tut |

**Önerim:** `docling` (IBM tarafından) tüm formatlar için tek noktadan dönüştürme yapıyor — pilot için bunu değerlendirmek 1 günlük iş, başarılı olursa custom pipeline'dan kurtarır.

### 3.2. Chunking stratejisi

- **Semantic chunking** birinci tercih (LlamaIndex'te `SemanticSplitterNodeParser`)
- Fallback: heading-aware structural chunking (heading hiyerarşisi korunur)
- Chunk boyutu: **512 token + 64 overlap** başlangıç, deneyle iterate
- Her chunk'a metadata: dosya yolu, sayfa, başlık zinciri, son değişiklik tarihi

### 3.3. Metadata indeksleme

Chunk'ların yanına aşağıdaki metadata zorunlu olmalı (filtering için):
- `source_path` — dosya yolu (kaynak göstermek için)
- `doc_type` — pdf / docx / xlsx / ...
- `project` — proje adı (varsa, dosya yoluna göre çıkarılabilir)
- `last_modified` — son değişiklik
- `language` — TR / EN / mixed
- `heading_path` — "Bölüm 3 > 3.2 Fatigue Analizi" gibi

Metadata sayesinde "sadece X projesinin son 6 aylık dokümanlarında ara" gibi sorgular mümkün olur.

---

## 4. Faz Planı

### Faz 0 — Hazırlık (Hafta 0, paralel)
- [ ] BT ile inference engine üzerinde **OpenAI-compatible API endpoint** açılması
- [ ] Pilot için **GPU/CPU + storage tahsisi** (vector DB + embedding inference)
- [ ] Pilot ekibinin **veri seti tanımlanması** (~500–2000 doküman, ortak diskte path)
- [ ] **Veri sınıflandırma değerlendirmesi** — gizli/IP içeren dokümanlar pilot dışı tutulacak mı?

### Faz 1 — MVP (Hafta 1–3)
- Hafta 1: Ingestion pipeline (format dönüştürme + chunking + metadata)
- Hafta 2: Qdrant kurulum + bge-m3 ile indeksleme + basit retriever
- Hafta 3: LLM entegrasyonu + minimal UI (Streamlit/Gradio yeterli) + ilk sorgular

**Çıktı:** Çalışan MVP, kaynak referanslı yanıt veriyor, tek kullanıcı için canlı.

### Faz 2 — Kalite ve Geri Bildirim (Hafta 4–6)
- Hafta 4: **Reranker** ekleme + hibrit search aktif (BM25 + dense)
- Hafta 5: Pilot ekibe açma, **structured feedback** toplama (👍/👎 + serbest yorum)
- Hafta 6: Yanlış cevap analizi, prompt iyileştirme, edge case'ler

**Çıktı:** Pilot ekibin gerçek sorgularıyla test edilmiş, ölçülen doğruluk metriği olan sistem.

### Faz 3 — Genişletme (Ay 2–3, opsiyonel)
- Açık WebUI entegrasyonu (mevcut webchat'in içine)
- Daha geniş kullanıcı kitlesi (2–3 ekip)
- Otomatik re-indexing (dosya değiştikçe vector DB güncellensin)
- Vision model (Qwen2.5-VL) — çizim okuma için
- Conversation memory (multi-turn)

---

## 5. Donanım/Kaynak Tahmini

### Embedding inference (bge-m3)
- ~2.3 GB model, **1× orta seviye GPU** (örn. T4, A10) yeter, hatta CPU bile dayanır (yavaş)
- 2000 doküman × ~10 chunk/doc = 20K chunk → ~10 dakika ilk indeksleme

### Vector DB (Qdrant)
- 20K chunk × 1024 dim = ~80MB vektör verisi + payload
- **2 GB RAM, 1 vCPU** yeterli — Docker container, ayrı sunucu gerek yok

### LLM inference
- Zaten kurulu mevcut HPC üzerinden API çağrısı — **ek donanım gerekmiyor**

### Toplam pilot için ek donanım: minimal — büyük ihtimal mevcut altyapıya sıkıştırılabilir.

---

## 6. Başarı Metrikleri

Pilot başarısının ölçülebilir olması kritik. Önerdiğim metrikler:

| Metrik | Hedef (Faz 1 sonu) | Hedef (Faz 2 sonu) |
|--------|-------------------|-------------------|
| **Top-5 retrieval recall** (gold set üzerinde) | %70 | %85 |
| **Cevap doğruluk oranı** (manuel değerlendirme, 50 sorgu) | %60 | %80 |
| **Ortalama yanıt süresi** | <8 sn | <5 sn |
| **Kullanıcı tatmini** (👍 oranı) | %50 | %70 |
| **Hallucination oranı** (kaynak desteği olmayan iddia) | <%20 | <%10 |

Pilot bitiminde bu rakamların hangi seviyede olduğu, **production'a geçiş veya pivot** kararını verir.

### Gold set hazırlığı
Pilot ekipten 30–50 adet "tipik soru + beklenen cevap + ilgili kaynak doküman" üçlüsü toplanmalı. Bu set hem geliştirme sırasında benchmark olur, hem de regresyon testi için kullanılır.

---

## 7. Riskler ve Önlemler

| Risk | Olasılık | Etki | Önlem |
|------|----------|------|-------|
| Doküman kalitesi düşük (taranmış PDF, kötü format) | **Yüksek** | Yüksek | Faz 0'da örnek seti incele, OCR pipeline hazırla |
| Türkçe embedding kalitesi yetersiz | Orta | Yüksek | bge-m3 kullan; gold set ile early test |
| LLM hallucination (kaynak yokken yanıt uyduruyor) | **Yüksek** | Yüksek | Strict prompt: "kaynaklarda yoksa 'bilmiyorum' de"; context-grounded scoring |
| Inference engine API'si tutarsız/yavaş | Orta | Orta | BT ile SLA, fallback queue, async pattern |
| Hassas IP'nin pilot dataset'ine sızması | Orta | **Çok yüksek** | Faz 0'da explicit veri sınıflandırma; access control |
| Pilot ekipten yetersiz feedback | Orta | Orta | Champion belirleme, haftalık 30 dk feedback session |
| Scope creep (herkes kendi dokümanını ekletmeye çalışır) | **Yüksek** | Orta | Net "pilot scope" tanımı, bekleme listesi |

---

## 8. Ekip ve Effort

### Tahmin (6 hafta toplam)

| Rol | Effort | Notlar |
|-----|--------|--------|
| **Tech lead / ML mühendisi** | 0.6 FTE | Pipeline, retrieval, prompt engineering |
| **Backend / DevOps** | 0.3 FTE | Qdrant kurulum, API, deployment |
| **Frontend / UI** | 0.2 FTE | Streamlit veya Open WebUI customization |
| **Domain expert (pilot ekipten)** | 0.2 FTE | Gold set, evaluation, feedback |
| **BT (kapasite + güvenlik)** | 0.1 FTE | Erişim, GPU tahsisi |

**Toplam:** ~1.4 FTE × 6 hafta. Eğer iç ekip varsa yapılabilir; dışarı verilirse 1 senior ML mühendisi + 1 backend, 6 hafta proje kapsamı.

---

## 9. Karar Verilmesi Gerekenler (BT/Yönetim ile)

Pilot başlamadan önce netleştirilmesi gereken şeyler:

1. **Inference engine API erişimi** — açılacak mı, kim açacak, hangi auth ile?
2. **Pilot için doküman seti** — hangi ekip, hangi klasör, kaç doküman?
3. **Veri sınıflandırma** — ITAR / IP / gizlilik filtreleme nasıl yapılacak?
4. **Ekip atama** — yukarıdaki rolleri kimler dolduracak?
5. **Bütçe** — ek donanım yoksa sıfır marjinal maliyet, ek donanım gerekirse onay süreci?
6. **UI tercihi** — mevcut webchat'e entegre mi, yoksa standalone pilot UI mı?
7. **Pilot sonrası karar kriteri** — "başarılı" sayılması için hangi metrikler hangi seviyede olmalı?

---

## 10. Pilot Sonrası — Ne Olur?

**Eğer başarılıysa (metrikler tutarsa):**
- Faz 3'e geçilir: 2–3 ekip daha eklenir, mevcut webchat'e entegre edilir
- Otomatik re-indexing kurulur
- 6 ay içinde tüm AR-GE'ye yayılır
- Teamcenter entegrasyonu için Faz 4 planlanır

**Eğer kısmen başarılıysa (bazı metrikler düşük):**
- Sorunlu alan tespit edilir (retrieval mi, generation mi, doküman kalitesi mi)
- Hedeflenmiş iyileştirme: reranker, fine-tuning, doküman temizleme
- 2–3 ay ek geliştirme sonrası tekrar değerlendirme

**Eğer başarısızsa:**
- Sebep analizi yapılır, açık raporlanır
- Alternatif yaklaşımlar değerlendirilir (örn. Teamcenter Copilot satın alma)
- Yatırım: 6 hafta ekip zamanı — kabul edilebilir öğrenme maliyeti

---

## Özet (TL;DR)

- **Stack:** Qdrant + bge-m3 + LlamaIndex + mevcut LLM (gpt-oss-120b)
- **Süre:** 6 hafta (3 hafta MVP + 3 hafta kalite/feedback)
- **Effort:** ~1.4 FTE
- **Donanım:** Mevcut altyapıya sıkışır, ek maliyet minimal
- **İlk eylem:** BT ile inference API erişimi + pilot doküman seti tanımı
- **Başarı kriteri:** %85 retrieval recall, %80 cevap doğruluğu, %70 kullanıcı tatmini

Sorular ve yorumlar için ben her zaman müsaitim.

— Emre
