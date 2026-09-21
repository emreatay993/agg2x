# HPC Static Parts — PDR konuşmacı notları

İki ana slayt ve bir kaynakça sayfası. Ana slayt başına yaklaşık 3 dakikalık anlatım; TR ve EN alternatif metinlerdir.

## Slayt 1 — PDR Odağı

TÜRKÇE — KONUŞMACI NOTLARI
Ana anlatım: yaklaşık 3 dakika. İngilizce metin aşağıda aynı anlatımın karşılığıdır; ikisi birlikte okunmaz.

Açılış
Parça sahiplerinin tasarım durumunu veya takvimini tekrar etmek yerine, yapısal analiz açısından modül genelinde ortaklaştırmamız gereken substantiation başlıklarını özetliyorum. Buradaki öncelikler kamuya açık servis deneyimi ve raporlardan türetilmiştir; kendi motorumuzda tespit edilmiş uygunsuzluklar değildir. Önceki ekip tarafından yapılmış çalışmaları yok saymak ya da her testi yeniden istemek gibi bir öneri sunmuyorum. Önce konfigürasyon, yükler, assumptions ve kanıtın geçerlilik alanını birlikte görünür hale getirmeyi öneriyorum.

Öncelik tablosunu nasıl okuyalım?
P1 ve P2, önerilen çalışma öncelikleridir; onaylı critical-parts listesi veya LLP beyanı değildir. Resmî sınıflandırma, gerçek parça/feature’ın failure chain’i, uçak üzerindeki sonucu ve kabul edilen program basis’i ile yapılmalı. Tek motorlu bir kurulum söz konusuysa, sustained thrust kaybının sonucu ayrıca değerlendirilir; programımızın kurulumunu burada varsaymıyoruz. Discharge casing ve primary supports satırı da yalnız bu fonksiyonlar subsection kapsamına veya arayüzüne tahsis edilmişse geçerlidir. [1,3]

Üç ana odak
Cases için pressure strength, cyclic life, alignment/clearance, blade containment ve fire ayrı sorulardır. Flange, bolt-hole, boss ve port gibi local feature’lar; gross stress iyi görünse bile sonucu belirleyebilir. VSV tarafında yalnız airfoil değil, retention hardware, bushing/bore, lever ve actuator/controls interface birlikte değerlendirilir. Ring position’ın doğru olması her vane’in doğru konumda ve retained olduğunun kanıtı değildir. Fixed stator tarafında da retention/orientation ve airfoil HCF ayrı hedeflerdir. [1–5,7,8]

Neden küçük feature’ları öne çıkarıyoruz?
Osan 2012 F110 incelemesinde yanlış yerleştirilen/reversed stator sectors, rotor-blade HCF ve sustained power kaybına uzanan bir mekanizma gösteriyor. CFM56-7B 2019 olayı ise VSV retaining hardware kaybının rotor damage ve titanium fire ile bağlantısını gösteriyor. Bunlar aynı arıza değildir; ortak ders, statik parçanın kendi gerilmesinden daha geniş bir failure chain’in incelenmesi gerektiğidir. Seals ve anti-rotation features için de destructive rotor contact, debris veya fire yolu varsa öncelik yükselir. [6,7,9]

Sağdaki gündem ve riskler
Önerim, önceki modelleri ve testleri yeniden kullanabilmek için onları bugünkü requirement ve konfigürasyonla eşleştirmek. Risk başlıkları bir audit bulgusu değil, screening sorularıdır: nominal stress dışındaki mekanizmalar kapsanıyor mu, rig–engine thermal/BC/damping farkları biliniyor mu ve literatürdeki katsayı ya da sınıflar yanlışlıkla doğrudan requirement’a dönüşüyor mu? Bir sonraki slayt bu soruları analiz–test planına çeviriyor.

EK AÇIKLAMA / SORU GELİRSE
• P1 seçimi bütün parçalar için aynı test paketi demek değildir; etkisi yüksek ve henüz yeterince desteklenmemiş mekanizmaya öncelik vermektir.
• Liste bir severity/probability risk skoru değildir. Kabul edilen FMECA ve ownership review sonucu değişebilir.
• Kaynak raporların tier ve kapsamı aynı değil. Bu sunum, D1’in açıkça yaptığı ayrımı esas alıyor; D2–D3’ün feature ve failure-chain bulgularını bunu desteklemek için kullanıyor. D4’ün fixed-stator için daha düşük başlangıç sırası ve D5’in bazı genel sınıf/katsayı ifadeleri aynen aktarılmadı; bu tercih D1’in konsolidasyonuyla uyumludur.

ENGLISH — SPEAKER NOTES
Main delivery: approximately 3 minutes. This is the English counterpart of the Turkish script, not an additional speech.

Opening
Rather than repeat the part owners’ design status or schedules, I am summarizing the cross-module substantiation topics from a structural-analysis perspective. These priorities come from the supplied public-source reviews and service experience; they are not findings of noncompliance on our engine. The proposal is not to disregard the previous team’s work or repeat every test. It is to make the configuration, loads, assumptions and validity of the existing evidence visible to the current team.

How to read the priority table
P1 and P2 are proposed work priorities, not an approved critical-parts list or an LLP declaration. Formal classification needs the actual part/feature failure chain, its installed-aircraft consequence and the accepted program basis. Where a single-engine installation is involved, loss of sustained thrust deserves explicit consideration; this presentation does not assume our installation. The discharge-casing and primary-support row applies only where those functions are allocated to our subsection or its interfaces. [1,3]

Three main areas
For cases, pressure strength, cyclic life, alignment/clearance, blade containment and fire are separate questions. Local features such as flanges, bolt holes, bosses and ports can govern the result despite acceptable gross stress. For VSVs, the assessment needs the complete retention, bushing/bore, lever and actuation/controls interface, not just the airfoil. Correct ring position does not prove that every vane is correctly located and retained. For fixed stators, correct retention/orientation and airfoil HCF are also separate objectives. [1–5,7,8]

Why highlight small features?
The 2012 Osan F110 investigation establishes a chain from misinstalled/reversed stator sectors to rotor-blade HCF and loss of sustained power. The 2019 CFM56-7B event links missing VSV retaining hardware to rotor damage and titanium fire. These are different events; the shared lesson is to assess the full failure chain rather than the static part’s own stress alone. Seal and anti-rotation features likewise escalate where destructive rotor contact, debris or fire is credible. [6,7,9]

Agenda and screening risks
I suggest mapping inherited models and tests to the present requirements and configuration so that valid evidence can be reused. The risk headings are screening questions, not audit findings: do we cover mechanisms beyond nominal stress, understand the rig-to-engine thermal/BC/damping differences, and avoid converting a literature factor or class directly into a program requirement? The next slide translates these questions into an analysis–test plan.

DISCUSSION SUPPORT
• P1 does not prescribe the same test package for every part. It prioritizes high-consequence mechanisms with unresolved evidence needs.
• This is not a quantified severity/probability risk matrix. The accepted FMECA and ownership review may change the list.
• The reports use different tiers and module boundaries. This presentation adopts D1’s explicit consolidated distinctions and uses D2–D3 for supporting feature/failure-chain lessons. D4’s lower initial fixed-stator ranking and D5’s blanket classifications/factors are not carried over unchanged; this follows D1’s stated reconciliation.

SOURCE MAP / KAYNAK EŞLEŞMESİ
D1: Executive findings; §§1–3, 5.1–5.3, 6.1–6.4, 9.
D2: §§1, 4.4, 5.1–5.3, 7. D3: §§4, 6–7.
D4: Executive summary; Scope and evidence base; Program gates. D5: criticality matrix and historical cases, with D1’s qualifications.
Primary-source crosswalk: cases [1,2,3,5]; VSV [4,7,8]; fixed stators [1,4,6]; seals/fire [7,9]; applicability [1,3,10].
PDR meeting actions and the treatment of inherited work are proposed for the team context supplied by the user, not statements about the current status of internal records.

---

## Slayt 2 — Kanıt Planı

TÜRKÇE — KONUŞMACI NOTLARI
Ana anlatım: yaklaşık 3 dakika. Bu slayt tamamlanmış iş listesi veya onaylanmış test taahhüdü değil, PDR için önerilen karar çerçevesidir.

PDR’de hedefimiz
Bu aşamada her qualification testinin bitmiş olmasını önermiyorum. Hangi requirement’ın hangi analiz, test veya inspection ile kapatılacağını; hangi mevcut kanıtın kullanılabileceğini; hangi girdinin ve kimin aksiyonunun eksik olduğunu birlikte tanımlamamız gerekiyor. “Uygulanabilir / koşullu / açık” etiketleri, önceki işlerin bugün hangi koşullarda kullanılabileceğini ayırmak için önerdiğim durumlar. Henüz hiçbir iç dokümana bu statüyü atamıyoruz.

Tablonun ilk iki satırı
Cases ve joints için actual pressure differential, metal-temperature field, mission transients, interface loads, joint preload ve material/process assumptions aynı load envelope içinde izlenebilir olmalı. Local LCF/DT veya life yöntemi, kabul edilen criticality ve basis’e bağlıdır. Pressure/cyclic rig’den gelen strain verisi bazı mekanizmaları doğrulayabilir; ancak rig gerçek engine thermal gradient’lerini üretmiyorsa metal-T verisi, thermal-model correlation ve farkın stress/clearance etkisinin değerlendirilmesi gerekir. Her farkın analizle otomatik kabul edilebileceğini söylemiyoruz. [1–3; D1 §6.1]

VSV ve fixed stators satırı, iki farklı paketi özetliyor: VSV için friction/wear/travel ve her vane’in retention’ı; fixed stator için assembly/retention ile modal/forced-response/HCF. Modal veya shaker testi modelin dinamik özelliklerini destekler, fakat aerodynamic forcing’i tek başına yeniden üretmez. Vibration survey için representative complete-module seçeneği bazı koşullarla mümkün olabilir; yalnız bir stage’i test etmek bununla eşdeğer değildir. [4,6–8]

Diğer mekanizmalar
Seals ve anti-rotation için actual material pair, clearance, thermal growth, permitted motion ve retention birlikte ele alınmalı; uygun rub/retention ve bond/braze evidence seçilmeli. Surge, blade-out/unbalance ve internal fire aynı acceptance statement altında kapatılamaz. Gerekli gösterim ve analiz alternatifi her objective için ayrıca belirlenir. Destructive fault’ları gelişigüzel full engine üzerinde yaratmayı değil, güvenli ve temsil gücü gerekçelendirilmiş bir verification planı öneriyoruz. [5,7,9; D1 §§6.2,6.4]

Yönetimden ve ekipten beklenen ortak karar
Önce aday P1 kapsamını ve modül sınırını teyit edelim. Sonra her açık item için evidence owner, hedef tarih ve closure yolunu tanımlayalım. Part owner design/requirement closure bağlamını; FEA modellerin yük izlenebilirliğini, assumptions ve correlation kapsamını; rotor/aero/controls ekipleri kendi forcing, displacement ve fault girdilerini; test ve material/process ekipleri ilgili fiziksel kanıtı birlikte netleştirsin. Bu öneri mevcut organizasyonun onaylı RACI’si yerine geçmez.

Kapanış
Benim önerdiğim ortak çıktı, requirement–MoC–evidence–owner matrisi ve öncelikli test/correlation planıdır. Böylece geçerli eski çalışmayı korur, tekrarlı işi azaltır ve kalan ihtiyaçları net bir karar listesine dönüştürürüz. Review/Chief Engineering onayı governance açısından önemlidir; ancak fiziksel validation verisinin veya modelin geçerlilik alanının yerine geçmez. [2; D1 §7]

EK AÇIKLAMA / SORU GELİRSE
• Pressure test tek başına containment veya internal metal-fire gösterimi değildir. Ortak hardware kullanılsa da acceptance criteria ayrıdır.
• Test talep etmeden önce mevcut configuration, material/process, load coverage, instrumentation ve correlation raporları incelenmeli. Uygun verinin kredilendirilmesi mümkündür; bunun extent’i dokumente edilir.
• Thermal/BC/damping uncertainty kritik sonucu değiştiriyorsa ilave representative test gerekebilir.
• Sunumda genel geçer pressure factors, life multipliers, VSV angle/wear limits veya defect sizes verilmez. Bunlar kontrollü program kaynağından gelmelidir.

ENGLISH — SPEAKER NOTES
Main delivery: approximately 3 minutes. This slide is a proposed PDR decision framework, not a completion report or an approved test commitment.

Our objective at PDR
I am not suggesting that all qualification testing should be complete now. We need to agree how each requirement will be closed by analysis, test or inspection, what existing evidence can be credited, and which input or action is still needed. “Applicable / conditional / open” are proposed labels for the reuse status of inherited evidence. No internal document has been assigned one of those statuses by this presentation.

The first two table rows
For cases and joints, the actual pressure differential, metal-temperature field, mission transients, interface loads, joint preload and material/process assumptions need traceability within one load envelope. The local LCF/DT or life methodology depends on the accepted criticality and basis. Strain data from a pressure/cyclic rig may validate identifiable mechanisms. If the rig cannot reproduce engine thermal gradients, metal-temperature data, thermal-model correlation and assessment of the resulting stress/clearance differences are needed. We are not claiming that every mismatch can automatically be accepted analytically. [1–3; D1 §6.1]

The VSV/fixed-stator row summarizes two distinct packages: VSV friction/wear/travel and individual-vane retention; fixed-stator assembly/retention plus modal, forced-response and HCF evidence. Modal or shaker testing supports the model’s dynamic properties but does not reproduce aerodynamic forcing by itself. A representative complete-compressor-module vibration survey may be an acceptable alternative under specified conditions; an isolated stage is not equivalent. [4,6–8]

The other mechanisms
For seals and anti-rotation hardware, the actual material pairing, clearances, thermal growth, permitted motion and retention need a combined assessment, supported by appropriate rub/retention and bond/braze evidence. Surge, blade-out/unbalance and internal fire cannot be closed under a single acceptance statement. The demonstration and any analysis alternative are selected for each objective. The proposal is a safe, justified verification plan, not arbitrary creation of destructive faults on a full engine. [5,7,9; D1 §§6.2,6.4]

Joint decisions requested
First confirm the candidate P1 scope and module boundary. Then assign an evidence owner, target date and closure route to each open item. Part owners provide the design/requirement closure context; FEA addresses model assumptions, load traceability and correlation; rotor/aero/controls teams provide the relevant forcing, displacement and fault inputs; test and material/process teams define the corresponding physical evidence. This is a suggested interface discussion, not a replacement for an approved organizational RACI.

Close
The proposed shared output is one requirement–MoC–evidence–owner matrix and a prioritized test/correlation plan. That preserves valid inherited work, reduces duplication and turns remaining needs into a clear decision list. A review or Chief Engineering approval is important governance, but does not replace physical validation data or extend a model beyond its demonstrated validity. [2; D1 §7]

DISCUSSION SUPPORT
• Pressure testing alone does not establish blade containment or internal metal-fire resistance. Acceptance criteria remain separate even if test hardware is shared.
• Before commissioning a new test, review the available configuration, material/process, load coverage, instrumentation and correlation records. Applicable data may be credited, with its extent documented.
• Additional representative testing may be needed where thermal/BC/damping uncertainty changes the critical conclusion.
• The slides do not prescribe universal pressure factors, life multipliers, VSV angle/wear limits or defect sizes. Those need a controlled, applicable program source.

SOURCE MAP / KAYNAK EŞLEŞMESİ
D1: §§4, 6.1–6.6, 7.1–7.2, 8–8.2; D2: substantiation matrix §6 and cross-cutting recommendations §7; D3: analyst implications §7. D4: Program gates and Minimum documentation set. D5: development verification matrix only as qualified by D1.
Primary sources [1–5] support the method/acceptance boundaries; [6–9] support the mechanisms requiring targeted evidence. [10] provides the EASA edition/applicability entry point.
The proposed ownership discussion, status labels and PDR meeting outputs are presentation recommendations tailored to the user’s team context. No internal schedule, test result, assigned owner or maturity percentage is asserted.

---

## Slayt 3 — Bağlantılı Kaynakça

TÜRKÇE — KONUŞMACI NOTLARI
Bu sayfa ana anlatıma ek bir appendix’tir; tek tek okunması gerekmiyor. İlk iki sayfadaki numaralar kaynakçaya, bu sayfadaki başlıklar kaynak dokümanlara bağlıdır. Soru gelirse mekanizma veya ilgili clause’u açmak için kullanılabilir.

Sunumun doğrudan temeli kullanıcı tarafından sağlanan beş rapordur. D1’in explicit consolidation ve caveat’ları esas alındı. Diğer raporların öncelik ve kapsam farklılıkları ortalama alınmadı. Örneğin D2 discharge casing/CRF’yi scope dışında tutarken D3 dahil eder; bu sunum onları conditional scope olarak gösterir. Fixed-stator retention başlangıçta P1 olarak gösterilir; bu, D1’in Osan olayıyla gerekçelendirdiği consolidation’dır.

[1] public military guidance’dır; kendi başına binding requirement değildir. [2–5,9] ve [10] sivil criterion/guidance benchmark’larıdır; askeri programımıza kendiliğinden uygulanmış sayılmaz. [6] USAF’nin birincil AIB raporunun üçüncü taraf aynasıdır. [7] NTSB birincil final raporudur. [8] belirli CFM56 modelleri için yayımlanmış bir AD’dir; yeni motorumuzun bakım talimatı değildir.

EASA index’te Amendment 8 ve correction metadata görülür. D1’in detailed CS-E clause crosswalk’ı archived Amendment 7 üzerinden hazırlanmıştır. Bu sunum, tüm Amendment 8 clauses yeniden doğrulanmış gibi bir iddia taşımaz; uygulanacak edition ve tam clause metni program tarafından teyit edilmelidir. Hyperlink kontrolü, kullanıcı raporlarındaki bütün iddiaların baştan independent re-verification’ı değildir.

Public-source bulguları proje evidence’ı olarak sunmuyoruz. Önceki ekibin gerçek analiz/test kapsamı, yalnız iç kayıtlar incelenerek belirlenebilir. Toplantı önerileri, kullanıcı tarafından tarif edilen PDR ve ekip ortamına göre hazırlanmıştır.

ENGLISH — SPEAKER NOTES
This is a reference appendix, not a third narrative page to read aloud. Numbers on the first two slides lead here; source titles lead to the original documents. Use it when a question requires the underlying mechanism or clause.

The immediate basis is the five supplied reports. D1’s explicit consolidation and caveats are adopted; differing report priorities and scopes are not averaged. D2 excludes the discharge casing/CRF, whereas D3 includes it; this presentation therefore treats it as conditional scope. Initial P1 priority for fixed-stator retention follows D1’s consolidation supported by the Osan investigation.

[1] is public military guidance, not automatically a binding requirement. [2–5,9] and [10] are civil criteria/guidance benchmarks, not an assumed military program basis. [6] is a primary USAF AIB report on a third-party mirror. [7] is a primary NTSB final report. [8] is an AD applicable to specified CFM56 models, not a maintenance instruction for our new engine.

The EASA index identifies Amendment 8 and correction metadata. D1’s detailed CS-E clause crosswalk was checked against archived Amendment 7. This presentation does not claim a full re-verification of every Amendment 8 clause; the adopted edition and complete clause text require program confirmation. Checking hyperlinks is not a fresh independent audit of every claim in the supplied reports.

Public-source findings are not presented as project evidence. The actual extent of the previous team’s analyses and tests can only be established from internal records. The meeting actions are proposed for the user’s stated PDR/team context.

D1–D5 — REPORT BASIS / RAPOR TEMELİ
D1 — HPC_Static_Verified_Consolidated_Report(2).docx
HPC Static Parts: Verified consolidation, criticality and substantiation plan; 21 September 2026. Primary synthesis for priority/formal-class distinctions, pressure/life caveats, five separate case objectives, analysis sufficiency and PDR gates. Key sections: Executive findings, §§2–8 and 9.
D2 — HPC_Static_Parts_Criticality_Report.md
Critical Static Parts of an F110-Class High-Pressure Compressor: Criticality Ranking and Substantiation Needs; 21 September 2026. Supporting material: §§1, 4.4, 5–7. Explicitly excludes CRF/diffuser; includes VSV actuation/FADEC. Numerical guidance is not transferred as a universal design requirement.
D3 — hpc-static-parts-criticality-r1.docx
HPC Static Parts — What Is Critical and Why; rev r1, 21 September 2026. Supporting material: §§4, 6–7. Includes compressor discharge/CRF. Its tiers are a literature-review judgement, not formal classification.
D4 — Static Compressor Parts Requiring Enhanced Substantiation in a High-Performance Military-Style Turbofan
Original English report retrieved from the user’s Library (uploaded DOCX copy: 46f99866-bc3d-45a5-baef-af79c673ac4b.docx). Used for the initial priority framework, evidence chain, program gates and documentation set, with D1’s subsequent reconciliation.
D5 — Compressor Static Parts Criticality Analysis
Original Turkish report: Yüksek Performanslı Askeri Gaz Türbini Motorlarında Kompresör Statik Bileşenlerinin Kritiklik Analizi ve ENSIP Doğrulama Programı. Retrieved from the user’s Library (uploaded DOCX copy: 669d927b-f0d8-4dff-8653-b30d905e323a.docx). Historical failure mechanisms and the verification matrix are interpreted using D1’s explicit corrections. Unverified universal materials, temperatures, test factors and LLP claims are not carried into the slides.

PRIMARY REFERENCES / BİRİNCİL DAYANAKLAR
[1] MIL-HDBK-1783B, Chg 2 — ENSIP
A.4.7; A.4.10.9; A.4.13.3.2; A.4.17 Criticality, cases, HCF ve inspectability; guidance.
https://quicksearch.dla.mil/WMX/Default.aspx?token=449683

[2] 14 CFR 33.64 — Pressurized static parts
33.64(a)–(b) Combined loads ve test / validated-analysis yolu.
https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-33/subpart-E/section-33.64

[3] FAA AC 33.70-1, Chg 1 — Static LLP
7.c(1); 8.e • 14 CFR 33.70 için guidance HP cases, life ve inspection kredisi koşulları.
https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_33_70-1_Chg_1.pdf

[4] FAA AC 33.83-2B — Vibration test
7.3; 8.7; 10; 12 Module / engine survey, vane tolerances ve faults.
https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC-33.83-2B.pdf

[5] 14 CFR 33.94 — Containment / unbalance
33.94(a)–(b) Blade-release gösterimi; alternatif yöntemin sınırı.
https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-33/subpart-F/section-33.94

[6] USAF AIB — F110 / Osan, 2012
F-16CM 90-0771 • printed pp. 19–21 Sector misassembly → rotor HCF → thrust loss.
https://www.airandspaceforces.com/PDF/AircraftAccidentReports/Documents/2012/032112_F-16CM_Osan.pdf

[7] NTSB ENG19IA013 — CFM56-7B, 2019
Final report • Analysis / Probable Cause Retention kaybı → rotor damage → titanium fire.
https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain/GenerateNewestReport/99006/pdf

[8] FAA AD 2017-14-08 — CFM56 VSV
Discussion / Unsafe Condition Bore corrosion ve kısıtlanan vane travel.
https://www.federalregister.gov/documents/2017/07/14/2017-14545/airworthiness-directives-cfm-international-sa-turbofan-engines

[9] 14 CFR 33.17 — Fire protection
33.17(a) Internal fire ve yapısal / hazardous etkiler.
https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-33/subpart-B/section-33.17

[10] EASA CS-E — Publication index
CS-E 515 / 520 / 640 / 650 / 810 cross-reference Uygulanacak edition ve clause metni teyit edilmeli.
https://www.easa.europa.eu/en/document-library/certification-specifications/group/cs-e-engines

Traceability note: The deck is a PDR planning summary. It does not provide proprietary F110 design values, an approved parts list, test results, formal classification, a numerical risk assessment or a program compliance finding.
