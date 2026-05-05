🎓 Student Performance Prediction (Regression)
Bu proje, öğrencilerin demografik bilgilerini, çalışma alışkanlıklarını ve geçmiş başarılarını analiz ederek final sınav notlarını (final_score) 
tahmin etmeyi amaçlayan bir makine öğrenmesi çalışmasıdır. Veri setini temizlemekten model optimizasyonuna kadar uçtan uca bir pipeline içermektedir.

🚀 Proje Özeti
Veri setindeki her bir öğrencinin başarı grafiğini etkileyen faktörleri (eğitim seviyesi, ders çalışma süresi, katılım oranı vb.) 
inceledim ve bu verilerle bir Ridge Regresyon modeli kurdum. Modelin başarısını artırmak için hiperparametre optimizasyonu ve özellik mühendisliği adımlarını uyguladım.

🛠️ Neler Yapıldı?
1. Veri Temizleme & Hazırlık
Gereksiz Kolonlar: Tahminleme sürecine katkısı olmayan student_id ve regresyon modelini bozabilecek (target ile doğrudan ilişkili) passed kolonlarını eledim.

Eksik Veri (Imputation): parent_education kolonundaki eksik değerleri, veri setindeki en sık tekrar eden (mode) değerle doldurarak veri kaybını önledim.

Veri Tiplerini Dönüştürme:

gender, extracurricular ve internet_access gibi ikili (binary) verileri 0 ve 1'e mapledim.

parent_education kolonunu, eğitimin hiyerarşik yapısını korumak adına Ordinal Encoding ile (High School < Bachelor < Master < PhD) sayısal hale getirdim.

2. Özellik Mühendisliği & Ölçeklendirme
ColumnTransformer: Farklı veri tiplerine (kategorik ve numerik) aynı anda müdahale edebilmek için pipeline yapısına uygun bir transformasyon uyguladım.

StandardScaler: Modelin katsayıları daha dengeli yorumlayabilmesi ve daha hızlı yakınsaması için tüm verileri standart normal dağılıma çektim.

3. Model Seçimi ve Tuning
Başlangıçta birçok regresyon modelini hızlıca test etmek için LazyRegressor kullandım (kodda opsiyonel olarak mevcut).

Final modeli olarak Ridge Regression seçildi.

GridSearchCV kullanılarak en uygun alpha parametresi belirlendi (Cross-validation ile doğrulandı).

📊 Sonuçlar ve Değerlendirme
Modelin başarısını ölçmek için R² (R-squared) ve RMSE (Root Mean Squared Error) metriklerini kullandım. Ayrıca projenin sonunda 
hangi faktörün (örneğin çalışma saatleri veya devamsızlık) başarı üzerinde daha etkili olduğunu görmek için katsayı (coefficient) analizini tablo haline getirdim.