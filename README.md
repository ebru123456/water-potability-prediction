# Water Potability Prediction

Bu proje, `water_potability.csv` veri setindeki su kalitesi özelliklerini kullanarak suyun içilebilir olup olmadığını (`Potability`: `0` = İçilemez, `1` = İçilebilir) tahmin eden bir makine öğrenmesi çalışmasıdır.

## Yapılan İşlemler
* **Keşifsel Veri Analizi (EDA):** Veri özet istatistikleri (`describe()`), sınıf dağılımı pasta grafiği (`Plotly`), sınıflara göre KDE dağılım grafikleri (`Seaborn` / `Matplotlib`) ve eksik veri matrisi (`missingno`) incelendi.
* **Eksik Veri Doldurma:** `ph`, `Sulfate` ve `Trihalomethanes` sütunlarındaki eksik değerler sütun ortalaması (`mean`) ile dolduruldu.
* **Veri Bölme ve Normalizasyon:** Veri seti `%70` eğitim ve `%30` test olarak ayrıldı (`random_state=42`) ve sütun bazında Min-Max normalizasyon uygulandı.
* **Modelleme ve Değerlendirme:** `DecisionTreeClassifier` (`max_depth=10`) ve `RandomForestClassifier` (`n_estimators=200`, `max_depth=12`) modelleri eğitilip Confusion Matrix ve Karar Ağacı grafikleriyle görselleştirildi.

## Model Sonuçları

| Model | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: |
| **DTC** | 0.6348 | 0.5200 | 0.2486 | 0.3364 |
| **RF** | 0.6867 | 0.7042 | 0.2732 | 0.3937 |

**Öne Çıkan Bulgu:** **Random Forest (`RF`)**, tüm değerlendirme metriklerinde tek bir karar ağacına (`DTC`) kıyasla daha yüksek performans göstermiştir. Her iki modelde de `Recall` değerinin görece düşük olması, içilebilir (`1`) sınıfı yakalamada temkinli davranıldığını ve veri setindeki gürültü ile ortalama doldurma etkisini göstermektedir.

## Kullanılan Kütüphaneler
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `plotly`
* `missingno`
* `scikit-learn`

## Çalıştırma
1. Gerekli kütüphaneleri yükle:
   ```bash
   pip install numpy pandas matplotlib seaborn plotly missingno scikit-learn
