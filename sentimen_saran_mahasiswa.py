import pandas as pd # baca, ngelola dataset
import re # membersihkan teks (regex: hapus angka & simbol)

# Preprocessing teks
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # membuat stemmer Bahasa Indonesia
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory # mengambil daftar kata umum (dan, yang, di, ke, dll)

# modeling
from sklearn.model_selection import train_test_split # membagi data latih & uji
from sklearn.feature_extraction.text import TfidfVectorizer # mengubah teks ke angka
from sklearn.naive_bayes import MultinomialNB # algoritma klasifikasi teks
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #akurasi, precision, recall, f1, detail kesalahan klasifikasi

# 1. LOAD DATASET
data = pd.read_csv("dataset_komentar_mahasiswa_variasi.csv")

#jumlah total data dan jumlah saran & bukan saran untuk cek keseimbangan data
print("Jumlah data awal:", len(data))
print("Distribusi label awal:")
print(data["label"].value_counts())
print("-" * 50)

# PREPROCESSING
factory = StemmerFactory()
stemmer = factory.create_stemmer() # membuat variabel stemmer bahasa

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words()) # mengambil daftar stopword/kata umum.

def preprocess(text):
    text = text.lower() # case folding agar semua make huruf kecil
    text = re.sub(r'[^a-z\s]', '', text) # menghapus angka, tanda baca, simbol
    tokens = text.split() # tokenisasi agar kalimat jadi kata-kata
    tokens = [t for t in tokens if t not in stopwords] # menghapus kata tidak bermakna (dan, yang, ke, dari)
    text = " ".join(tokens) # digabungkan lagi jadi kalimat
    text = stemmer.stem(text) # menghilangkan kata berimbuhan (perbaikan - baik, ditambahkan - tambah)
    return text

data["komentar_clean"] = data["komentar"].apply(preprocess) # jika alatnya sudah siap, ambil bahannya di dataset dan mulai memasak

# 3. FUNCTION TRAIN & EVALUATE
def train_evaluate(df):
    tfidf = TfidfVectorizer() # ubah teks jadi angka vector
    X = tfidf.fit_transform(df["komentar_clean"]) # mengubah komentar jadi vector + transform data agar lebih terstruktur sesuai untuk analisis
    y = df["label"] # target klasifikasi

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, # membagi data 80% latih, 20% uji
        random_state=42, # nilai seed umum untuk memastikan proses pemecahan dataset tetap konsisten
        stratify=y # proporsi label tetap seimbang (semua kelas diuji agar tidak terjadi bias dataset)
    )

# buat model naive bayes & latih model dengan data latih
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) # prediksi data uji

    print("Akurasi:", accuracy_score(y_test, y_pred)) # menghitung akurasi model
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred)) # nampilin precision,recal,F1

    cm = confusion_matrix(y_test, y_pred) # menghitung kesalahan klasifikasi
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_) # nampilin dalam bentuk tabel
    print("\nConfusion Matrix:")
    print(cm_df)

    return model, tfidf # dikembalikan untuk digunakan lagi

# 4. TRAIN & EVALUATE (DATA AWAL)
print("=== HASIL SEBELUM DATA OTOMATIS ===")
model, tfidf = train_evaluate(data) # melatih model pada data
print("-" * 50)

# 5. INPUT USER → LABEL OTOMATIS (PSEUDO-LABELING)
CONFIDENCE_THRESHOLD = 0.75 # ambang batas kepercayaan >= 75%

while True:
    komentar_user = input(
        "\nMasukkan komentar mahasiswa "
        "(ketik 'exit' untuk selesai input): "
    )

    if komentar_user.lower() == "exit":
        break

    komentar_clean = preprocess(komentar_user)
    komentar_tfidf = tfidf.transform([komentar_clean])

    # Prediksi + confidence
    prob = model.predict_proba(komentar_tfidf)[0] # probabilitas tiap kelas diambil
    pred_label = model.classes_[prob.argmax()] # label dengan probabilitas tertinggi
    confidence = prob.max() # tingkat keyakinan model

    print(f"Hasil Prediksi: {pred_label}")
    print(f"Confidence     : {confidence:.2f}")

    # Simpan hanya jika confidence tinggi
    if confidence >= CONFIDENCE_THRESHOLD:
        data.loc[len(data)] = [
            komentar_user,
            pred_label,
            komentar_clean
        ]
        print("✔ Data otomatis ditambahkan ke dataset")
    else:
        print("✖ Confidence rendah → data tidak disimpan")

# 6. SIMPAN DATASET BARU
data.to_csv("dataset_komentar_mahasiswa_updated.csv", index=False)
print("\nDataset baru disimpan!")
print("Jumlah data sekarang:", len(data))
print("-" * 50)

# 7. TRAIN & EVALUATE ULANG (SETELAH DATA OTOMATIS)
print("=== HASIL SETELAH DATA OTOMATIS ===")
model, tfidf = train_evaluate(data)