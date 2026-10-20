# Analisis Sentimen Komunitas CS2 dan VALORANT di Reddit

Workspace penelitian untuk **“Analisis Perbandingan Sentimen Komunitas Pemain Counter-Strike 2 dan VALORANT pada Reddit Menggunakan Algoritma Naive Bayes”**.

Proyek ini menggunakan komentar Reddit sebagai unit analisis. VADER dipakai untuk menghasilkan **saran label awal** yang perlu ditinjau, sedangkan TF-IDF dan Multinomial Naive Bayes digunakan sebagai baseline klasifikasi tiga kelas: `positive`, `neutral`, dan `negative`.

## Struktur workspace

```text
data/
  raw/                  # Data Reddit mentah, misalnya reddit_comments_raw.json
  processed/            # CSV saran label dan CSV label final
reports/                # Laporan distribusi dan evaluasi model
src/
  prepare_labels.py     # Menghasilkan saran label VADER untuk ditinjau
  sentiment_analysis.py # Training dan evaluasi Naive Bayes per game
tests/                  # Tes otomatis
README.md
requirements.txt
pyproject.toml
```

Data lokal di `data/raw/`, `data/processed/`, dan keluaran di `reports/` diabaikan oleh Git. Jangan commit dataset yang berisi informasi pribadi atau data yang tidak berhak dipublikasikan.

## Persiapan environment

Gunakan Python 3.10 atau lebih baru. Di PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Jika aktivasi environment diblokir, gunakan `.\.venv\Scripts\python.exe` untuk menjalankan perintah Python.

## 1. Siapkan saran label

Letakkan file JSON mentah di `data/raw/reddit_comments_raw.json`. JSON harus berupa daftar objek komentar, dengan setidaknya kolom `game` dan `comment_text`. Nama game yang dikenali adalah `CS2` dan `VALORANT`.

Jalankan:

```powershell
python -m src.prepare_labels
```

Secara default, skrip membaca `data/raw/reddit_comments_raw.json` dan menghasilkan `data/processed/reddit_comments_label_review.csv`. Opsi file input/output:

```powershell
python -m src.prepare_labels --input data/raw/reddit_comments_raw.json --output data/processed/reddit_comments_label_review.csv
```

CSV tinjauan memuat:

- `sentiment_suggestion`: saran VADER (`positive`, `neutral`, atau `negative`)
- `vader_compound`: skor VADER dari -1 sampai 1
- `sentiment`: sengaja kosong untuk diisi dengan label final setelah peninjauan
- `review_status`: awalnya `needs_review`
- `game`, `text`, dan metadata komentar

Buka CSV tersebut di spreadsheet. Tinjau saran terhadap isi komentar dan konteksnya, isi kolom `sentiment`, kemudian ubah `review_status` menjadi `reviewed`. Simpan salinan final sebagai `data/processed/reddit_comments_labeled.csv`. Jangan menimpa JSON mentah. Skrip juga tidak menimpa file CSV tinjauan yang sudah ada.

**Penting:** VADER adalah alat bantu leksikon, bukan anotator yang selalu benar. Slang game, sarkasme, komentar sangat singkat, dan teks di luar topik dapat menghasilkan saran keliru. Label yang hanya menyalin VADER bukan ground truth independen; evaluasi model terhadap label tersebut mengukur kesesuaian dengan saran VADER, bukan akurasi terhadap anotasi manusia yang tervalidasi.

## 2. Format CSV berlabel

Kolom wajib untuk analisis:

| Kolom | Nilai |
|---|---|
| `game` | `cs2` atau `valorant` |
| `text` | Teks komentar |
| `sentiment` | `positive`, `neutral`, atau `negative` |

Kolom lain boleh dipertahankan. Jika ada kolom `review_status`, analisis akan menolak baris yang masih berstatus `needs_review`. Kedua game harus tersedia dan setiap kelas sentimen memerlukan setidaknya dua contoh per game; untuk split stratified 80:20, jumlah lebih besar mungkin diperlukan agar seluruh kelas terwakili di train dan test.

## 3. Training dan evaluasi

Jalankan analisis dengan pembagian stratified 80% training dan 20% testing:

```powershell
python -m src.sentiment_analysis --input data/processed/reddit_comments_labeled.csv --test-size 0.2 --random-state 42
```

Secara default, laporan disimpan di:

- `reports/sentiment_distribution.csv` — jumlah dan proporsi label per game.
- `reports/model_metrics.json` — akurasi, macro/weighted F1, laporan per kelas, dan confusion matrix.

Model dibuat dan dievaluasi secara terpisah untuk CS2 dan VALORANT. Gunakan prosedur pengumpulan, penyaringan, dan anotasi yang sebanding saat membandingkan kedua komunitas.

## Hasil run terakhir

Run terakhir yang tersimpan menggunakan 5.000 komentar per game, pembagian 4.000 training dan 1.000 testing untuk masing-masing game, serta `random_state=42`.

| Game | Akurasi | Macro F1 |
|---|---:|---:|
| CS2 | 0,483 | 0,304 |
| VALORANT | 0,512 | 0,268 |

Model baseline menunjukkan performa terbatas, terutama pada kelas negatif dan netral. Lihat confusion matrix serta metrik per kelas di `reports/model_metrics.json`; jangan hanya mengandalkan akurasi. Hasil run ini mencerminkan label di dataset saat analisis dijalankan dan bukan otomatis hasil anotasi manusia independen.

## Menjalankan tes

```powershell
python -m unittest discover -s tests -v
```

## Batasan penelitian

- Naive Bayes belajar dari label yang diberikan; ia tidak mengubah label otomatis menjadi ground truth.
- Pembagian acak stratified bukan evaluasi lintas waktu atau lintas penulis. Duplikasi atau komentar dari post yang sama dapat membuat data train dan test berkorelasi.
- TF-IDF dan Multinomial Naive Bayes adalah baseline. Dokumentasikan pilihan prapemrosesan dan keterbatasan untuk bahasa, slang, negasi, serta sarkasme.
- Sentimen komentar tidak membuktikan sebab-akibat antara game dan opini komunitas.
- Patuhi ketentuan Reddit, aturan subreddit, dan persyaratan etik yang berlaku; minimalkan penyimpanan informasi pribadi.
