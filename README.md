# Analisis Perbandingan Sentimen Komunitas CS2 dan VALORANT

Workspace penelitian berjudul **“Analisis Perbandingan Sentimen Komunitas Pemain Counter-Strike 2 dan VALORANT pada Reddit Menggunakan Algoritma Naive Bayes”**.

## Tujuan dan rancangan

Penelitian ini membandingkan distribusi sentimen positif, netral, dan negatif pada unggahan Reddit tentang Counter-Strike 2 dan VALORANT. Pipeline awal menggunakan TF-IDF dan Multinomial Naive Bayes, lalu melaporkan distribusi label serta metrik evaluasi untuk masing-masing komunitas.

Alur kerja yang disarankan:

1. Tentukan periode pengambilan data, subreddit, kriteria inklusi/eksklusi, dan unit analisis (misalnya satu unggahan atau satu komentar).
2. Kumpulkan data menggunakan API dan ketentuan Reddit yang berlaku serta aturan subreddit. Jangan mengumpulkan informasi pribadi yang tidak diperlukan.
3. Anonimkan atau hapus nama pengguna, deduplikasi, dan dokumentasikan langkah prapemrosesan.
4. Beri label sentimen secara manual dengan pedoman anotasi yang konsisten. Simpan label sebagai `positive`, `neutral`, atau `negative`; catat cara menangani ketidaksepakatan anotator.
5. Simpan CSV di `data/raw/reddit_posts.csv`, jalankan pipeline, lalu tinjau metrik dan distribusi hasil.

**Catatan metodologis:** pipeline melatih dan mengevaluasi model secara terpisah untuk tiap game pada data berlabelnya. Nilai sentimen komunitas diringkas dari label data yang tersedia, bukan prediksi model atas data tak berlabel. Metrik antar-game hanya layak dibandingkan jika proses sampling, pedoman anotasi, ukuran data, dan pembagian train/test sebanding. Hasil bukan bukti bahwa game menyebabkan sentimen tertentu.

## Struktur

```text
data/
  raw/                  # Dataset lokal; tidak masuk Git
  processed/            # Data hasil prapemrosesan lokal; tidak masuk Git
reports/                # Ringkasan hasil CSV dan JSON
src/
  sentiment_analysis.py # Pipeline TF-IDF + Multinomial Naive Bayes
tests/                  # Tes otomatis
```

## Persiapan

Gunakan Python 3.10 atau yang lebih baru.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Jika aktivasi environment dibatasi kebijakan PowerShell, jalankan executable Python dari `.venv` secara langsung.

## Format dataset

CSV wajib memiliki kolom:

| Kolom | Isi |
|---|---|
| `game` | `cs2` atau `valorant` |
| `text` | Teks unggahan/komentar yang dianonimkan |
| `sentiment` | `positive`, `neutral`, atau `negative` |

Kolom metadata tambahan diperbolehkan, misalnya `post_id`, `created_at`, `subreddit`, dan `source_type`. Jangan menyertakan nama pengguna atau informasi pribadi yang tidak diperlukan. Ketiga kelas sentimen harus tersedia pada tiap game dan masing-masing memerlukan sedikitnya dua contoh; jumlah minimum ini belum tentu cukup untuk ukuran test tertentu. Untuk split default 80:20, gunakan sekurangnya lima contoh per kelas sebagai titik awal, lalu pastikan ukuran train dan test dapat memuat semua kelas.

Contoh header:

```csv
post_id,game,text,sentiment,created_at,subreddit
```

## Menjalankan analisis

```powershell
python -m src.sentiment_analysis --input data/raw/reddit_posts.csv
```

Secara default, hasil ditulis ke `reports/sentiment_distribution.csv` dan `reports/model_metrics.json`. Opsi yang tersedia:

```powershell
python -m src.sentiment_analysis --input data/raw/reddit_posts.csv --output-dir reports --test-size 0.2 --random-state 42
```

Jalankan tes:

```powershell
python -m unittest discover -s tests -v
```

## Batasan

- Pipeline mengharapkan label yang telah dianotasi; model tidak menghasilkan label ground truth.
- Pembagian acak stratified bukan pemisahan berdasarkan waktu atau penulis. Pertimbangkan potensi kebocoran data jika unit analisis atau duplikasi tidak ditangani.
- TF-IDF dan Naive Bayes adalah baseline; dokumentasikan perubahan bahasa, slang, negasi, sarkasme, dan pilihan prapemrosesan.
- Data dan hasil analisis dapat sensitif. Dataset lokal diabaikan Git; periksa ketentuan platform dan persetujuan/ketentuan etik yang relevan sebelum publikasi.
