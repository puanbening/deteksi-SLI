# deteksi-SLI

## Panduan Penggunaan Streamlit 

Sistem ini menyediakan antarmuka web **Deteksi Dini Specific Language Impairment (SLI)** yang memungkinkan pengguna mengunggah file audio untuk dilakukan skrining awal gangguan bahasa. Pengguna dapat mengunggah file melalui mekanisme **drag and drop** maupun dengan menekan tombol **Upload**.

### Format File yang Didukung

* WAV (.wav)
* MP3 (.mp3)
* Ukuran maksimum file: 200 MB

> **Disclaimer:** Aplikasi ini merupakan alat bantu skrining awal dan bukan pengganti diagnosis klinis. Hasil yang diperoleh sebaiknya dikonsultasikan lebih lanjut kepada tenaga medis atau profesional yang berwenang.

### Keterbatasan Model

Model yang digunakan dalam sistem ini dilatih menggunakan **dataset LANNA**, yang berisi rekaman suara anak-anak penutur bahasa Ceko. Oleh karena itu, model memiliki keterbatasan berupa **bias linguistik**, sehingga performa deteksi dapat menurun ketika digunakan pada audio dari penutur bahasa selain bahasa Ceko.

Perbedaan karakteristik fonologis, intonasi, dan pola akustik antarbahasa dapat memengaruhi kemampuan model dalam mengklasifikasikan kondisi SLI secara akurat. Dengan demikian, hasil prediksi pada audio berbahasa Indonesia maupun bahasa lainnya perlu diinterpretasikan secara hati-hati dan tidak dapat dijadikan sebagai dasar diagnosis klinis.

### Validasi File

Setelah pengguna mengunggah file, sistem secara otomatis melakukan validasi format file. Apabila file yang diunggah tidak sesuai dengan format yang didukung, seperti file video berekstensi `.mp4`, sistem akan menolak file tersebut dan menampilkan indikator kesalahan. Proses prediksi tidak akan dijalankan hingga pengguna mengunggah file audio yang valid.

### Proses Prediksi

Ketika file audio yang diunggah memenuhi ketentuan format, sistem akan secara otomatis memproses audio tanpa memerlukan interaksi tambahan dari pengguna. Selama proses berlangsung, sistem menampilkan pemutar audio (*audio player*) sehingga pengguna dapat memutar ulang file yang telah diunggah.

### Hasil Prediksi

Hasil prediksi ditampilkan dalam dua komponen utama:

* **Skor Prediksi**, yaitu nilai probabilitas antara 0 hingga 1.
* **Label Klasifikasi**, yaitu hasil akhir prediksi yang dihasilkan oleh model.

#### Hasil Sehat

Apabila model mengklasifikasikan audio sebagai kondisi sehat, skor prediksi akan berada mendekati angka **0**, yang menunjukkan tingkat keyakinan model bahwa subjek tidak mengalami gangguan bahasa. Hasil ditampilkan dengan indikator berwarna hijau dan label **"Hasil: Sehat"**, disertai keterangan bahwa audio tidak menunjukkan indikasi *Specific Language Impairment*.

#### Hasil Specific Language Impairment (SLI)

Apabila model mengklasifikasikan audio sebagai kondisi *Specific Language Impairment (SLI)*, skor prediksi akan berada mendekati angka **1**, yang menunjukkan tingkat keyakinan model bahwa subjek terindikasi mengalami gangguan bahasa. Hasil ditampilkan dengan label **"Hasil: Specific Language Impairment (SLI)"** beserta informasi bahwa diperlukan pemeriksaan lebih lanjut oleh tenaga profesional untuk memperoleh diagnosis yang akurat.
