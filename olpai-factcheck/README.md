# OLP AI 2026 – Bài 1: Kiểm chứng tin tức (ViFactCheck)

Phân loại một nhận định (claim) so với bài báo gốc thành `SUPPORTED` / `REFUTED` / `NEI`, chấm bằng macro-F1.
Ràng buộc của đề: **không dùng Transformer hay LLM pretrained**, chỉ dùng thư viện trong danh sách của thể lệ, không học
bất cứ thứ gì trên văn bản test.

| | Macro-F1 |
|---|---|
| Baseline của BTC (validation) | ~0.57 |
| **Lời giải này, CV 5-fold theo bài báo trên train** | **0.7103** |
| Mô phỏng test (giữ 3–5 claim/bài), chọn argmax | 0.716 |
| Mô phỏng test, giải mã chung theo bài | **0.742** |

Lịch sử các thử nghiệm (cái gì tăng điểm, cái gì không) nằm ở [EXPERIMENTS.md](EXPERIMENTS.md).

## Cách tiếp cận

1. **Retriever học có giám sát.** Tách bài báo thành câu; một HistGradientBoosting chấm từng câu bằng 13 đặc trưng khớp
   claim–câu (độ phủ idf, bigram/trigram, cosine fastText, vị trí), nhãn lấy từ cột `evidence` của train. Lấy 2 câu điểm cao
   nhất làm bằng chứng (R@1 0.90 so với 0.86 của TF-IDF).
2. **~170 đặc trưng tay** so claim với bằng chứng và toàn bài: độ phủ idf và n-gram, các phép **diff** claim↔bằng chứng
   (thay thế ngắn ≤4 token là dấu hiệu REFUTED, chèn thêm là dấu hiệu NEI), số / thực thể viết hoa / từ phủ định mới,
   căn chỉnh mềm bằng cosine fastText.
3. **Ba model**
   - HistGradientBoosting trên đặc trưng + xác suất của LR-BoW (stacking lồng, BoW của từ mới / từ bị bỏ / cặp thay thế).
   - Logistic Regression trên đặc trưng chuẩn hoá + BoW.
   - ESIM (BiLSTM + cross-attention) học từ đầu trên (claim, 2 câu bằng chứng); embedding fastText đóng băng + n-gram ký tự;
     pretrain 4 epoch trên claim tổng hợp sinh từ bài báo train (đổi số / thực thể / từ lân cận / phủ định → REFUTED,
     ghép mảnh câu bài khác → NEI, cắt mệnh đề → SUPPORTED), rồi fine-tune trên dữ liệu thật. Trung bình 6 seed.
4. **Trộn** log-xác suất với trọng số 0.4 / 0.4 / 0.2 (chọn bằng CV).
5. **Giải mã chung theo bài.** Bài báo của train hầu hết có đúng 2 claim mỗi nhãn, và test giữ ngẫu nhiên 3–5 claim/bài.
   Với mỗi bài, chọn bộ nhãn tối đa hoá tổng log-xác suất + `LAM` × log P(số nhãn | rút ngẫu nhiên từ bài 2/2/2).
   Không học gì trên test; đây chỉ là prior khi suy luận. Đặt `LAM = 0` để tắt (điểm kỳ vọng ~0.716 thay vì ~0.742).

## Cấu trúc

```
fact_check/                  root repo
├── train.csv, articles.csv, public_test.csv, private_test.csv, private_articles.csv   dữ liệu BTC
├── statement.md, README.md, scorer.py, notebook_mau.ipynb, sample_submission.csv       tài liệu BTC
└── olpai-factcheck/         lời giải
    ├── pipeline.py          mã nguồn duy nhất (các ô ngăn bởi `# %%`)
    ├── solution.ipynb       notebook sinh từ pipeline.py, chạy tuần tự từ trên xuống
    ├── py2ipynb.py          sinh lại solution.ipynb sau khi sửa pipeline.py
    ├── EXPERIMENTS.md       nhật ký thử nghiệm
    └── requirements.txt
```

Không có artifact nào được commit: model, cache, OOF, dữ liệu tổng hợp và bài nộp đều được `pipeline.py` sinh lại.
Thứ duy nhất phải tải là vector fastText.

## Chạy

```
cd olpai-factcheck
pip install -r requirements.txt   # hoặc dùng môi trường của BTC
mkdir -p vec && curl -o vec/cc.vi.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz   # 1.2GB
python pipeline.py                # hoặc chạy solution.ipynb từ đầu đến cuối
```

Dữ liệu được đọc từ thư mục cha (`..`). Có `private_test.csv` thì test = public ∪ private (1.625 claim), không có thì
chỉ public. Output là `bai1_submission.csv` và `submission.zip`, ghi trong `olpai-factcheck/`.

Biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DATA_DIR` | `..` | thư mục chứa dữ liệu BTC |
| `VEC_GZ` | `vec/cc.vi.300.vec.gz` | file vector fastText |
| `RUN_CV` | `0` | `1` = chạy thêm CV 5-fold đầu-cuối trên train (lâu ~5×), lưu `oof_pipeline*.npy` |
| `NO_FINAL` | | đặt bất kỳ giá trị = dừng sau CV, không ghi bài nộp |
| `NN_SEEDS` | `6` | số seed của ESIM |
| `SMOKE` | | đặt bất kỳ giá trị = chạy thử trên lát dữ liệu nhỏ |

Thời gian: khoảng 20 phút cho một lần dự đoán trên 1 GPU RTX 2080 Ti (phần lớn là ESIM và đọc file fastText);
`RUN_CV=1` khoảng 5× lâu hơn. Kết quả có thể lệch nhẹ giữa các lần chạy do tính không tất định của cuDNN.

## Tuân thủ thể lệ

- Không Transformer/LLM pretrained. Thành phần pretrained duy nhất là vector từ fastText `cc.vi.300`, chỉ tra cứu (đóng băng).
  Lưu ý môi trường thi chặn Internet, nên file vector phải có sẵn trên máy chấm.
- Thư viện: numpy, pandas, scipy, scikit-learn, torch (đều trong danh sách). Không dùng xgboost/lightgbm.
- IDF, từ vựng BoW, retriever và mọi model chỉ fit trên train và bài báo của train. Văn bản test chỉ được biến đổi bằng
  những thứ đã học từ train.
- Dữ liệu tổng hợp chỉ sinh từ bài báo của train.
