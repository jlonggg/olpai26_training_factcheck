# Nhật ký thử nghiệm

Mọi con số là macro-F1 trên **CV 5-fold GroupKFold theo bài báo** của `train.csv` (4.434 claim, 625 bài), trừ khi ghi khác.
Không có điểm leaderboard: trang nộp bài đã đóng, nên train-CV là thước đo duy nhất. Độ nhiễu giữa các fold khoảng ±0.01,
và chênh lệch dưới ~0.003 trên toàn bộ OOF được coi là nhiễu.

"Mô phỏng test" = lấy ngẫu nhiên 3–5 claim mỗi bài trong OOF (đúng cách BTC tạo test), lặp 10–20 lần rồi lấy trung bình.

## Kết quả cuối (bản trong repo)

| Thành phần | CV |
|---|---|
| HGB + đặc trưng + LR-BoW (stack) | 0.699 |
| LR trên đặc trưng + BoW | 0.682 |
| ESIM + fastText + pretrain synth | 0.672 |
| **Blend 0.4 / 0.4 / 0.2** | **0.7103** (từng fold 0.6915 / 0.7124 / 0.7081 / 0.7313 / 0.7072) |
| Mô phỏng test, argmax | 0.716 |
| Mô phỏng test, giải mã chung theo bài (`LAM=0.5`) | 0.742 |

## Các bước đã giữ lại (theo thứ tự)

| Bước | Trước → sau | Ghi chú |
|---|---|---|
| Baseline BTC (TF-IDF + 10 đặc trưng + LR) | 0.57 | số của BTC |
| Đặc trưng tay v1 + HGB | 0.661 | độ phủ, n-gram, difflib, số/thực thể/phủ định mới |
| + LR trên BoW (từ mới / từ bị bỏ / claim) xếp chồng vào HGB | 0.661 → 0.670 | OOF lồng bên trong fold |
| + đặc trưng **thay thế ngắn** (cả hai phía ≤4 token), cặp `cũ>mới` vào BoW, w2 theo thứ tự trong bài | HGB 0.661 → 0.678, stack 0.687 | REFUTED thường là thay một cụm ngắn |
| + phủ định nằm trong phần diff | stack 0.690 | trong nhiễu nhưng giữ |
| **Retriever học từ cột `evidence`** | R@1 0.857 → 0.901; HGB 0.679 → 0.688; stack 0.687 → 0.696 | HGB nhị phân trên 13 đặc trưng/câu |
| Chỉnh HGB (lr 0.02, 500 vòng, 7 lá, min_leaf 50) | F1 như cũ, log-loss 0.731 → 0.704; stack 0.699 | xác suất hiệu chỉnh tốt hơn cho blend |
| ESIM học từ đầu | 0.636 | |
| + vector fastText đóng băng | 0.636 → 0.653 → 0.669 (khi dùng retriever học) | |
| + pretrain trên claim tổng hợp (7k) | 0.669 → 0.673 | +0.01 ở 3/5 fold |
| LR trên đặc trưng chuẩn hoá + BoW | 0.682 | thành phần thứ ba cho blend |
| Blend stack + LR + ESIM | 0.714 (trọng số chọn trên OOF), 0.709 (chọn trọng số lồng, trung thực) | |
| Giải mã chung theo bài (prior 2/2/2) | mô phỏng test 0.720 → 0.743 | `LAM` 0.5 tốt nhất trong {0, 0.5, 1, 1.5, 2} |

## Các hướng đã thử và loại bỏ

| Thử nghiệm | Kết quả | Kết luận |
|---|---|---|
| Trần retrieval: thay retriever bằng câu evidence (chỉ đặc trưng, không đặc trưng toàn bài) | 0.730 | nút thắt là bộ phân loại, không phải retrieval |
| Trần retrieval: thay retriever bằng câu evidence (đủ đặc trưng, HGB) | 0.687 → 0.708 | xem dòng retriever v2 |
| Retriever v2: thêm độ phủ mềm fastText + điểm câu lân cận | R@1 0.9008 → 0.9019 | null. Các ca trượt chủ yếu là bằng chứng do người gán nhãn chép thành đoạn nhiều câu, hoặc câu được chọn cũng liên quan không kém; khoảng trống tới trần là lựa chọn của người gán nhãn |
| Đặc trưng căn chỉnh mềm fastText trong HGB | HGB 0.675 → 0.679, stack 0.690 → 0.687 | null |
| Thêm claim tổng hợp vào HGB (trọng số 0.1 / 0.3) | 0.687 → 0.687 / 0.686 | null |
| ESIM + cặp (claim, evidence gold) khi huấn luyện | 0.669 → 0.666 | null |
| Transformer cross-encoder 3 lớp học từ đầu | 0.660; trọng số blend 0 | null |
| ExtraTrees trên đặc trưng | 0.681; trọng số blend 0 | null |
| Synth gấp 3.7× (26k), pretrain 2 hoặc 4 epoch | trung bình 3 fold +0.001 / +0.003 | null |
| Synth v3 (NEI = mệnh đề từ bài cùng chủ đề; SUPPORTED = đảo mệnh đề / ghép 2 câu liền) | trung bình 3 fold −0.001 | null |
| Stacking cấp 2 (LR / HGB trên xác suất OOF, có hoặc không có đặc trưng) | 0.710 / 0.706 / 0.693 | không hơn blend |
| Trung bình nhiều biến thể ESIM | ESIM 0.673 → 0.675, blend không đổi | null |
| Quét C của LR (0.05 / 0.1 / 0.3 / 1) | 0.673 / 0.676 / **0.682** / 0.680 | giữ 0.3 |
| Bias cộng theo lớp tối ưu macro-F1 | +0.0015 | null |
| Prior cặp claim cùng câu bằng chứng (SS 0.14 so với 0.07 khi khác câu) | chỉ 5.5% số cặp | lợi ích kỳ vọng ≤0.001, không làm |
| Rò rỉ theo thứ tự `id` | không có (chuỗi nhãn theo id ngẫu nhiên) | |

## Phân tích lỗi của bản cuối

- F1 theo lớp: SUPPORTED 0.755, REFUTED 0.647, NEI 0.739. **REFUTED yếu nhất** (recall 0.615): 288 ca bị đoán thành
  SUPPORTED, 271 ca thành NEI.
- Lỗi còn lại chủ yếu cần tri thức ngữ nghĩa: trái nghĩa (`cái`↔`đực`, `đã`↔`chưa`), đổi thực thể cùng loại
  (`Bộ Tài chính`↔`Bộ Tư pháp`), đổi quan hệ nhân quả. Không có model ngôn ngữ pretrained thì khó học các quan hệ này từ
  4.4k mẫu. Vector fastText không phân biệt được trái nghĩa với đồng nghĩa.
- Kết luận: cách tiếp cận này bão hoà quanh 0.71 CV. Muốn đi xa hơn cần nguồn tri thức mới, ví dụ văn bản của bài 2/3
  trong contest để pretrain từ đầu.

## Kinh nghiệm vận hành

- `HistGradientBoostingClassifier` dùng mọi core qua OpenMP. Trên máy dùng chung, cần đặt `OMP_NUM_THREADS`, nếu không
  các job tranh nhau và chậm hẳn đi.
- Chọn trọng số blend trên chính OOF lạc quan khoảng +0.005 so với chọn lồng. CV đầu-cuối của pipeline (0.7103) khớp với
  ước lượng lồng (0.709).
